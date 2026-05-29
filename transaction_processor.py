"""
transaction_processor.py
------------------------
Part 1: Core data loading, storage, and analytics.

Data Structure Design:
- Each transaction is stored as a Transaction dataclass (immutable, lightweight).
- TransactionStore uses a dict[ticker -> list[Transaction]] for O(1) ticker lookup.
- Within each ticker's list, transactions are kept sorted by timestamp for time-based queries.
- A secondary dict[trader_id -> list[Transaction]] enables fast trader lookups.
- A flat sorted list (by timestamp) supports global time-series analysis.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transaction:
    """Immutable record representing a single trade."""
    timestamp: datetime
    ticker: str
    action: str          # "BUY" or "SELL"
    quantity: int
    price: float
    trader_id: str       # "UNKNOWN" when the source data has no trader_id
    trade_value: float = field(init=False)  # quantity * price, derived

    def __post_init__(self):
        # frozen=True means we can't assign directly; use object.__setattr__
        object.__setattr__(self, "trade_value", self.quantity * self.price)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "ticker": self.ticker,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "trader_id": self.trader_id,
            "trade_value": self.trade_value,
        }


# ---------------------------------------------------------------------------
# Storage and retrieval
# ---------------------------------------------------------------------------

class TransactionStore:
    """
    Stores and retrieves transactions efficiently.

    Internal indexes:
    - _by_ticker  : dict[ticker  -> list[Transaction]]  sorted by timestamp
    - _by_trader  : dict[trader_id -> list[Transaction]] sorted by timestamp
    - _all_sorted : list[Transaction] sorted by timestamp (global view)
    """

    def __init__(self):
        self._by_ticker: dict[str, list[Transaction]] = defaultdict(list)
        self._by_trader: dict[str, list[Transaction]] = defaultdict(list)
        self._all_sorted: list[Transaction] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_csv(self, filepath: str | Path) -> "DataLoadReport":
        """
        Read a CSV file into the store.
        Returns a DataLoadReport describing what was loaded and cleaned.
        """
        filepath = Path(filepath)
        raw_rows: list[dict] = []
        with filepath.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows.append(row)

        report = DataLoadReport(total_raw=len(raw_rows))

        seen: set[tuple] = set()          # for deduplication
        transactions: list[Transaction] = []

        for row in raw_rows:
            # --- Parse timestamp ---
            try:
                ts = datetime.fromisoformat(row["timestamp"])
            except (ValueError, KeyError):
                report.bad_timestamp += 1
                continue

            # --- Validate action ---
            action = row.get("action", "").strip().upper()
            if action not in ("BUY", "SELL"):
                report.bad_action += 1
                continue

            # --- Parse quantity ---
            try:
                quantity = int(row["quantity"])
                if quantity <= 0:
                    raise ValueError("non-positive quantity")
            except (ValueError, KeyError):
                report.bad_quantity += 1
                continue

            # --- Parse price ---
            try:
                price = float(row["price"])
                if price <= 0:
                    raise ValueError("non-positive price")
            except (ValueError, KeyError):
                report.bad_price += 1
                continue

            # --- Handle missing trader_id ---
            raw_trader = row.get("trader_id", "").strip()
            trader_id = raw_trader if raw_trader else "UNKNOWN"
            if not raw_trader:
                report.missing_trader_id += 1

            ticker = row.get("ticker", "").strip().upper()
            if not ticker:
                report.bad_ticker += 1
                continue

            # --- Deduplicate: exact match on all six logical fields ---
            key = (ts, ticker, action, quantity, price, trader_id)
            if key in seen:
                report.duplicates_removed += 1
                continue
            seen.add(key)

            txn = Transaction(
                timestamp=ts,
                ticker=ticker,
                action=action,
                quantity=quantity,
                price=price,
                trader_id=trader_id,
            )
            transactions.append(txn)

        # Sort once, then insert into indexes
        transactions.sort(key=lambda t: t.timestamp)
        for txn in transactions:
            self._by_ticker[txn.ticker].append(txn)
            self._by_trader[txn.trader_id].append(txn)
        self._all_sorted = transactions

        report.loaded = len(transactions)
        return report

    # ------------------------------------------------------------------
    # Retrieval API
    # ------------------------------------------------------------------

    def get_by_ticker(self, ticker: str) -> list[Transaction]:
        """Return all transactions for a ticker, sorted by time. O(1) lookup."""
        return self._by_ticker.get(ticker.upper(), [])

    def get_by_trader(self, trader_id: str) -> list[Transaction]:
        """Return all transactions for a trader, sorted by time."""
        return self._by_trader.get(trader_id, [])

    def get_all(self) -> list[Transaction]:
        """Return all transactions sorted by timestamp."""
        return self._all_sorted

    def get_in_timerange(
        self,
        start: datetime,
        end: datetime,
        ticker: Optional[str] = None,
    ) -> list[Transaction]:
        """
        Return transactions within [start, end].
        Optionally filter to a single ticker.
        Uses already-sorted lists and a simple linear scan.
        For much larger datasets this could be upgraded to binary search via bisect.
        """
        source = self.get_by_ticker(ticker) if ticker else self._all_sorted
        return [t for t in source if start <= t.timestamp <= end]

    def tickers(self) -> list[str]:
        return sorted(self._by_ticker.keys())

    def traders(self) -> list[str]:
        return sorted(self._by_trader.keys())

    def __len__(self) -> int:
        return len(self._all_sorted)


# ---------------------------------------------------------------------------
# Load report
# ---------------------------------------------------------------------------

@dataclass
class DataLoadReport:
    total_raw: int = 0
    loaded: int = 0
    duplicates_removed: int = 0
    missing_trader_id: int = 0
    bad_timestamp: int = 0
    bad_action: int = 0
    bad_quantity: int = 0
    bad_price: int = 0
    bad_ticker: int = 0

    def __str__(self) -> str:
        issues = []
        if self.duplicates_removed:
            issues.append(f"{self.duplicates_removed} duplicate rows removed")
        if self.missing_trader_id:
            issues.append(f"{self.missing_trader_id} rows with missing trader_id (set to 'UNKNOWN')")
        if self.bad_timestamp:
            issues.append(f"{self.bad_timestamp} rows with unparseable timestamp (skipped)")
        if self.bad_action:
            issues.append(f"{self.bad_action} rows with invalid action (skipped)")
        if self.bad_quantity:
            issues.append(f"{self.bad_quantity} rows with invalid quantity (skipped)")
        if self.bad_price:
            issues.append(f"{self.bad_price} rows with invalid price (skipped)")
        if self.bad_ticker:
            issues.append(f"{self.bad_ticker} rows with missing ticker (skipped)")
        issues_str = "; ".join(issues) if issues else "none"
        return (
            f"Loaded {self.loaded}/{self.total_raw} transactions. "
            f"Data issues: {issues_str}."
        )


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class Analytics:
    """
    Computes statistics from a populated TransactionStore.
    All methods return plain dicts/lists — easy to pass to dashboard or LLM.
    """

    def __init__(self, store: TransactionStore):
        self.store = store

    # --- Volume ---

    def volume_per_ticker(self) -> dict[str, int]:
        """Total shares traded (buys + sells) per ticker."""
        result = {}
        for ticker in self.store.tickers():
            txns = self.store.get_by_ticker(ticker)
            result[ticker] = sum(t.quantity for t in txns)
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def trade_value_per_ticker(self) -> dict[str, float]:
        """Total dollar value traded per ticker."""
        result = {}
        for ticker in self.store.tickers():
            txns = self.store.get_by_ticker(ticker)
            result[ticker] = round(sum(t.trade_value for t in txns), 2)
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    # --- Net position ---

    def net_position_per_ticker(self) -> dict[str, int]:
        """
        Net position = total BUY quantity - total SELL quantity.
        Positive = net long, Negative = net short.
        """
        result = {}
        for ticker in self.store.tickers():
            txns = self.store.get_by_ticker(ticker)
            buys = sum(t.quantity for t in txns if t.action == "BUY")
            sells = sum(t.quantity for t in txns if t.action == "SELL")
            result[ticker] = buys - sells
        return result

    # --- Traders ---

    def most_active_traders(self, top_n: int = 10) -> list[dict]:
        """
        Rank traders by transaction count.
        Returns list of dicts with trader_id, count, buy_count, sell_count, total_value.
        """
        result = []
        for trader_id in self.store.traders():
            txns = self.store.get_by_trader(trader_id)
            result.append({
                "trader_id": trader_id,
                "transaction_count": len(txns),
                "buy_count": sum(1 for t in txns if t.action == "BUY"),
                "sell_count": sum(1 for t in txns if t.action == "SELL"),
                "total_value": round(sum(t.trade_value for t in txns), 2),
            })
        result.sort(key=lambda x: x["transaction_count"], reverse=True)
        return result[:top_n]

    # --- Time-based ---

    def transactions_per_hour(self) -> dict[str, int]:
        """Count transactions grouped by hour (ISO format string)."""
        counts: dict[str, int] = defaultdict(int)
        for txn in self.store.get_all():
            hour_key = txn.timestamp.strftime("%Y-%m-%d %H:00")
            counts[hour_key] += 1
        return dict(sorted(counts.items()))

    def busiest_hours(self, top_n: int = 5) -> list[tuple[str, int]]:
        hourly = self.transactions_per_hour()
        return sorted(hourly.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def price_range_per_ticker(self) -> dict[str, dict]:
        """Min, max, and average price per ticker."""
        result = {}
        for ticker in self.store.tickers():
            prices = [t.price for t in self.store.get_by_ticker(ticker)]
            result[ticker] = {
                "min": min(prices),
                "max": max(prices),
                "avg": round(sum(prices) / len(prices), 2),
            }
        return result

    # --- Summary (for LLM context) ---

    def full_summary(self) -> dict:
        """Return all analytics in a single dict for easy serialization."""
        return {
            "total_transactions": len(self.store),
            "tickers": self.store.tickers(),
            "volume_per_ticker": self.volume_per_ticker(),
            "trade_value_per_ticker": self.trade_value_per_ticker(),
            "net_position_per_ticker": self.net_position_per_ticker(),
            "most_active_traders": self.most_active_traders(),
            "busiest_hours": self.busiest_hours(),
            "price_range_per_ticker": self.price_range_per_ticker(),
        }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    store = TransactionStore()
    report = store.load_csv("sample_transactions.csv")
    print(report)

    analytics = Analytics(store)

    print("\n--- Volume per ticker ---")
    for ticker, vol in analytics.volume_per_ticker().items():
        print(f"  {ticker}: {vol:,} shares")

    print("\n--- Net position per ticker ---")
    for ticker, pos in analytics.net_position_per_ticker().items():
        sign = "+" if pos >= 0 else ""
        print(f"  {ticker}: {sign}{pos:,}")

    print("\n--- Top 5 traders ---")
    for t in analytics.most_active_traders(5):
        print(f"  {t['trader_id']}: {t['transaction_count']} txns  "
              f"(B:{t['buy_count']} S:{t['sell_count']})  "
              f"${t['total_value']:,.0f}")

    print("\n--- Busiest hours ---")
    for hour, count in analytics.busiest_hours():
        print(f"  {hour}: {count} transactions")
