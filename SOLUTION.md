---
editor_options: 
  markdown: 
    wrap: 72
---

# Solution Documentation

## Overview

This application processes financial trading data from a CSV file,
computes analytics, and uses an LLM API to generate natural language
insights. It is organised into three decoupled modules: a
`TransactionStore` for data storage and retrieval, an
`InsightsGenerator` for LLM integration, and a `Streamlit` dashboard
that ties both together.

------------------------------------------------------------------------

## Data Structure Choices

### 1. What data structures did you use?

| Structure | Where used |
|----|----|
| `dataclass (frozen=True)` | `Transaction` — one immutable record per trade |
| `dict[str → list[Transaction]]` | `_by_ticker` and `_by_trader` indexes in `TransactionStore` |
| `list[Transaction]` (sorted) | `_all_sorted` — global time-ordered view; also the per-ticker/per-trader lists |
| `set[tuple]` | Deduplication during CSV load — O(1) membership test |
| `dict` | Analytics output, LLM summary payload |

### 2. Why these structures?

**`dict` for ticker/trader indexes** — the primary access pattern is
"give me all trades for AAPL". A dict keyed by ticker gives O(1) lookup
regardless of total transaction count. A linear scan over all 1000 rows
for each query would be O(n).

**Sorted `list` within each ticker** — transactions are appended in
sorted order once at load time (one sort, O(n log n)). All subsequent
time-range queries stay O(k) where k is the number of matching records,
because we iterate an already-ordered list. This is better than
re-sorting on every query.

**`frozen=True` dataclass** — makes `Transaction` hashable and
immutable. Hashability lets us put tuples of field values into a `set`
for fast deduplication. Immutability prevents accidental mutation after
load.

**`set[tuple]` for deduplication** — checking "have I seen this row
before" is O(1) per row with a set, versus O(n) if we scanned a list.

### 3. Trade-offs considered

-   **pandas DataFrame vs. custom store**: pandas would have been faster
    to write, but a custom `TransactionStore` demonstrates explicit data
    structure reasoning, which is the point of the assignment. The
    analytics layer uses plain Python loops over lists, making the logic
    transparent.
-   **`defaultdict(list)` vs plain `dict`**: `defaultdict` avoids an
    `if ticker not in d` guard on every insert — cleaner and slightly
    faster.

------------------------------------------------------------------------

## Data Processing

### 1. How was the CSV processed?

`TransactionStore.load_csv()` reads the file with `csv.DictReader`,
validates each row field by field, and only inserts rows that pass all
checks. Validated rows are sorted by timestamp once before being added
to the indexes.

### 2. Data quality issues found and how they were handled

| Issue | Count | Decision |
|----|----|----|
| Fully duplicate rows (all 6 fields identical) | 30 | Removed. These appear to be double-recorded entries — keeping them would double-count volume and distort analytics. |
| Missing `trader_id` | 51 | Kept the transaction, set `trader_id = "UNKNOWN"`. The trade itself is valid; the missing attribution is a data quality gap worth preserving for analysis. |
| Negative or zero prices/quantities | 0 | Guard is in place but no cases found. |
| Unparseable timestamps | 0 | Guard is in place but no cases found. |
| Invalid action values (not BUY/SELL) | 0 | Guard is in place but no cases found. |

After cleaning: **1,000 transactions** loaded from 1,030 raw rows.

### 3. Assumptions

-   A row is a duplicate if and only if all six fields (timestamp,
    ticker, action, quantity, price, trader_id) are identical. Two
    trades at the same second for the same stock but different traders
    are kept.
-   `trader_id = "UNKNOWN"` transactions are included in all analytics.
    They appear as a single aggregated trader called "UNKNOWN" in trader
    rankings.
-   Prices and quantities are assumed to be in USD and whole shares
    respectively.

------------------------------------------------------------------------

## LLM Integration

### 1. How was the LLM integrated?

`InsightsGenerator` sends three separate prompts to the LLM (patterns,
risk, unusual activity), each prefixed with a compact JSON summary of
the analytics results. The class supports both Anthropic and OpenAI via
a `provider` argument; the caller never touches the raw HTTP layer.

### 2. How are API responses handled?

Each `_call()` invocation returns a plain string. If the API raises an
exception (bad key, network error, rate limit), the exception is caught
in `generate()` and returned as an `InsightResult` with `error` set —
the dashboard then displays an error message instead of crashing.

### 3. Optimisations to reduce API cost

-   **Send aggregated summary, not raw rows** — the JSON payload sent to
    the LLM is \~1 KB (pre-aggregated stats) rather than \~80 KB (all
    1,000 raw transactions). This reduces prompt tokens by roughly 95%.
-   **Rule-based fallback** — if no API key is set, `generate()` returns
    deterministic insights derived from the analytics summary. The
    dashboard is fully functional without any external dependency.
-   **`max_tokens=512` per call** — the prompts ask for concise bullet
    points, so capping output at 512 tokens is sufficient and avoids
    runaway costs.

------------------------------------------------------------------------

## AI Tools Used

**Claude (Anthropic)** — Used as a coding assistant to brainstorm the
architecture, identify potential bugs, debug errors, and improve
documentation. I independently reviewed, tested, and modified the
AI-generated suggestions before including them in the final solution.

------------------------------------------------------------------------

## Challenges & Learnings

### Most challenging part

Deciding on the right data structure trade-offs. A pandas DataFrame
would have been the fastest path to working analytics, but the
assignment explicitly asks for data structure reasoning, so building a
custom `TransactionStore` with explicit indexes required thinking
carefully about access patterns first.

### What I would do differently with more time

-   Add proper binary search (`bisect`) for time-range queries instead
    of a linear scan — relevant if the dataset grew to millions of rows.
-   Persist insights to disk so they survive a page refresh.
-   Add unit tests for each analytics method and the data cleaning
    logic.
-   Support streaming LLM responses in the dashboard so users see output
    as it arrives.

### What I learned

Designing the API (what methods make sense) before writing the
implementation forced clearer thinking about data access patterns than
jumping straight to code.

------------------------------------------------------------------------

## Testing

### How the solution was tested

-   Ran `python transaction_processor.py` as a smoke test after each
    change to verify load counts, deduplication counts, and analytics
    output.
-   Verified the deduplication logic manually against the raw CSV (rows
    2-3 and 5-6 are obvious duplicates visible in the first few rows).
-   Tested the dashboard locally with `streamlit run dashboard.py`.

### Edge cases considered

-   Rows with missing `trader_id` — kept with sentinel value.
-   All transactions for one ticker being SELL-only (net position would
    be negative).
-   The "UNKNOWN" trader bucket appearing in trader rankings.
-   LLM API key not set — handled gracefully through deterministic
    rule-based insights, so the dashboard remains fully usable without
    external credentials.
