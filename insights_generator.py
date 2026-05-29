"""
insights_generator.py
---------------------
Part 2: LLM integration for generating natural language insights.

Design decisions:
- Supports both Anthropic and OpenAI via a unified InsightsGenerator interface.
- The LLM receives a compact JSON summary of analytics — NOT raw rows —
  to minimize token usage and API cost.
- Three focused prompts cover patterns, risks, and unusual activity.
- If no API key is available, falls back to deterministic rule-based insights
  so the dashboard remains fully functional without external dependencies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class InsightResult:
    provider: str
    model: str
    patterns: str
    risks: str
    unusual_activity: str
    error: Optional[str] = None

    @property
    def has_error(self) -> bool:
        return self.error is not None

    def full_text(self) -> str:
        if self.has_error:
            return f"Error generating insights: {self.error}"
        return (
            f"## Trading Pattern Analysis\n{self.patterns}\n\n"
            f"## Concentration & Risk Flags\n{self.risks}\n\n"
            f"## Unusual Activity\n{self.unusual_activity}"
        )


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def generate_rule_based_insights(summary: dict) -> "InsightResult":
    """
    Produce deterministic insights from the analytics summary.
    Used when no LLM API key is available, so the dashboard is always functional.
    """
    net_positions = summary["net_position_per_ticker"]
    top_ticker = next(iter(summary["volume_per_ticker"]))
    top_trader = summary["most_active_traders"][0]
    largest_short = min(net_positions.items(), key=lambda x: x[1])
    largest_long  = max(net_positions.items(), key=lambda x: x[1])
    busiest_hour  = summary["busiest_hours"][0][0] if summary["busiest_hours"] else "N/A"

    patterns = (
        f"- **{top_ticker}** is the most actively traded ticker by share volume.\n"
        f"- **{top_trader['trader_id']}** dominates transaction count with "
        f"{top_trader['transaction_count']} trades "
        f"({top_trader['buy_count']} buys / {top_trader['sell_count']} sells).\n"
        f"- Peak trading activity occurs around **{busiest_hour}**, suggesting "
        f"concentrated intraday patterns."
    )

    risks = (
        f"- Largest net **short** exposure: **{largest_short[0]}** "
        f"({largest_short[1]:,} shares net sold).\n"
        f"- Largest net **long** exposure: **{largest_long[0]}** "
        f"({largest_long[1]:+,} shares net bought).\n"
        f"- {top_trader['trader_id']} accounts for a disproportionate share of total "
        f"activity — concentration risk worth monitoring."
    )

    unusual = (
        f"- **{top_trader['trader_id']}** has materially higher activity than any "
        f"other trader and may warrant further review.\n"
        f"- Missing trader IDs were grouped as **UNKNOWN** — "
        f"this is a data quality flag that should be investigated upstream.\n"
        f"- Exact duplicate rows were removed during ingestion to avoid "
        f"double-counting volume; their origin (system retry, double submission) is unknown."
    )

    return InsightResult(
        provider="rule_based",
        model="deterministic",
        patterns=patterns,
        risks=risks,
        unusual_activity=unusual,
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompts(summary: dict) -> tuple[str, str, str]:
    """Return (patterns_prompt, risks_prompt, unusual_prompt)."""
    context = json.dumps(summary, indent=2)

    base = (
        "You are a financial analyst reviewing a 3-day window of equity trading data. "
        "Below is a JSON summary of transaction analytics.\n\n"
        f"```json\n{context}\n```\n\n"
    )

    patterns = (
        base
        + "Analyze the trading patterns across tickers and traders. "
        "Comment on: which tickers are most actively traded, buy vs sell pressure "
        "per ticker, time-of-day patterns, and any notable trader behaviour. "
        "Be concise (3-5 bullet points)."
    )

    risks = (
        base
        + "Identify concentration risks and potential red flags. "
        "Consider: tickers with heavily net-long or net-short positions, "
        "traders who dominate a single ticker, tickers with extreme price ranges, "
        "and overall portfolio concentration. "
        "Be concise (3-5 bullet points)."
    )

    unusual = (
        base
        + "Flag any unusual or suspicious trading activity. "
        "Look for: traders with abnormally high transaction counts, "
        "unusually large single trades, tickers with sudden volume spikes, "
        "and any patterns that could suggest coordinated trading. "
        "Be concise (3-5 bullet points)."
    )

    return patterns, risks, unusual


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str, api_key: str, model: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _call_openai(prompt: str, api_key: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

class InsightsGenerator:
    """
    Generate LLM insights from an analytics summary dict.

    Usage:
        gen = InsightsGenerator(provider="anthropic")   # reads ANTHROPIC_API_KEY
        result = gen.generate(analytics.full_summary())
        print(result.full_text())
    """

    ANTHROPIC_DEFAULT_MODEL = "claude-3-5-haiku-20241022"
    OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider.lower()
        if self.provider not in ("anthropic", "openai"):
            raise ValueError("provider must be 'anthropic' or 'openai'")

        # API key: explicit arg > environment variable
        if api_key:
            self.api_key = api_key
        elif self.provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")

        # Model selection
        if model:
            self.model = model
        elif self.provider == "anthropic":
            self.model = self.ANTHROPIC_DEFAULT_MODEL
        else:
            self.model = self.OPENAI_DEFAULT_MODEL

        self._cache: Optional[InsightResult] = None

    def _call(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError(
                f"No API key found. Set the "
                f"{'ANTHROPIC_API_KEY' if self.provider == 'anthropic' else 'OPENAI_API_KEY'} "
                f"environment variable."
            )
        if self.provider == "anthropic":
            return _call_anthropic(prompt, self.api_key, self.model)
        else:
            return _call_openai(prompt, self.api_key, self.model)

    def generate(self, summary: dict, force_refresh: bool = False) -> InsightResult:
        """
        Generate insights from an analytics summary dict.

        - If no API key is set, falls back to deterministic rule-based insights.
        - Cost optimisation: sends pre-aggregated summary (~1 KB) rather than
          all 1000+ raw transactions, reducing token usage by ~95%.
        """
        if self._cache and not force_refresh:
            return self._cache

        # Graceful degradation: no key → rule-based fallback
        if not self.api_key:
            return generate_rule_based_insights(summary)

        patterns_prompt, risks_prompt, unusual_prompt = _build_prompts(summary)

        try:
            patterns = self._call(patterns_prompt)
            risks = self._call(risks_prompt)
            unusual = self._call(unusual_prompt)
        except Exception as exc:
            result = InsightResult(
                provider=self.provider,
                model=self.model,
                patterns="",
                risks="",
                unusual_activity="",
                error=str(exc),
            )
            return result

        result = InsightResult(
            provider=self.provider,
            model=self.model,
            patterns=patterns,
            risks=risks,
            unusual_activity=unusual,
        )
        self._cache = result
        return result


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transaction_processor import TransactionStore, Analytics

    store = TransactionStore()
    store.load_csv("sample_transactions.csv")
    summary = Analytics(store).full_summary()

    gen = InsightsGenerator(provider="anthropic")
    result = gen.generate(summary)

    if result.has_error:
        print("Error:", result.error)
    else:
        print(result.full_text())
