"""
dashboard.py
------------
Part 3: Streamlit dashboard for visualising transaction data and LLM insights.

Run with:
    streamlit run dashboard.py

Set your API key before running:
    export ANTHROPIC_API_KEY=your_key_here
    # or
    export OPENAI_API_KEY=your_key_here
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from transaction_processor import Analytics, TransactionStore
from insights_generator import InsightsGenerator

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading (cached so it only runs once per session)
# ---------------------------------------------------------------------------

CSV_PATH = Path(__file__).parent / "sample_transactions.csv"


@st.cache_resource
def load_data():
    store = TransactionStore()
    report = store.load_csv(CSV_PATH)
    analytics = Analytics(store)
    return store, analytics, report


store, analytics, load_report = load_data()
summary = analytics.full_summary()

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")

    provider = st.selectbox("LLM Provider", ["anthropic", "openai"])

    api_key_label = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = st.text_input(
        f"{api_key_label}",
        value=os.environ.get(api_key_label, ""),
        type="password",
        help="Your API key. Never stored — only used in this session.",
    )

    st.divider()
    st.caption("📂 Data load report")
    st.caption(str(load_report))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("📈 Financial Transaction Dashboard")
st.caption(
    f"Analysing **{summary['total_transactions']:,}** transactions across "
    f"**{len(summary['tickers'])}** tickers  |  "
    f"Tickers: {', '.join(summary['tickers'])}"
)

# ---------------------------------------------------------------------------
# Row 1 — KPI cards
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

total_value = sum(summary["trade_value_per_ticker"].values())
top_ticker = list(summary["volume_per_ticker"].keys())[0]
top_trader = summary["most_active_traders"][0]["trader_id"]
top_trader_count = summary["most_active_traders"][0]["transaction_count"]

col1.metric("Total Transactions", f"{summary['total_transactions']:,}")
col2.metric("Total Trade Value", f"${total_value:,.0f}")
col3.metric("Most Traded Ticker", top_ticker)
col4.metric("Most Active Trader", f"{top_trader} ({top_trader_count} txns)")

st.divider()

# ---------------------------------------------------------------------------
# Row 2 — Volume and Net position
# ---------------------------------------------------------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Volume per Ticker (shares)")
    vol_df = pd.DataFrame(
        list(summary["volume_per_ticker"].items()),
        columns=["Ticker", "Volume (shares)"],
    )
    st.bar_chart(vol_df.set_index("Ticker"))

with col_right:
    st.subheader("⚖️ Net Position per Ticker")
    net_df = pd.DataFrame(
        list(summary["net_position_per_ticker"].items()),
        columns=["Ticker", "Net Position"],
    ).sort_values("Net Position")
    st.bar_chart(net_df.set_index("Ticker"))

# ---------------------------------------------------------------------------
# Row 3 — Traders and raw data
# ---------------------------------------------------------------------------

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("🏆 Most Active Traders")
    traders_df = pd.DataFrame(summary["most_active_traders"])
    traders_df.columns = ["Trader", "Txns", "Buys", "Sells", "Total Value ($)"]
    traders_df["Total Value ($)"] = traders_df["Total Value ($)"].map(
        lambda x: f"${x:,.0f}"
    )
    st.dataframe(traders_df, use_container_width=True, hide_index=True)

with col_right2:
    st.subheader("💰 Trade Value per Ticker ($)")
    value_df = pd.DataFrame(
        list(summary["trade_value_per_ticker"].items()),
        columns=["Ticker", "Trade Value ($)"],
    )
    st.bar_chart(value_df.set_index("Ticker"))

# ---------------------------------------------------------------------------
# Row 4 — Transaction activity over time
# ---------------------------------------------------------------------------

st.subheader("🕐 Transaction Activity by Hour")
hourly = summary["busiest_hours"]
hourly_df = pd.DataFrame(hourly, columns=["Hour", "Transaction Count"])
st.bar_chart(hourly_df.set_index("Hour"))

# ---------------------------------------------------------------------------
# Row 5 — Full transaction table
# ---------------------------------------------------------------------------

with st.expander("📋 Raw Transaction Data", expanded=False):
    all_txns = [t.to_dict() for t in store.get_all()]
    raw_df = pd.DataFrame(all_txns)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df["trade_value"] = raw_df["trade_value"].map(lambda x: f"${x:,.2f}")

    # Filters
    ticker_filter = st.multiselect(
        "Filter by Ticker", options=sorted(raw_df["ticker"].unique()), default=[]
    )
    action_filter = st.multiselect(
        "Filter by Action", options=["BUY", "SELL"], default=[]
    )
    filtered = raw_df.copy()
    if ticker_filter:
        filtered = filtered[filtered["ticker"].isin(ticker_filter)]
    if action_filter:
        filtered = filtered[filtered["action"].isin(action_filter)]

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(filtered):,} of {len(raw_df):,} transactions")

# ---------------------------------------------------------------------------
# Row 6 — LLM Insights
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🤖 AI-Generated Insights")
st.caption(
    "Sends aggregated analytics (not raw data) to the LLM — "
    "minimises token usage and API cost."
)

if st.button("✨ Generate Insights", type="primary", use_container_width=False):
    with st.spinner("Generating insights..."):
        gen = InsightsGenerator(provider=provider, api_key=api_key)
        result = gen.generate(summary)

    if result.has_error:
        st.error(f"LLM error: {result.error}")
    else:
        if result.provider == "rule_based":
            st.info(
                "No API key provided — showing **rule-based insights** "
                "instead of LLM-generated ones."
            )
        else:
            st.success(f"Insights generated using **{result.provider} / {result.model}**")

        tab1, tab2, tab3 = st.tabs(
            ["📈 Trading Patterns", "⚠️ Risk Flags", "🚨 Unusual Activity"]
        )
        with tab1:
            st.markdown(result.patterns)
        with tab2:
            st.markdown(result.risks)
        with tab3:
            st.markdown(result.unusual_activity)

        with st.expander("Export full insight text"):
            st.text_area("", value=result.full_text(), height=300)
