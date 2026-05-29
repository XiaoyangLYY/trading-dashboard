# Financial Transaction Analyser

A Python application that processes financial transaction data, analyses trading patterns, and uses an LLM to generate insights.

## Setup

``` bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Set your LLM API key for AI-generated insights
export ANTHROPIC_API_KEY=your_key_here
# or
export OPENAI_API_KEY=your_key_here

# 3. Make sure sample_transactions.csv is in the same directory

# 4. Run the dashboard
streamlit run dashboard.py
```

> **No API key?** The dashboard runs fully without one. Clicking "Generate Insights" will return deterministic rule-based insights derived from the analytics data.

## Project Structure

```         
.
├── transaction_processor.py   # Part 1: data loading, storage, analytics
├── insights_generator.py      # Part 2: LLM integration
├── dashboard.py               # Part 3: Streamlit dashboard
├── sample_transactions.csv    # Input data
├── requirements.txt
├── README.md
└── SOLUTION.md
```

## Running individual modules

``` bash
# Test data loading and analytics only
python transaction_processor.py

# Test insight generation (uses API key if available, otherwise rule-based fallback)
python insights_generator.py
```

## Switching LLM providers

The dashboard has a dropdown in the sidebar. You can also instantiate directly:

``` python
from insights_generator import InsightsGenerator

# Anthropic
gen = InsightsGenerator(provider="anthropic")  # reads ANTHROPIC_API_KEY

# OpenAI
gen = InsightsGenerator(provider="openai")     # reads OPENAI_API_KEY
```
