# config.py — account registry and project-wide defaults

# ---------------------------------------------------------------------------
# Account registry
# Add a new account here when you have its balance + transaction files.
# Pattern strings are matched against filenames in data/raw/balances/ and
# data/raw/transactions/ respectively.
# ---------------------------------------------------------------------------

ACCOUNTS = {
    "Retirement": {
        "label":                "Retirement",
        "transactions_pattern": "*348*",
        "balances_pattern":     "*348*",
    },
    "Trade_and_leaps": {
        "label":                "Trade & Leaps",
        "transactions_pattern": "*887*",
        "balances_pattern":     "*887*",
    },
}

# ---------------------------------------------------------------------------
# Folder paths (relative to project root)
# ---------------------------------------------------------------------------
RAW_DATA_DIR       = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
REPORTS_OUTPUT_DIR = "reports"

# ---------------------------------------------------------------------------
# Performance defaults
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.0

# Benchmark ticker — not used in v1; plug in later
DEFAULT_BENCHMARK = "SPY"
