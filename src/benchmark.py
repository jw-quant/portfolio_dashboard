# benchmark.py — SPY benchmark data loader via src-core / Polygon

import os
import sys
import warnings
from pathlib import Path

import pandas as pd

# Path to the sibling src-core repo
_SRC_CORE = Path(__file__).resolve().parents[2] / "src-core"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_benchmark_returns(
    raw_dir: str,
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Return daily total-return series for `ticker` over [start_date, end_date].

    Delegates to src-core's fetch_range(ticker, start, end), which reads from
    the cached CSV when it already covers the window, or fetches only what's
    missing (extending forward, or backfilling if start predates the cache)
    and merges it in. Never overwrites cached history outside the requested
    window — the previous hand-rolled cache check here did, if the requested
    window didn't happen to fully cover what was already cached.

    Requires POLYGON_API_KEY in portfolio_dashboard/.env (or env variable).

    Returns
    -------
    pd.Series  daily returns (pct_change of adj_close),
               date-indexed as pd.Timestamp, name=ticker.upper().
               NaN on first day (no prior close available).
    """
    # Use shared DATA_ROOT prices folder if set; fall back to raw_dir/prices
    data_root = os.getenv("DATA_ROOT")
    if data_root:
        prices_dir = Path(data_root) / "data" / "prices"
    else:
        prices_dir = Path(raw_dir, "prices")
    prices_dir.mkdir(parents=True, exist_ok=True)

    df = _fetch_from_polygon(ticker, start_date, end_date, prices_dir)

    # Slice to requested window
    df = df[
        (df["date"] >= pd.Timestamp(start_date))
        & (df["date"] <= pd.Timestamp(end_date))
    ].copy()
    df = df.sort_values("date").reset_index(drop=True)

    price_col = (
        "adj_close"
        if "adj_close" in df.columns
        else "close"
    )
    prices = df.set_index("date")[price_col]
    daily_rets = prices.pct_change()
    daily_rets.name = ticker.upper()
    return daily_rets


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_load_env() -> None:
    """Load POLYGON_API_KEY from .env in portfolio_dashboard root or src-core root."""
    candidates = [
        Path(__file__).resolve().parents[1] / ".env",  # portfolio_dashboard/.env
        _SRC_CORE / ".env",                             # src-core/.env
    ]
    for path in candidates:
        if path.exists():
            _parse_dotenv(path)
            return


def _parse_dotenv(path: Path) -> None:
    """Minimal .env parser — sets env vars that aren't already set."""
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
    except OSError:
        pass


def _fetch_from_polygon(
    ticker: str,
    start: str,
    end: str,
    prices_dir: Path,
) -> pd.DataFrame:
    """Ensure the cached CSV covers [start, end] via src-core's fetch_range,
    fetching only what's missing, and return the (possibly cached) DataFrame."""
    _try_load_env()

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key or api_key.startswith("your_polygon"):
        raise RuntimeError(
            "POLYGON_API_KEY not found.\n"
            "Create portfolio_dashboard/.env with:\n"
            "  POLYGON_API_KEY=<your_key>\n"
            "Free keys available at https://polygon.io"
        )

    if str(_SRC_CORE) not in sys.path:
        sys.path.insert(0, str(_SRC_CORE))

    try:
        from src.market import fetch_range
    except ImportError as exc:
        raise ImportError(
            f"Cannot import src-core from {_SRC_CORE}.\n"
            "Make sure the src-core repo is at the expected sibling path."
        ) from exc

    df = fetch_range(ticker, start, end, data_dir=prices_dir)
    if df.empty:
        raise RuntimeError(
            f"Polygon returned 0 rows for {ticker} [{start}..{end}]"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
