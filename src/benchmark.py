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

    Reads from data/raw/prices/<TICKER>.csv when the file already covers the
    requested window.  Otherwise fetches from Polygon.io via src-core's
    PolygonClient, saves the full response to CSV, then returns the slice.

    Requires POLYGON_API_KEY in portfolio_dashboard/.env (or env variable).

    Returns
    -------
    pd.Series  daily returns (pct_change of adj_close_total_return),
               date-indexed as pd.Timestamp, name=ticker.upper().
               NaN on first day (no prior close available).
    """
    prices_dir = Path(raw_dir, "prices")
    prices_dir.mkdir(parents=True, exist_ok=True)
    csv_path = prices_dir / f"{ticker.upper()}.csv"

    df = _load_csv_if_covers(csv_path, start_date, end_date)

    if df is None:
        df = _fetch_from_polygon(ticker, start_date, end_date, prices_dir, csv_path)

    # Slice to requested window
    df = df[
        (df["date"] >= pd.Timestamp(start_date))
        & (df["date"] <= pd.Timestamp(end_date))
    ].copy()
    df = df.sort_values("date").reset_index(drop=True)

    price_col = (
        "adj_close_total_return"
        if "adj_close_total_return" in df.columns
        else "close"
    )
    prices = df.set_index("date")[price_col]
    daily_rets = prices.pct_change()
    daily_rets.name = ticker.upper()
    return daily_rets


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_csv_if_covers(
    csv_path: Path, start: str, end: str
) -> "pd.DataFrame | None":
    """Return DataFrame if CSV exists and fully covers [start, end]; else None."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        if df.empty:
            return None
        covers = (
            df["date"].min() <= pd.Timestamp(start)
            and df["date"].max() >= pd.Timestamp(end)
        )
        return df if covers else None
    except Exception:
        return None


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
    csv_path: Path,
) -> pd.DataFrame:
    """Fetch OHLC + dividends from Polygon, save full CSV, return DataFrame."""
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
        from src.market.polygon import PolygonClient
    except ImportError as exc:
        raise ImportError(
            f"Cannot import src-core from {_SRC_CORE}.\n"
            "Make sure the src-core repo is at the expected sibling path."
        ) from exc

    client = PolygonClient(data_dir=prices_dir, rate_limit_secs=12)
    raw = client.fetch_range_ohlc(ticker, start, end)
    if raw.empty:
        raise RuntimeError(
            f"Polygon returned 0 rows for {ticker} [{start}..{end}]"
        )

    div = client.fetch_range_dividends(ticker, start, end)
    df = PolygonClient.apply_total_return_adjustment(raw, div)
    df.to_csv(csv_path, index=False)
    print(f"[benchmark] {ticker}: {len(df)} rows saved -> {csv_path.name}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
