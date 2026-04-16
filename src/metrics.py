# metrics.py — time-weighted return engine and performance metrics

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Core return engine
# ---------------------------------------------------------------------------

def _settle_cashflows(nav_vals: np.ndarray, cf_arr: np.ndarray) -> np.ndarray:
    """
    Adjust CF dates to avoid mathematically impossible returns (R < -1).

    When a CF is dated on day t but the balance doesn't fully reflect it
    until day t+1 (common with Schwab journals and same-day MoneyLink
    transfers), the Modified Dietz formula produces R_t < -1, which
    permanently corrupts the cumulative wealth index.

    Fix: assume 0% investment return on the problematic day and shift
    the excess CF (the portion not explained by the NAV change) to the
    next day, where the balance actually reflects it.
    """
    cf = cf_arr.copy()
    n = len(nav_vals)
    for i in range(1, n):
        c = cf[i]
        if abs(c) < 1e-6:
            continue
        prev = nav_vals[i - 1]
        denom = prev + 0.5 * c
        if denom < 1e-6:
            skip_val = True
        else:
            r = (nav_vals[i] - prev - c) / denom
            skip_val = r < -1.0
        if skip_val:
            # Assume 0% investment return: all of delta_NAV is settled CF
            settled = nav_vals[i] - prev
            excess  = c - settled
            cf[i]   = settled
            if i + 1 < n:
                cf[i + 1] += excess
            # else: last day — absorb the excess (edge case)
    return cf


def compute_daily_returns(nav: pd.Series, cashflows: pd.Series) -> pd.Series:
    """
    Compute daily Modified Dietz returns adjusted for external cash flows.

    Formula (mid-day flow assumption):
        R_t = (NAV_t - NAV_{t-1} - CF_t) / (NAV_{t-1} + 0.5 * CF_t)

    Parameters
    ----------
    nav        : pd.Series, date-indexed daily portfolio values
    cashflows  : pd.Series, date-indexed daily net external cash flows
                 (positive = inflow, negative = outflow)
                 Dates not in nav index are ignored; missing dates → 0.

    Returns
    -------
    pd.Series of daily returns (NaN for first day and any invalid day)
    """
    cf_raw = cashflows.reindex(nav.index, fill_value=0.0)

    # Adjust CF dates for settlement timing issues (see _settle_cashflows)
    cf_adj  = _settle_cashflows(nav.values, cf_raw.values)
    cf      = pd.Series(cf_adj, index=nav.index)

    nav_prev  = nav.shift(1)
    numerator = nav - nav_prev - cf
    denom     = nav_prev + 0.5 * cf

    returns = pd.Series(np.nan, index=nav.index, dtype=float)
    valid   = denom.notna() & (denom.abs() > 1e-6) & nav_prev.notna()
    returns[valid] = numerator[valid] / denom[valid]
    return returns


def cumulative_return(daily_returns: pd.Series) -> pd.Series:
    """
    Chain daily returns into a cumulative wealth index starting at 1.0.
    NaN days are treated as zero-return days (no change).
    """
    r = daily_returns.fillna(0.0)
    return (1 + r).cumprod()


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def total_return(daily_returns: pd.Series) -> float:
    """Total TWR over the full period."""
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    return (1 + r).prod() - 1


def annualized_return(daily_returns: pd.Series, nav: pd.Series) -> float:
    """
    CAGR using actual calendar days elapsed (not trading day count).
    """
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    days = (nav.index[-1] - nav.index[0]).days
    if days <= 0:
        return float("nan")
    n_years = days / 365.25
    total = (1 + r).prod()
    return total ** (1.0 / n_years) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns (trading-day basis)."""
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    return r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns: pd.Series,
                 risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio."""
    r    = daily_returns.dropna()
    ann  = r.mean() * TRADING_DAYS_PER_YEAR - risk_free_rate
    vol  = r.std()  * np.sqrt(TRADING_DAYS_PER_YEAR)
    return ann / vol if vol > 1e-10 else float("nan")


def max_drawdown(daily_returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown on the investment return curve.
    Returns a negative decimal, e.g. -0.25 means -25%.
    """
    cum = cumulative_return(daily_returns)
    rolling_peak = cum.cummax()
    dd = (cum - rolling_peak) / rolling_peak
    return dd.min()


def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    """Full drawdown time series for plotting."""
    cum = cumulative_return(daily_returns)
    rolling_peak = cum.cummax()
    return (cum - rolling_peak) / rolling_peak


def monthly_returns(daily_returns: pd.Series) -> pd.Series:
    """
    Compound daily returns to monthly returns.
    Returns pd.Series indexed by month-end dates.
    """
    r = daily_returns.fillna(0.0)
    # group by year-month, compound within each month
    monthly = r.groupby(pd.Grouper(freq="ME")).apply(lambda x: (1 + x).prod() - 1)
    return monthly


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def benchmark_metrics_summary(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Partial metrics for a benchmark index (return-based rows only).

    Returns a single-column DataFrame with the same row labels as
    metrics_summary() so the two can be joined side-by-side.
    Rows that don't apply to a benchmark are filled with "—".
    """
    r = daily_returns.dropna()
    if len(r) < 2:
        total = float("nan")
        ann   = float("nan")
    else:
        days    = (r.index[-1] - r.index[0]).days
        n_years = days / 365.25 if days > 0 else float("nan")
        total   = float((1 + r).prod() - 1)
        ann     = (float(total) + 1) ** (1.0 / n_years) - 1 if n_years else float("nan")

    rows = {
        "Start date":                  "—",
        "End date":                    "—",
        "Start NAV ($)":               "—",
        "End NAV ($)":                 "—",
        "Net external cash flows ($)": "—",
        "Net investment gain ($)":     "—",
        "Total return (TWR)":          f"{total:.2%}" if not pd.isna(total) else "—",
        "Annualized return":           f"{ann:.2%}"   if not pd.isna(ann)   else "—",
        "Annualized volatility":       f"{annualized_volatility(daily_returns):.2%}",
        "Sharpe ratio":                f"{sharpe_ratio(daily_returns, risk_free_rate):.2f}",
        "Max drawdown":                "—",
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Benchmark"])


def metrics_summary(daily_returns: pd.Series,
                    nav: pd.Series,
                    cashflows: pd.Series,
                    risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Return a one-column DataFrame of key performance metrics.
    """
    total_cf = cashflows.sum()
    end_nav  = nav.iloc[-1] if not nav.empty else float("nan")
    start_nav = nav.iloc[0] if not nav.empty else float("nan")
    net_gain = end_nav - start_nav - total_cf

    rows = {
        "Start date":             str(nav.index[0].date()) if not nav.empty else "—",
        "End date":               str(nav.index[-1].date()) if not nav.empty else "—",
        "Start NAV ($)":          f"{start_nav:,.2f}",
        "End NAV ($)":            f"{end_nav:,.2f}",
        "Net external cash flows ($)": f"{total_cf:,.2f}",
        "Net investment gain ($)":     f"{net_gain:,.2f}",
        "Total return (TWR)":     f"{total_return(daily_returns):.2%}",
        "Annualized return":      f"{annualized_return(daily_returns, nav):.2%}",
        "Annualized volatility":  f"{annualized_volatility(daily_returns):.2%}",
        "Sharpe ratio":           f"{sharpe_ratio(daily_returns, risk_free_rate):.2f}",
        "Max drawdown":           f"{max_drawdown(daily_returns):.2%}",
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Value"])
