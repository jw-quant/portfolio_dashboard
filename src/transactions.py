# transactions.py — external cash flow classification for Schwab transactions

import pandas as pd

# ---------------------------------------------------------------------------
# External cash flow actions
#
# These are Schwab action strings that represent money crossing the account
# boundary from/to the outside world.  At the ACCOUNT level, every entry
# here is a cash flow.  In a COMBINED multi-account view, transfers between
# the selected accounts net to zero automatically when daily CFs are summed.
#
# NOT included: buys/sells, dividends, interest, DRIP, option expirations,
# stock splits, CD lifecycle, mergers — those are all internal investment
# activity that doesn't cross the boundary.
# ---------------------------------------------------------------------------

_EXTERNAL_CF_ACTIONS = frozenset({
    # Bank / electronic transfers
    "moneylink transfer",
    "wire sent",
    "wire received",
    "wire funds adj",
    "funds received",
    "funds paid",
    # Cash access
    "atm withdrawal",
    "schwab atm rebate",
    "visa purchase",
    "visa credit",
    # Fees and misc cash (affect NAV but not investment return)
    "service fee",
    "misc cash entry",
    "misc credits",
    # Automatic debits/credits (tax payments, bank ACH)
    "auto s1 debit",
    "auto s1 credit",
    # Inter-account journals
    # At the account level these are real flows; in a combined view the
    # offsetting legs naturally cancel when summed by date.
    "journal",
})


def is_external_cashflow(action: str) -> bool:
    """Return True if this Schwab action is an external cash flow."""
    return str(action).strip().lower() in _EXTERNAL_CF_ACTIONS


def tag_cashflows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an `is_external_cf` boolean column to a cleaned transactions DataFrame.
    """
    df = df.copy()
    df["is_external_cf"] = df["action"].apply(is_external_cashflow)
    return df


def daily_cashflows(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate external cash flows to a daily net series.

    Parameters
    ----------
    df : cleaned + tagged transactions DataFrame for ONE account
         (must have columns: date, amount, is_external_cf)

    Returns
    -------
    pd.Series indexed by date, values = net external cash flow that day
    (positive = inflow into account, negative = outflow from account)
    """
    cf = df[df["is_external_cf"]].copy()
    if cf.empty:
        return pd.Series(dtype=float, name="cashflow")
    daily = cf.groupby("date")["amount"].sum()
    daily.name = "cashflow"
    return daily
