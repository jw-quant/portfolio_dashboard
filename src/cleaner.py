# cleaner.py — parsing utilities for Schwab CSV exports

import re
import pandas as pd


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_schwab_date(val) -> pd.Timestamp:
    """
    Parse a Schwab date cell.

    Handles two formats:
      "04/09/2026"                      → normal trade date
      "09/02/2025 as of 08/29/2025"    → take the FIRST date (trade date)
    """
    s = str(val).strip()
    s = s.split(" as of ")[0].strip()
    return pd.to_datetime(s, format="%m/%d/%Y")


def parse_balance_date(val) -> pd.Timestamp:
    """
    Parse a Schwab balance CSV date: "4/15/2026" (no zero-padding).
    """
    return pd.to_datetime(str(val).strip(), format="%m/%d/%Y")


# ---------------------------------------------------------------------------
# Amount parsing
# ---------------------------------------------------------------------------

_AMOUNT_RE = re.compile(r"[^0-9.\-]")  # keep only digits, dot, minus


def parse_amount(val) -> float:
    """
    Parse a Schwab dollar amount into a plain float.

    Examples:
      "-$33,114.13"  →  -33114.13   (negative: minus before $)
      "$1,000.00"    →   1000.0
      ""             →      0.0
      "--"           →      0.0
    """
    s = str(val).strip()
    if not s or s in ("--", "nan", "None"):
        return 0.0
    # Regex keeps digits, decimal point, and minus sign — sign is preserved
    clean = _AMOUNT_RE.sub("", s)
    if not clean or clean == "-":
        return 0.0
    try:
        return float(clean)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Quantity parsing
# ---------------------------------------------------------------------------

def parse_quantity(val) -> float:
    """
    Parse a Schwab quantity cell.

    Examples:
      "4"        →    4.0
      "31,080"   → 31080.0
      "-10"      →  -10.0
      ""         →   NaN
    """
    s = str(val).strip()
    if not s or s in ("--", "nan", "None"):
        return float("nan")
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Clean transactions DataFrame
# ---------------------------------------------------------------------------

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and type-cast a raw transactions DataFrame loaded from Schwab CSV.

    Input columns (all strings): Date, Action, Symbol, Description,
                                  Quantity, Price, Fees & Comm, Amount, account

    Output adds:
      date     : pd.Timestamp
      action   : str  (stripped)
      symbol   : str  (stripped, '' if blank)
      quantity : float (NaN if blank)
      price    : float (0.0 if blank)
      fees     : float (0.0 if blank)
      amount   : float (0.0 if blank)
    """
    out = df.copy()
    out["date"]     = out["Date"].apply(parse_schwab_date)
    out["action"]   = out["Action"].str.strip()
    out["symbol"]   = out["Symbol"].str.strip().fillna("")
    out["quantity"] = out["Quantity"].apply(parse_quantity)
    out["price"]    = out["Price"].apply(parse_amount)
    out["fees"]     = out["Fees & Comm"].apply(parse_amount)
    out["amount"]   = out["Amount"].apply(parse_amount)
    # Drop original raw columns, keep cleaned ones + account
    keep = ["date", "account", "action", "symbol", "Description",
            "quantity", "price", "fees", "amount"]
    out = out[keep].rename(columns={"Description": "description"})
    return out.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Clean balances DataFrame
# ---------------------------------------------------------------------------

def clean_balances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and type-cast a raw balances DataFrame.

    Input columns (strings): Date, Amount, account
    Output: date (Timestamp), account (str), nav (float)
    """
    out = df.copy()
    out["date"] = out["Date"].apply(parse_balance_date)
    out["nav"]  = out["Amount"].apply(parse_amount)
    return out[["date", "account", "nav"]].sort_values("date").reset_index(drop=True)
