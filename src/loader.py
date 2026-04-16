# loader.py — file discovery and raw CSV loading

import csv
import io
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_file(directory: Path, pattern: str) -> Path:
    """
    Return the most-recently-modified file in `directory` matching `pattern`.
    Raises FileNotFoundError if nothing matches.
    """
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found in {directory}\n"
            f"Place the Schwab export there and try again."
        )
    return matches[-1]


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

def load_transactions(raw_dir: str, account_key: str, pattern: str) -> pd.DataFrame:
    """
    Load a Schwab transactions CSV for one account.
    Returns a raw DataFrame with an added 'account' column.
    All columns are kept as strings; cleaning happens in cleaner.py.
    """
    path = _find_file(Path(raw_dir, "transactions"), pattern)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["account"] = account_key
    return df


# ---------------------------------------------------------------------------
# Balances
# ---------------------------------------------------------------------------

def load_balances(raw_dir: str, account_key: str, pattern: str) -> pd.DataFrame:
    """
    Load a Schwab daily balance CSV for one account.
    Returns raw DataFrame with 'account' column added.
    Expected columns: Date, Amount
    """
    path = _find_file(Path(raw_dir, "balances"), pattern)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["account"] = account_key
    return df


# ---------------------------------------------------------------------------
# Positions (multi-section Schwab export)
# ---------------------------------------------------------------------------

def load_positions(raw_dir: str) -> pd.DataFrame:
    """
    Parse the Schwab 'All-Accounts Positions' CSV.
    The file has one section per account, each preceded by an unquoted
    account-name line.  Returns a flat DataFrame with an 'account' column.
    """
    pos_dir = Path(raw_dir, "positions")
    # Accept both .csv and .CSV
    matches = sorted(pos_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime) + \
              sorted(pos_dir.glob("*.CSV"), key=lambda p: p.stat().st_mtime)
    if not matches:
        return pd.DataFrame()
    return _parse_positions_file(matches[-1])


def _parse_positions_file(path: Path) -> pd.DataFrame:
    """
    Parse a multi-section Schwab positions file into a flat DataFrame.

    File structure:
        "Positions for All-Accounts as of ..."   ← title, skip
        (blank)
        AccountName ...XXX                       ← unquoted account label
        "Symbol","Description",...               ← header row (quoted)
        "AAPL","APPLE INC",...                   ← data rows (quoted)
        "Positions Total",...                    ← summary, skip
        (blank)
        NextAccountName ...YYY
        ...
    """
    sections: list[dict] = []
    current_account: str | None = None
    header: list[str] | None = None

    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue

            first = row[0].strip()

            # Title line
            if first.startswith("Positions for"):
                continue

            # Summary / total line — skip
            if first in ("Positions Total", "Futures Cash",
                         "Futures Positions Market Value"):
                continue

            # Header row — identified by "Symbol" in first cell
            if first == "Symbol":
                header = [c.strip() for c in row]
                continue

            # Account name line — unquoted, single cell or short row,
            # and we have no header yet for this section
            if header is None:
                current_account = first
                continue

            # If we hit a new account line after having a header, reset
            # (the positions file doesn't always have a blank line between sections)
            # Heuristic: if the first cell doesn't look like a symbol or "Cash"
            # and the row has only 1–2 non-empty cells, treat it as account name
            if len([c for c in row if c.strip()]) <= 2 and first not in ("", "--"):
                # Looks like a new account label
                current_account = first
                header = None
                continue

            # Data row
            if current_account and header:
                record = dict(zip(header, [c.strip() for c in row]))
                record["account"] = current_account
                sections.append(record)

    df = pd.DataFrame(sections)
    # Drop trailing-comma artifact column if present
    df = df.loc[:, ~df.columns.str.strip().eq("")]
    return df
