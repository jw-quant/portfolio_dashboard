# report.py — run_report() orchestrator + generate_final_report()

import base64
import io
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
import loader

# Repo root = parent of src/
_REPO_ROOT = Path(__file__).resolve().parent.parent
import cleaner
import transactions as txn
import metrics as m


# ---------------------------------------------------------------------------
# Public API — pipeline
# ---------------------------------------------------------------------------

def run_report(
    raw_dir: str,
    accounts: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    risk_free_rate: float = config.DEFAULT_RISK_FREE_RATE,
    benchmark: str | None = None,
) -> dict:
    """
    Run the portfolio analytics pipeline for one or more accounts.

    Parameters
    ----------
    raw_dir        : path to the data/raw directory
    accounts       : list of account keys, e.g. ["Retirement"]
                     or ["Retirement", "Trade_and_leaps"]
    start_date     : ISO date string "YYYY-MM-DD" — required for meaningful
                     annualized metrics; pass None only for exploratory runs
    end_date       : ISO date string "YYYY-MM-DD", or None for latest available
    risk_free_rate : annualized Rf for Sharpe (decimal)
    benchmark      : ticker string e.g. "SPY", or None to skip

    Returns
    -------
    dict with keys:
      nav               pd.Series  combined daily NAV (date-indexed)
      account_navs      dict       {account_key: pd.Series} per-account NAV
      daily_cf          pd.Series  combined daily net external cash flows
      cumulative_cf     pd.Series  cumulative sum of daily_cf
      daily_returns     pd.Series  Modified Dietz daily returns
      cum_return        pd.Series  cumulative wealth index starting at 1.0
      drawdown          pd.Series  drawdown time series
      monthly_rets      pd.Series  monthly returns (ME-indexed)
      metrics           pd.DataFrame  scalar metrics summary (Value + Benchmark cols)
      benchmark_returns pd.Series | None  daily returns for benchmark
      benchmark_metrics pd.DataFrame | None  partial metrics for benchmark
    """
    # ------------------------------------------------------------------
    # 1. Validate account keys
    # ------------------------------------------------------------------
    for key in accounts:
        if key not in config.ACCOUNTS:
            known = list(config.ACCOUNTS.keys())
            raise ValueError(
                f"Unknown account '{key}'. Known accounts: {known}\n"
                f"Register new accounts in src/config.py → ACCOUNTS."
            )

    # ------------------------------------------------------------------
    # 2. Load and clean data for each account
    # ------------------------------------------------------------------
    nav_series: dict[str, pd.Series] = {}
    cf_series:  dict[str, pd.Series] = {}

    for key in accounts:
        acct = config.ACCOUNTS[key]

        # --- balances ---
        raw_bal  = loader.load_balances(raw_dir, key, acct["balances_pattern"])
        bal      = cleaner.clean_balances(raw_bal)
        nav_s    = bal.set_index("date")["nav"]
        nav_s.index = pd.to_datetime(nav_s.index)
        nav_s.name  = key
        nav_series[key] = nav_s

        # --- transactions ---
        raw_txn = loader.load_transactions(raw_dir, key, acct["transactions_pattern"])
        tx      = cleaner.clean_transactions(raw_txn)
        tx      = txn.tag_cashflows(tx)
        cf_s    = txn.daily_cashflows(tx)
        cf_s.index = pd.to_datetime(cf_s.index)
        cf_series[key] = cf_s

    # ------------------------------------------------------------------
    # 3. Build combined NAV and CF series
    # ------------------------------------------------------------------
    all_navs     = pd.DataFrame(nav_series).sort_index().ffill()
    combined_nav = all_navs.sum(axis=1)
    combined_nav.name = "combined_nav"

    all_cf = pd.concat(cf_series.values()).groupby(level=0).sum()
    all_cf.index = pd.to_datetime(all_cf.index)
    all_cf.name  = "cashflow"

    # ------------------------------------------------------------------
    # 4. Reconciliation warning for same-day inter-account journals
    # ------------------------------------------------------------------
    if len(accounts) > 1:
        _warn_unbalanced_journals(cf_series)

    # ------------------------------------------------------------------
    # 5. Apply date filter
    # ------------------------------------------------------------------
    if start_date is None and end_date is None:
        warnings.warn(
            "No start_date or end_date specified.  Metrics will span the full "
            "history, which may not be meaningful if accounts began at different "
            "times.  Set start_date in the notebook parameters cell.",
            UserWarning, stacklevel=2,
        )

    t0 = pd.Timestamp(start_date) if start_date else combined_nav.index.min()
    t1 = pd.Timestamp(end_date)   if end_date   else combined_nav.index.max()

    nav_slice = combined_nav.loc[t0:t1]
    cf_slice  = all_cf.reindex(nav_slice.index, fill_value=0.0)

    account_nav_slices = {k: v.reindex(nav_slice.index).ffill()
                          for k, v in nav_series.items()}

    # ------------------------------------------------------------------
    # 6. Compute returns and metrics
    # ------------------------------------------------------------------
    daily_rets = m.compute_daily_returns(nav_slice, cf_slice)
    cum_ret    = m.cumulative_return(daily_rets)
    dd_series  = m.drawdown_series(daily_rets)
    mon_rets   = m.monthly_returns(daily_rets)
    cum_cf     = cf_slice.cumsum()
    portfolio_metrics = m.metrics_summary(daily_rets, nav_slice, cf_slice, risk_free_rate)

    # ------------------------------------------------------------------
    # 7. Load benchmark (optional)
    # ------------------------------------------------------------------
    benchmark_rets    = None
    benchmark_metrics = None

    if benchmark:
        try:
            import benchmark as bm_mod
            benchmark_rets = bm_mod.load_benchmark_returns(
                raw_dir,
                benchmark,
                str(nav_slice.index[0].date()),
                str(nav_slice.index[-1].date()),
            )
            benchmark_rets = benchmark_rets.reindex(nav_slice.index)
            benchmark_metrics = m.benchmark_metrics_summary(
                benchmark_rets, risk_free_rate
            )
        except Exception as exc:
            warnings.warn(
                f"Benchmark '{benchmark}' skipped: {exc}",
                UserWarning, stacklevel=2,
            )

    # Merge benchmark column into metrics table when available
    if benchmark_metrics is not None:
        summary = portfolio_metrics.join(benchmark_metrics)
    else:
        summary = portfolio_metrics

    # ------------------------------------------------------------------
    # 8. Return results
    # ------------------------------------------------------------------
    return {
        "nav":               nav_slice,
        "account_navs":      account_nav_slices,
        "daily_cf":          cf_slice,
        "cumulative_cf":     cum_cf,
        "daily_returns":     daily_rets,
        "cum_return":        cum_ret,
        "drawdown":          dd_series,
        "monthly_rets":      mon_rets,
        "metrics":           summary,
        "benchmark_returns": benchmark_rets,
        "benchmark_metrics": benchmark_metrics,
    }


# ---------------------------------------------------------------------------
# Public API — final report
# ---------------------------------------------------------------------------

def generate_final_report(
    results: dict,
    output_dir: str | Path | None = None,
    processed_base: str | Path | None = None,
) -> Path:
    """
    Save processed artifacts to a timestamped folder under data/processed/,
    then write a self-contained HTML report to reports/.

    Artifacts saved to data/processed/<timestamp>/:
      metrics.csv, nav.csv, cumulative_cf.csv, cum_return.csv,
      monthly_returns.csv, benchmark_returns.csv (if available),
      equity_curve.png, cum_return.png, drawdown.png, nav_vs_cashflows.png

    Returns the path to the HTML report file.
    """
    import plots  # import here to avoid circular issues at module level

    if output_dir is None:
        output_dir = _REPO_ROOT / "reports" 
    if processed_base is None:
        processed_base = _REPO_ROOT / "data" / "processed"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    proc_dir = Path(processed_base) / timestamp
    proc_dir.mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    results["metrics"].to_csv(proc_dir / "metrics.csv")
    results["nav"].to_csv(proc_dir / "nav.csv", header=True)
    results["cumulative_cf"].to_csv(proc_dir / "cumulative_cf.csv", header=True)
    results["cum_return"].to_csv(proc_dir / "cum_return.csv", header=True)
    results["monthly_rets"].dropna().to_csv(proc_dir / "monthly_returns.csv", header=True)

    if results.get("benchmark_returns") is not None:
        results["benchmark_returns"].to_csv(
            proc_dir / "benchmark_returns.csv", header=True
        )

    # ------------------------------------------------------------------
    # Render and save charts (PNG + base64 for HTML embedding)
    # ------------------------------------------------------------------
    bm_rets = results.get("benchmark_returns")
    accounts = results.get("account_navs", {})

    chart_specs = [
        ("equity_curve",    lambda: plots.plot_equity_curve(
            results["nav"],
            account_navs=accounts if len(accounts) > 1 else None,
            benchmark=bm_rets,
        )),
        ("cum_return",      lambda: plots.plot_cum_return(
            results["cum_return"], benchmark=bm_rets
        )),
        ("drawdown",        lambda: plots.plot_drawdown(results["drawdown"])),
        ("nav_vs_cashflows",lambda: plots.plot_nav_vs_cashflows(
            results["nav"], results["cumulative_cf"]
        )),
    ]

    chart_b64: dict[str, str] = {}
    for name, render_fn in chart_specs:
        fig = render_fn()
        # Save PNG to processed dir
        fig.savefig(proc_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        # Encode as base64 for self-contained HTML
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        chart_b64[name] = base64.b64encode(buf.read()).decode()
        import matplotlib.pyplot as plt
        plt.close(fig)

    # ------------------------------------------------------------------
    # Build HTML report
    # ------------------------------------------------------------------
    html = _build_html(results, chart_b64, timestamp)
    report_path = Path(output_dir) / f"report_{timestamp}.html"
    report_path.write_text(html, encoding="utf-8")

    print(f"[report] Processed artifacts -> {proc_dir}")
    print(f"[report] HTML report         -> {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def _build_html(results: dict, chart_b64: dict, timestamp: str) -> str:
    metrics_html = results["metrics"].to_html(
        classes="metrics-table", border=0, na_rep="—"
    )

    mon = results["monthly_rets"].dropna()
    mon_html = (
        mon.map(lambda x: f"{x:.2%}")
        .rename("Monthly Return")
        .to_frame()
        .to_html(classes="metrics-table", border=0)
    )

    start = results["nav"].index[0].date()
    end   = results["nav"].index[-1].date()
    end_nav = results["nav"].iloc[-1]

    def img(name):
        return (
            f'<img src="data:image/png;base64,{chart_b64[name]}" '
            f'style="width:100%;max-width:960px;margin:12px 0">'
        )

    bm_note = ""
    if results.get("benchmark_returns") is not None:
        ticker = getattr(results["benchmark_returns"], "name", "Benchmark")
        bm_note = f" &nbsp;·&nbsp; Benchmark: {ticker}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Portfolio Report {start} – {end}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1000px; margin: 40px auto; padding: 0 20px;
         color: #222; background: #fff; }}
  h1   {{ font-size: 1.5rem; border-bottom: 2px solid #1f77b4;
         padding-bottom: 8px; }}
  h2   {{ font-size: 1.1rem; color: #555; margin-top: 32px; }}
  .meta {{ color: #666; font-size: 0.9rem; margin: -6px 0 20px; }}
  .metrics-table {{ border-collapse: collapse; font-size: 0.9rem; }}
  .metrics-table th, .metrics-table td {{
    padding: 5px 14px; text-align: right; border-bottom: 1px solid #eee; }}
  .metrics-table th {{ background: #f4f4f4; text-align: center; }}
  .metrics-table tr:first-child th {{ border-top: 2px solid #ccc; }}
  .metrics-table td:first-child {{ text-align: left; color: #444; }}
</style>
</head>
<body>
<h1>Portfolio Report</h1>
<p class="meta">
  {start} → {end} &nbsp;·&nbsp; End NAV: <strong>${end_nav:,.0f}</strong>{bm_note}
  &nbsp;·&nbsp; Generated: {timestamp[:8]}
</p>

<h2>Performance Metrics</h2>
{metrics_html}

<h2>Portfolio Value Over Time</h2>
{img("equity_curve")}

<h2>Cumulative Return (TWR)</h2>
{img("cum_return")}

<h2>Drawdown</h2>
{img("drawdown")}

<h2>NAV vs Cumulative Cash Flows</h2>
{img("nav_vs_cashflows")}

<h2>Monthly Returns</h2>
{mon_html}

</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _warn_unbalanced_journals(_cf_series: dict[str, pd.Series]) -> None:
    pass   # placeholder — no false-alarm noise for v1
