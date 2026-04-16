# plots.py — chart helpers for portfolio dashboard

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _date_axis(ax, nav_len: int) -> None:
    """Configure x-axis date formatting based on series length."""
    if nav_len > 365 * 2:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def _fig(title: str, figsize=(13, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(alpha=0.25, linestyle="--")
    return fig, ax


def _benchmark_cum_return(benchmark: pd.Series) -> pd.Series | None:
    """Compute cumulative wealth index (starts at 1.0) from daily returns."""
    if benchmark is None:
        return None
    r = benchmark.fillna(0.0)
    return (1 + r).cumprod()


# ---------------------------------------------------------------------------
# Chart 1 — Equity curve (combined + per-account + optional benchmark)
# ---------------------------------------------------------------------------

def plot_equity_curve(
    nav: pd.Series,
    account_navs: dict | None = None,
    benchmark: pd.Series | None = None,
    title: str = "Portfolio Value Over Time",
) -> plt.Figure:
    """
    Line chart of the combined portfolio NAV.
    If account_navs dict is supplied, individual accounts are shown as
    lighter dashed lines beneath the combined line.
    If benchmark daily-return series is supplied, it is normalized to the
    portfolio's starting NAV and overlaid as a gray reference line.
    """
    fig, ax = _fig(title)

    # Per-account lines (optional)
    if account_navs and len(account_navs) > 1:
        colors = plt.cm.tab10.colors
        for i, (key, series) in enumerate(account_navs.items()):
            s = series.dropna()
            ax.plot(s.index, s.values / 1e3,
                    linewidth=1, linestyle="--", alpha=0.55,
                    color=colors[i % len(colors)], label=key)

    # Combined portfolio line
    ax.plot(nav.index, nav.values / 1e3,
            linewidth=2, color="#1f77b4",
            label="Portfolio" if benchmark is not None or
                  (account_navs and len(account_navs) > 1) else None)

    # Benchmark overlay — normalized to portfolio starting NAV
    if benchmark is not None:
        bm_cum = _benchmark_cum_return(benchmark)
        if bm_cum is not None:
            start_nav = nav.iloc[0]
            bm_nav = bm_cum * start_nav
            bm_name = getattr(benchmark, "name", "Benchmark")
            ax.plot(bm_nav.index, bm_nav.values / 1e3,
                    linewidth=1.5, linestyle=":", color="#7f7f7f",
                    alpha=0.85, label=bm_name)

    ax.set_ylabel("Portfolio Value ($K)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    _date_axis(ax, len(nav))

    has_legend = (
        benchmark is not None
        or (account_navs and len(account_navs) > 1)
    )
    if has_legend:
        ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 2 — Drawdown
# ---------------------------------------------------------------------------

def plot_drawdown(
    drawdown: pd.Series,
    title: str = "Drawdown (Investment Return Basis)",
) -> plt.Figure:
    """Area chart of the drawdown series (already computed from TWR curve)."""
    fig, ax = _fig(title)
    dd = drawdown.dropna()
    ax.fill_between(dd.index, dd.values * 100, 0,
                    alpha=0.45, color="#d62728")
    ax.plot(dd.index, dd.values * 100,
            linewidth=1, color="#d62728")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    _date_axis(ax, len(dd))
    # Annotate max drawdown
    min_val = dd.min()
    min_idx = dd.idxmin()
    ax.annotate(
        f"Max: {min_val:.1%}",
        xy=(min_idx, min_val * 100),
        xytext=(15, -20), textcoords="offset points",
        fontsize=8, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 3 — Portfolio value vs cumulative cash flows
# ---------------------------------------------------------------------------

def plot_nav_vs_cashflows(
    nav: pd.Series,
    cumulative_cf: pd.Series,
    title: str = "Portfolio Value vs Cumulative Net Cash Flows",
) -> plt.Figure:
    """
    Overlay two lines:
      - Portfolio NAV (left axis)
      - Cumulative net external cash flows (right axis)
    """
    fig, ax1 = plt.subplots(figsize=(13, 4))
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=10)

    ax1.plot(nav.index, nav.values / 1e3,
             linewidth=2, color="#1f77b4", label="Portfolio NAV")
    ax1.set_ylabel("Portfolio Value ($K)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))

    ax2 = ax1.twinx()
    cf_aligned = cumulative_cf.reindex(nav.index, fill_value=None).ffill().fillna(0)
    ax2.plot(cf_aligned.index, cf_aligned.values / 1e3,
             linewidth=1.5, linestyle="--", color="#ff7f0e", label="Cumulative CF")
    ax2.set_ylabel("Cumulative Net Cash Flows ($K)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ax1.grid(alpha=0.25, linestyle="--")
    _date_axis(ax1, len(nav))
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 4 — Cumulative return (TWR wealth index) + optional benchmark
# ---------------------------------------------------------------------------

def plot_cum_return(
    cum_return: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Cumulative Investment Return (TWR)",
) -> plt.Figure:
    """
    Line chart of the cumulative return index (starts at 1.0).
    If benchmark daily-return series is supplied, its cumulative return
    is overlaid as a gray reference line starting at 1.0.
    """
    fig, ax = _fig(title)
    cr = cum_return.dropna()
    ax.plot(cr.index, cr.values, linewidth=2, color="#2ca02c",
            label="Portfolio")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Growth of $1 (TWR)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"${y:.2f}"))
    _date_axis(ax, len(cr))

    # Annotate portfolio final value
    final = cr.iloc[-1]
    ax.annotate(
        f"{(final - 1):.1%}",
        xy=(cr.index[-1], final),
        xytext=(-40, 10), textcoords="offset points",
        fontsize=9, color="#2ca02c",
    )

    # Benchmark overlay
    if benchmark is not None:
        bm_cum = _benchmark_cum_return(benchmark)
        if bm_cum is not None:
            bm_aligned = bm_cum.reindex(cr.index).ffill()
            bm_name = getattr(benchmark, "name", "Benchmark")
            ax.plot(bm_aligned.index, bm_aligned.values,
                    linewidth=1.5, linestyle=":", color="#7f7f7f",
                    alpha=0.85, label=bm_name)
            # Annotate benchmark final value
            bm_final = bm_aligned.dropna().iloc[-1]
            ax.annotate(
                f"{(bm_final - 1):.1%}",
                xy=(bm_aligned.dropna().index[-1], bm_final),
                xytext=(5, -14), textcoords="offset points",
                fontsize=8, color="#7f7f7f",
            )
        ax.legend(fontsize=9)

    plt.tight_layout()
    return fig
