# Portfolio Analytics Engine

Python-based portfolio analytics system for broker account exports.

Built to ingest raw brokerage CSVs, adjust returns for external cash flows, benchmark against SPY, and generate a self-contained HTML performance report.

See processed preview screenshot below of a sample report:
<img width="809" height="720" alt="image" src="https://github.com/user-attachments/assets/1791e41f-a41b-4fe1-81a4-d6671e2c6422" />


## Why this project exists

Broker exports are noisy, cash movements distort naive return calculations, and multi-account performance reporting is tedious to do manually.

This project standardizes the workflow:
- parse raw brokerage exports
- clean balances and transactions
- classify external cash flows
- compute cash-flow-adjusted returns
- benchmark against SPY
- generate a shareable HTML report

## Features

- Cash-flow-adjusted performance using Modified Dietz-style daily returns
- Settlement-timing correction for same-day transfer distortions
- Multi-account aggregation
- Optional SPY benchmark comparison
- Drawdown, monthly return, volatility, and Sharpe metrics
- Self-contained HTML report output
- Processed CSV and chart artifacts saved automatically

## Methodology

### Return calculation
Daily portfolio returns are computed using a Modified Dietz framework:

R_t = (NAV_t - NAV_{t-1} - CF_t) / (NAV_{t-1} + 0.5 * CF_t)

where:
- NAV_t = end-of-day portfolio value
- CF_t = net external cash flow on day t

### Settlement correction
Brokerage cash transfers are not always reflected in balances on the same date they appear in transactions.  
To prevent mathematically impossible return distortions, the engine shifts excess cash flow to the next day when needed.

### Benchmarking
When enabled, benchmark returns are loaded separately and normalized for side-by-side comparison in the report.

## Example output

The generated HTML report includes:
- portfolio value over time
- cumulative return (TWR)
- drawdown
- NAV vs cumulative cash flows
- monthly return table
- scalar performance metrics

## Project structure

data/
- raw/
- processed/

notebooks/
- dashboard.ipynb

reports/
- generated HTML reports

src/
- loader.py
- cleaner.py
- transactions.py
- metrics.py
- benchmark.py
- plots.py
- report.py

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
