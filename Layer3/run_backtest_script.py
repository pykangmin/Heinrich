from datetime import date

import pandas as pd

from backtest import calc_metrics, run_backtest

result = run_backtest(
    date(2024, 1, 2),
    date(2026, 4, 2),
    cost_pct=0.001,
    top_n=20,
)
portfolio, benchmark = result["portfolio"], result["benchmark"]
idx_dt = pd.to_datetime(portfolio.index)

for label, mask in [
    ("IN-SAMPLE 2024", idx_dt.year == 2024),
    ("OUT-OF-SAMPLE 2025+", idx_dt.year >= 2025),
]:
    m = calc_metrics(portfolio.loc[mask], benchmark.loc[mask])
    print(f"\n=== {label} ===")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")

portfolio.to_frame("portfolio").join(benchmark.rename("benchmark")).to_csv(
    "backtest_equity_curves.csv"
)
