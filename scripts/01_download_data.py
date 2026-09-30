"""Download every raw series into data/raw/."""
from regime_hmm.config import MACRO, RISK_FREE
from regime_hmm.data import download_fred, download_yahoo

for series_id in [spec.fred_id for spec in MACRO.values()] + [RISK_FREE]:
    s = download_fred(series_id).dropna()
    print(f"FRED  {series_id:<10} {s.index.min():%Y-%m-%d} ~ {s.index.max():%Y-%m-%d}  n={len(s)}")

close = download_yahoo()
for ticker in close.columns:
    s = close[ticker].dropna()
    print(f"Yahoo {ticker:<10} {s.index.min():%Y-%m-%d} ~ {s.index.max():%Y-%m-%d}  n={len(s)}")
