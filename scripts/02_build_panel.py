"""Build data/processed/macro_panel.csv and returns.csv and print their coverage."""
import pandas as pd

from regime_hmm.data import save_processed

panel, returns = save_processed()

for name, table in [("macro_panel", panel), ("returns", returns)]:
    print(f"\n{name}: {table.index.min():%Y-%m} ~ {table.index.max():%Y-%m}, {len(table)} months")
    coverage = pd.DataFrame(
        {
            "first": table.apply(lambda s: s.first_valid_index()).dt.strftime("%Y-%m"),
            "last": table.apply(lambda s: s.last_valid_index()).dt.strftime("%Y-%m"),
            "missing": table.isna().sum(),
        }
    )
    print(coverage.to_string())

gaps = panel[panel.isna().any(axis=1)]
if len(gaps):
    print("\nmonths with a missing value:")
    print(gaps.isna().apply(lambda row: ", ".join(row.index[row]), axis=1).to_string())
