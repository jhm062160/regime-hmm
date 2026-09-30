"""Data sources and the rules that turn each raw series into a panel column."""
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRED_RAW_DIR = ROOT / "data" / "raw" / "fred"
YAHOO_RAW_FILE = ROOT / "data" / "raw" / "yahoo" / "close.csv"
PANEL_FILE = ROOT / "data" / "processed" / "macro_panel.csv"
RETURNS_FILE = ROOT / "data" / "processed" / "returns.csv"
STABILITY_FILE = ROOT / "results" / "stability.csv"
MODELS_FILE = ROOT / "results" / "models.pkl"
PROBS_FILE = ROOT / "results" / "probs.pkl"
SUMMARY_FILE = ROOT / "results" / "regime_summary.csv"
PLOTS_DIR = ROOT / "results" / "plots"
BACKTEST_DIR = ROOT / "results" / "backtest"


@dataclass(frozen=True)
class Series:
    fred_id: str
    freq: str  # frequency of the raw series: "D", "W" or "M"
    transform: str  # "level", "diff" or "logdiff", applied after collapsing to months
    lag: int  # months from the end of the reference month until the value may be used
    agg: str = "last"  # how a daily series collapses to a month


# Publication lags are fixed approximations of the release calendar:
#   monthly macro   -> released during the following month   -> lag 1
#   market series   -> observable at month end               -> lag 0
# Values are the latest revised figures, not real-time vintages.
# Quarterly series (GDP, PNFI) are left out: with their release lag they
# reach the panel two to four months late.
MACRO = {
    "PCE": Series("PCE", "M", "logdiff", 1),
    "INDPRO": Series("INDPRO", "M", "logdiff", 1),
    "PAYEMS": Series("PAYEMS", "M", "logdiff", 1),
    "UNRATE": Series("UNRATE", "M", "diff", 1),
    "AWHMAN": Series("AWHMAN", "M", "diff", 1),
    "AWHNONAG": Series("AWHNONAG", "M", "diff", 1),
    "ICSA": Series("ICSA", "W", "logdiff", 0),
    "T10Y2YM": Series("T10Y2YM", "M", "level", 0),
    "BAA10YM": Series("BAA10YM", "M", "level", 0),
    "VIX": Series("VIXCLS", "D", "level", 0),
}

# Weekly claims are dated by the Saturday ending the week and released the
# following Thursday. Weeks are assigned to the month of their release.
WEEKLY_RELEASE_DELAY_DAYS = 5

RISK_FREE = "TB3MS"  # 3-month T-bill, percent per year

INDEX = "^GSPC"
BENCHMARK = "SPY"
# The nine Select Sector SPDRs listed in December 1998. XLRE (2015) and
# XLC (2018) are left out so the universe is the same in every period.
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
SECTOR_NAMES = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}

# Candidate numbers of sectors held per regime, chosen on validation together
# with the model. Holding all nine is the equal-weight benchmark, not a candidate.
TOP_N_GRID = range(1, len(SECTORS))

# Train runs from the first panel month; test runs to the last one.
TRAIN_END = "2012-12"
VAL_START = "2013-01"
VAL_END = "2018-12"
TEST_START = "2019-01"
PERIODS = {"train": slice(None, TRAIN_END), "val": slice(VAL_START, VAL_END), "test": slice(TEST_START, None)}

# The model grid of the first version: a fixed core plus every subset of the
# optional variables, 64 sets x 3 regime counts = 192 models.
CORE = ["SP500", "VIX", "T10Y2YM", "BAA10YM", "PCE"]
OPTIONAL = ["INDPRO", "PAYEMS", "AWHNONAG", "AWHMAN", "ICSA", "UNRATE"]
# Number of normal regimes; every model has the outlier state on top.
N_NORMAL = [2, 3, 4]
# Clustering runs per model in the stability screen. Only models whose train
# labels are identical in every run become candidates.
N_REPEAT = 100


def grid() -> dict[str, tuple[list[str], int]]:
    """Every model in the grid: name -> (variables, number of normal regimes)."""
    models = {}
    for size in range(len(OPTIONAL) + 1):
        for optional in combinations(OPTIONAL, size):
            for n_normal in N_NORMAL:
                models["+".join(("core", *optional)) + f"_r{n_normal}"] = (CORE + list(optional), n_normal)
    return models
