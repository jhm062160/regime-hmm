"""Regime charts."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from .config import SECTOR_NAMES, TRAIN_END, VAL_END

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# The outlier regime is its own category; the normal regimes are ordered from
# bear to bull, so they share one hue and darken with the market return.
OUTLIER_COLOR = "#e34948"
NORMAL_RAMPS = {
    2: ["#6da7ec", "#184f95"],
    3: ["#86b6ef", "#2a78d6", "#104281"],
    4: ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
}

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
REFERENCE_COLOR = MUTED  # the market benchmark is a reference line, not a series

# One fixed colour per sector, listed in stacking order from the bottom. With
# nine sectors and eight palette hues, one sector takes a neutral. The order
# keeps neighbours distinguishable for colour-blind readers: the top four are
# the sectors the chosen strategy drops at times, and any two of them, or any
# of them and the neutral below, remain apart when the one between vanishes.
SECTOR_COLORS = {
    "XLE": "#e34948",
    "XLU": "#eda100",
    "XLV": "#e87ba4",
    "XLB": "#008300",
    "XLI": "#52514e",
    "XLP": "#4a3aa7",
    "XLF": "#1baf7a",
    "XLY": "#eb6834",
    "XLK": "#2a78d6",
}

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def _growth_of_one(returns: pd.DataFrame) -> pd.DataFrame:
    """Cumulative wealth, starting from 1 at the month end before the first return."""
    wealth = (1 + returns).cumprod()
    wealth.loc[wealth.index[0] - MonthEnd(1)] = 1.0
    return wealth.sort_index()


def _wealth_axis(ax, wealth: pd.DataFrame) -> None:
    """Log scale with plain ticks, and a value label at each line end, pushed apart where they are close."""
    ax.set_yscale("log")
    ticks = [t for t in [1, 1.5, 2, 3, 4, 6, 8] if wealth.min().min() * 0.9 <= t <= wealth.max().max() * 1.1]
    ax.set_yticks(ticks, labels=[f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.grid(axis="y", color=GRID, linewidth=0.8)

    ends = np.log(wealth.iloc[-1].sort_values())
    gap = 0.05 * (np.log(wealth.max().max()) - np.log(wealth.min().min()))
    for i in range(1, len(ends)):
        ends.iloc[i] = max(ends.iloc[i], ends.iloc[i - 1] + gap)
    for name, y in ends.items():
        ax.annotate(
            f"{name}  {wealth[name].iloc[-1]:.2f}", (wealth.index[-1], np.exp(y)), xytext=(6, 0),
            textcoords="offset points", va="center", color=SECONDARY, fontsize=9, annotation_clip=False,
        )


def _mark_test_start(axes, label_ax, first: pd.Timestamp) -> None:
    test_start = pd.Timestamp(VAL_END) + MonthEnd(0)
    for ax in axes:
        ax.axvline(test_start, color=SECONDARY, linewidth=0.8)
    for name, at in {"Validation": first, "Test": test_start}.items():
        label_ax.annotate(
            name, (at, 0), xycoords=("data", "axes fraction"), xytext=(4, 6),
            textcoords="offset points", color=MUTED, fontsize=9,
        )


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)


def plot_wealth(returns: pd.DataFrame, reference: str, title: str, path: Path) -> None:
    """Growth of 1 for up to three strategies and a reference benchmark, labelled at the line ends."""
    wealth = _growth_of_one(returns)
    series = [name for name in wealth.columns if name != reference]
    colors = dict(zip(series, SERIES_COLORS, strict=True)) | {reference: REFERENCE_COLOR}

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=SURFACE, layout="constrained")
    for name in wealth.columns:
        ax.plot(wealth.index, wealth[name], color=colors[name], linewidth=1.5, label=name)
    _wealth_axis(ax, wealth)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=24)
    ax.text(0, 1.03, "Growth of 1, net of costs, log scale", transform=ax.transAxes, color=SECONDARY, fontsize=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=SECONDARY, fontsize=10)
    _mark_test_start([ax], ax, wealth.index[0])
    _style(ax)
    ax.margins(x=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_portfolio(returns: pd.DataFrame, weights: pd.DataFrame, title: str, path: Path) -> None:
    """Growth of 1 above the sector weights, drawn as one 100% stacked bar per month.

    `returns` holds the strategy first, then its benchmarks; they are drawn in
    neutral tones so that colour is left to the sectors below.
    """
    wealth = _growth_of_one(returns)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7.2), sharex=True, height_ratios=[2, 3], facecolor=SURFACE, layout="constrained"
    )

    for name, (color, width) in zip(wealth.columns, [(INK, 1.8), (MUTED, 1.4), (AXIS, 1.4)], strict=True):
        top.plot(wealth.index, wealth[name], color=color, linewidth=width)
    _wealth_axis(top, wealth)
    top.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=24)
    top.text(0, 1.04, "Growth of 1, net of costs, log scale", transform=top.transAxes, color=SECONDARY, fontsize=10)

    # Each bar spans the month in which the weights were held.
    base = np.zeros(len(weights))
    for sector, color in SECTOR_COLORS.items():
        bottom.bar(
            weights.index, weights[sector], bottom=base, width=-27, align="edge", color=color,
            edgecolor=SURFACE, linewidth=0.4, label=f"{sector}  {SECTOR_NAMES[sector]}",
        )
        base += weights[sector].to_numpy()
    bottom.set_ylim(0, 1)
    bottom.set_yticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    bottom.text(0, 1.03, "Sector weights held during each month", transform=bottom.transAxes, color=SECONDARY, fontsize=10)
    handles, labels = bottom.get_legend_handles_labels()
    bottom.legend(
        handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False,
        labelcolor=SECONDARY, fontsize=9, handlelength=1.2,
    )

    _mark_test_start([top, bottom], top, wealth.index[0])
    for ax in (top, bottom):
        _style(ax)
    bottom.margins(x=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_yearly(yearly: pd.DataFrame, reference: str, title: str, path: Path) -> None:
    """Calendar-year returns as grouped bars, with the same colours as the wealth chart."""
    series = [name for name in yearly.columns if name != reference]
    colors = dict(zip(series, SERIES_COLORS, strict=True)) | {reference: REFERENCE_COLOR}

    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE, layout="constrained")
    width = 0.8 / yearly.shape[1]
    positions = np.arange(len(yearly))
    for i, name in enumerate(yearly.columns):
        ax.bar(positions + (i + 0.5) * width - 0.4, yearly[name], width=width * 0.88, color=colors[name], label=name)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(positions, labels=[str(year) for year in yearly.index])
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=30)
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.0), ncols=yearly.shape[1], frameon=False, labelcolor=SECONDARY,
        fontsize=10, handlelength=1.2, columnspacing=1.6, borderaxespad=0.2,
    )
    _style(ax)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_selection(table: pd.DataFrame, benchmark: pd.Series, chosen: tuple, title: str, path: Path) -> None:
    """Validation against test Sharpe ratio for every model, with the benchmark's values as reference lines.

    `table` needs a boolean `stable` column: models that failed the stability
    screen are drawn in grey behind the candidates.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor=SURFACE, layout="constrained")
    ax.axvline(benchmark["val_sharpe"], color=AXIS, linewidth=1)
    ax.axhline(benchmark["test_sharpe"], color=AXIS, linewidth=1)
    ax.annotate(
        f"{benchmark.name} (both lines)", (benchmark["val_sharpe"], 0), xycoords=("data", "axes fraction"),
        xytext=(-5, 6), textcoords="offset points", ha="right", color=MUTED, fontsize=9,
    )

    others = table.drop(index=chosen)
    ring = {"edgecolors": SURFACE, "linewidths": 1}
    size = 42 if len(table) < 500 else 16
    for is_stable, color, label in [(False, AXIS, "Failed the stability screen"), (True, SERIES_COLORS[0], "Candidate")]:
        group = others[others["stable"] == is_stable]
        ax.scatter(group["val_sharpe"], group["test_sharpe"], s=size, color=color, label=label, **ring)
    ax.scatter(
        table.loc[chosen, "val_sharpe"], table.loc[chosen, "test_sharpe"], s=70, color=SERIES_COLORS[1],
        label="Chosen on validation", zorder=3, **ring,
    )

    ax.set_xlabel("Validation Sharpe ratio (2013–2018)", color=SECONDARY, fontsize=10)
    ax.set_ylabel("Test Sharpe ratio (2019 onward)", color=SECONDARY, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=False, labelcolor=SECONDARY, fontsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def regime_names(n_normal: int) -> list[str]:
    names = [f"Regime {i}" for i in range(1, n_normal + 1)]
    names[0] += " (bear)"
    names[-1] += " (bull)"
    return ["Outlier", *names]


def plot_regimes(probs: pd.DataFrame, level: pd.Series, title: str, path: Path) -> None:
    """S&P 500 above the stacked regime probabilities, with the train/validation/test split marked."""
    n_normal = probs.shape[1] - 1
    colors = [OUTLIER_COLOR, *NORMAL_RAMPS[n_normal]]
    level = level.reindex(probs.index)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 5.6), sharex=True, height_ratios=[3, 2], facecolor=SURFACE, layout="constrained"
    )
    top.plot(level.index, level, color=INK, linewidth=1.5)
    top.set_yscale("log")
    top.set_yticks([500, 1000, 2000, 4000], labels=["500", "1,000", "2,000", "4,000"])
    top.minorticks_off()
    top.grid(axis="y", color=GRID, linewidth=0.8)
    top.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=24)
    top.text(0, 1.06, "S&P 500, month-end close, log scale", transform=top.transAxes, color=SECONDARY, fontsize=10)

    bottom.stackplot(probs.index, probs.to_numpy().T, colors=colors, labels=regime_names(n_normal), step="mid", linewidth=0)
    bottom.set_ylim(0, 1)
    bottom.set_yticks([0, 0.5, 1])
    bottom.margins(x=0)
    bottom.legend(
        loc="lower left", bbox_to_anchor=(0, 1.0), ncols=len(colors), frameon=False, labelcolor=SECONDARY,
        fontsize=10, handlelength=1.2, columnspacing=1.6, borderaxespad=0.2,
    )
    bottom.text(1, 1.05, "Filtered regime probability", transform=bottom.transAxes, color=SECONDARY, fontsize=10, ha="right")

    # Month ends at which validation and test begin.
    starts = {
        "Train": probs.index[0],
        "Validation": pd.Timestamp(TRAIN_END) + MonthEnd(1),
        "Test": pd.Timestamp(VAL_END) + MonthEnd(1),
    }
    for name, start in starts.items():
        if name != "Train":
            for ax in (top, bottom):
                ax.axvline(start, color=SECONDARY, linewidth=0.8)
        top.annotate(
            name, (start, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
            textcoords="offset points", color=MUTED, fontsize=9,
        )

    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9, length=3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
