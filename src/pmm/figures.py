"""Figures of an evaluation, in a light and a dark theme, for the README and for the web.

The style, the fonts and the files come from ``figkit``, the figure style shared by all my
projects: every figure is written as ``<name>.svg``, ``<name>-dark.svg``, ``<name>.png`` and
``<name>-dark.png`` with a transparent background, and the social card as an opaque 1200 x 630
``card.png`` and ``card-dark.png``. Text uses the ink colours, never a series colour, and every
chart with two or more series has a legend.
"""

from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import figkit
import matplotlib.pyplot as plt
import numpy as np
from figkit import Theme
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from pmm.drift import ALERT, WARN
from pmm.report import LABELS, Evaluation

# Colour follows the candidate, not its rank: the registered model always gets the first slot.
CANDIDATE_SLOT = {
    "boosting + features": 0,
    "boosting": 1,
    "logistic + features": 2,
    "logistic": 3,
}
WIDTH = figkit.WEB_WIDTH

Draw = Callable[[Evaluation, Theme], Figure]


def _font(key: str) -> float:
    size: float = figkit.tokens()["font"][key]
    return size


def _arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float], theme: Theme) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=10, color=theme.muted, linewidth=1.2
        )
    )


def pipeline(_: Evaluation, theme: Theme) -> Figure:
    """The path of the model, from the pinned dataset to the monitored API."""
    figure, ax = plt.subplots(figsize=(WIDTH, 3.3))
    figkit.title(
        ax,
        "From pinned data to a monitored API",
        theme,
        subtitle="Training writes to MLflow; serving reads only the exported bundle.",
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    lanes = (
        (
            "Training",
            37,
            theme.series[0],
            (
                ("Pinned data", "AI4I 2020\nSHA-256 checked"),
                ("4 candidates", "MLflow runs\nselected on PR-AUC"),
                ("Registry", "champion alias\nonly if not worse"),
                ("Export", "skops bundle\nno pickle"),
            ),
        ),
        (
            "Serving",
            11,
            theme.series[2],
            (
                ("Bundle", "model, threshold,\ndrift reference"),
                ("FastAPI", "/predict, /drift\nnon-root Docker"),
                ("Monitoring", "/metrics, PSI,\nJSON request logs"),
            ),
        ),
    )
    width, height, gap, start = 20, 20, 3.6, 6
    for lane, y, colour, steps in lanes:
        ax.text(
            0,
            y,
            lane,
            rotation=90,
            va="center",
            ha="left",
            color=theme.secondary,
            fontweight="semibold",
        )
        for i, (step, detail) in enumerate(steps):
            x = start + i * (width + gap)
            ax.add_patch(
                FancyBboxPatch(
                    (x, y - height / 2),
                    width,
                    height,
                    boxstyle="round,pad=0,rounding_size=1.2",
                    facecolor=colour,
                    alpha=0.14,
                    edgecolor="none",
                )
            )
            ax.add_patch(
                Rectangle(
                    (x, y - height / 2 + 2), 0.7, height - 4, facecolor=colour, edgecolor="none"
                )
            )
            ax.text(x + 2.4, y + 4.6, step, fontweight="semibold", va="center")
            ax.text(
                x + 2.4,
                y - 3,
                detail,
                fontsize=_font("annotation"),
                color=theme.secondary,
                va="center",
                linespacing=1.4,
            )
            if i:
                _arrow(ax, (x - gap + 0.4, y), (x - 0.4, y), theme)
    # The exported bundle is what serving receives: an elbow from Export down to Bundle.
    top, bottom = lanes[0][1] - height / 2, lanes[1][1] + height / 2
    export_x = start + 3 * (width + gap) + width / 2
    bundle_x = start + width / 2
    middle = (top + bottom) / 2
    ax.plot(
        [export_x, export_x, bundle_x],
        [top - 0.5, middle, middle],
        color=theme.muted,
        linewidth=1.2,
        solid_joinstyle="round",
    )
    _arrow(ax, (bundle_x, middle), (bundle_x, bottom + 0.5), theme)
    return figure


def test_metrics(ev: Evaluation, theme: Theme) -> Figure:
    """Test metrics of the registered model with their bootstrap intervals."""
    figure, ax = plt.subplots(figsize=(WIDTH, 2.9))
    figkit.title(
        ax,
        "Test metrics with 95 % bootstrap intervals",
        theme,
        subtitle=f"{ev.test_rows:,} unseen rows with only {ev.test_failures} failures, "
        "so the intervals are wide.",
    )
    items = list(reversed(ev.intervals))
    colour = theme.series[0]
    for y, item in enumerate(items):
        ax.plot([item.low, item.high], [y, y], color=colour, linewidth=3, alpha=0.35)
        ax.plot(item.value, y, "o", markersize=7, color=colour)
        figkit.value(ax, item.high + 0.012, y, f"{item.value:.3f}", theme, va="center")
        figkit.value(
            ax,
            item.high + 0.07,
            y,
            f"[{item.low:.2f}, {item.high:.2f}]",
            theme,
            va="center",
            color=theme.secondary,
        )
    ax.set_yticks(range(len(items)), [LABELS[i.metric] for i in items])
    ax.set_xlim(0.5, 1.16)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.spines["bottom"].set_bounds(0.5, 1.0)
    ax.set_ylim(-0.6, len(items) - 0.4)
    figkit.category_axis(ax, "y")
    figkit.grid(ax, "x")
    return figure


def pr_curves(ev: Evaluation, theme: Theme) -> Figure:
    """Validation precision-recall curves of the four candidates."""
    figure, ax = plt.subplots(figsize=(WIDTH, 3.6))
    figkit.title(
        ax,
        "Boosted trees with physics features win on validation",
        theme,
        subtitle="Interpolated precision per candidate; the highest PR-AUC is registered.",
    )
    for curve in sorted(ev.candidates, key=lambda c: CANDIDATE_SLOT[c.name], reverse=True):
        # Interpolated precision: the best precision at this recall or more, the usual PR view.
        precision = np.maximum.accumulate(curve.precision)
        ax.plot(
            curve.recall,
            precision,
            color=theme.series[CANDIDATE_SLOT[curve.name]],
            drawstyle="steps-post",
            label=f"{curve.name}\nPR-AUC {curve.pr_auc:.3f}",
            zorder=3 if CANDIDATE_SLOT[curve.name] == 0 else 2,
        )
    figkit.reference_line(
        ax,
        ev.prevalence,
        f"random guess ({ev.prevalence:.1%} failures)",
        theme,
        axis="y",
        where=0.01,
    )
    ax.set_xlabel("Recall: share of failures found")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.04)
    figkit.percent(ax, "x")
    figkit.percent(ax, "y")
    figkit.grid(ax, "y")
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: CANDIDATE_SLOT[labels[i].split("\n")[0]])
    # The curves fill the plot, so the legend stands beside it.
    figure.subplots_adjust(right=0.72)
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        labelspacing=1.1,
    )
    return figure


def confusion(ev: Evaluation, theme: Theme) -> Figure:
    """Confusion matrix of the test split at the registered threshold."""
    c = ev.confusion
    found = c["true_positive"]
    failures = found + c["false_negative"]
    figure, ax = plt.subplots(figsize=(WIDTH, 3.4))
    figkit.title(
        ax,
        f"{found} of {failures} failures found, {c['false_positive']} false alarms",
        theme,
        subtitle=f"Test split at threshold {ev.threshold:.3f}, chosen on validation data.",
    )
    ax.set_xlim(-0.62, 2)
    ax.set_ylim(0, 2.14)
    ax.axis("off")
    right, wrong = theme.series[0], theme.series[1]
    cells = (
        (0, 1, c["true_negative"], "normal runs passed", right),
        (1, 1, c["false_positive"], "false alarms", wrong),
        (0, 0, c["false_negative"], "failures missed", wrong),
        (1, 0, c["true_positive"], "failures found", right),
    )
    for col, row, count, text, colour in cells:
        ax.add_patch(
            FancyBboxPatch(
                (col + 0.015, row + 0.025),
                0.97,
                0.95,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor=colour,
                alpha=0.16,
                edgecolor="none",
            )
        )
        figkit.value(
            ax,
            col + 0.5,
            row + 0.58,
            f"{count:,}",
            theme,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="semibold",
        )
        ax.text(col + 0.5, row + 0.3, text, ha="center", va="center", color=theme.secondary)
    for x, label in ((0.5, "predicted: normal"), (1.5, "predicted: failure")):
        ax.text(x, 2.03, label, ha="center", va="bottom", color=theme.secondary)
    for y, label in ((1.5, "actual:\nnormal"), (0.5, "actual:\nfailure")):
        ax.text(-0.05, y, label, ha="right", va="center", color=theme.secondary)
    return figure


def threshold_sweep(ev: Evaluation, theme: Theme) -> Figure:
    """Precision, recall and F1 over the threshold, on validation."""
    figure, ax = plt.subplots(figsize=(WIDTH, 3.6))
    figkit.title(
        ax,
        "The threshold trades missed failures against false alarms",
        theme,
        subtitle="Validation split; serving uses the threshold with the highest F1.",
    )
    sweep = ev.sweep
    for slot, key in enumerate(("precision", "recall", "f1")):
        ax.plot(
            sweep["threshold"],
            sweep[key],
            color=theme.series[slot],
            label=LABELS[key] if key != "f1" else "F1",
        )
    figkit.reference_line(
        ax, ev.threshold, f"served threshold {ev.threshold:.3f}", theme, where=0.12, side="left"
    )
    ax.set_xlabel("Decision threshold on the failure probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.04)
    figkit.percent(ax)
    figkit.grid(ax, "y")
    ax.legend(loc="lower center", ncols=3, bbox_to_anchor=(0.4, 0.2))
    return figure


def failure_modes(ev: Evaluation, theme: Theme) -> Figure:
    """Recall per failure mode on the test split."""
    modes = [m for m in ev.modes if m.total]
    absent = [m.mode for m in ev.modes if not m.total]
    note = f" No {', '.join(absent)} failure in the test split." if absent else ""
    figure, ax = plt.subplots(figsize=(WIDTH, 2.7))
    figkit.title(
        ax,
        "Rule-based failures are found, tool wear mostly is not",
        theme,
        subtitle=f"Share of test failures flagged, by recorded failure mode.{note}",
    )
    items = list(reversed(modes))
    shares = [m.detected / m.total for m in items]
    bars = ax.barh(range(len(items)), shares, height=0.56, color=theme.series[0])
    figkit.bar_labels(
        ax,
        bars,
        theme,
        labels=[
            f"{share:.0%}  ({m.detected} of {m.total})"
            for m, share in zip(items, shares, strict=True)
        ],
    )
    labels = [f"{m.name} ({m.mode})" if m.mode != "none" else m.name for m in items]
    ax.set_yticks(range(len(items)), labels)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines["bottom"].set_bounds(0, 1.0)
    figkit.percent(ax, "x")
    figkit.category_axis(ax, "y")
    figkit.grid(ax, "x")
    return figure


def importance(ev: Evaluation, theme: Theme) -> Figure:
    """Permutation importance of the raw inputs."""
    figure, ax = plt.subplots(figsize=(WIDTH, 2.9))
    figkit.title(
        ax,
        "Torque carries the most signal",
        theme,
        subtitle="Drop in test PR-AUC when one input is shuffled (mean ± sd over 10 repeats).",
    )
    items = list(reversed(ev.importance))
    ax.barh(range(len(items)), [i.mean for i in items], height=0.56, color=theme.series[0])
    ax.errorbar(
        [i.mean for i in items],
        range(len(items)),
        xerr=[i.std for i in items],
        fmt="none",
        ecolor=theme.secondary,
        elinewidth=1,
        capsize=3,
    )
    for y, item in enumerate(items):
        figkit.value(ax, item.mean + item.std + 0.015, y, f"{item.mean:.2f}", theme, va="center")
    ax.set_yticks(range(len(items)), [i.feature for i in items])
    figkit.mono_ticks(ax, "y")
    ax.set_xlim(0, max(i.mean + i.std for i in items) * 1.15)
    figkit.category_axis(ax, "y")
    figkit.grid(ax, "x")
    return figure


def drift(ev: Evaluation, theme: Theme) -> Figure:
    """PSI per feature for the unchanged test split and for shifted torque."""
    shifted = next(name for name in ev.drift if name != "test split")
    worst = max(ev.drift[shifted].items(), key=lambda kv: kv[1])
    figure, ax = plt.subplots(figsize=(WIDTH, 3.3))
    figkit.title(
        ax,
        "A miscalibrated torque sensor raises a drift alert",
        theme,
        subtitle=f"Population stability index per input against the training data; "
        f"{shifted} gives PSI {worst[1]:.2f}.",
    )
    features = list(ev.drift["test split"])
    features.sort(key=lambda f: ev.drift[shifted][f])
    height = 0.36
    for slot, (name, offset) in enumerate((("test split", height / 2), (shifted, -height / 2))):
        ax.barh(
            [y + offset for y in range(len(features))],
            [ev.drift[name][f] for f in features],
            height=height - 0.04,
            color=theme.series[slot],
            label=name,
            zorder=3,
        )
    for limit, status, label in ((WARN, "warning", "warn"), (ALERT, "critical", "alert")):
        figkit.reference_line(
            ax, limit, f"{label} {limit:g}", theme, colour=theme.status[status], where=1.0
        )
    y_worst = features.index(worst[0]) - height / 2
    figkit.value(
        ax, worst[1] + 0.01, y_worst, f"{worst[1]:.2f}", theme, va="center", fontweight="semibold"
    )
    ax.set_yticks(range(len(features)), features)
    figkit.mono_ticks(ax, "y")
    ax.set_xlim(0, max(0.35, worst[1] * 1.15))
    ax.set_ylim(-0.6, len(features) - 0.1)
    figkit.category_axis(ax, "y")
    figkit.grid(ax, "x")
    ax.legend(loc="lower right")
    return figure


def card(ev: Evaluation, theme: Theme) -> Figure:
    """Title card with the headline numbers, for the README top and social previews."""
    figure = plt.figure(figsize=figkit.CARD_INCHES)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 63)
    ax.axis("off")
    ax.text(6, 52, "Predictive Maintenance MLOps", fontsize=26, fontweight="semibold", va="center")
    ax.text(
        6,
        45,
        "Machine failure prediction on AI4I 2020, from pinned data to a monitored API",
        fontsize=12.5,
        color=theme.secondary,
        va="center",
    )
    c = ev.confusion
    failures = c["true_positive"] + c["false_negative"]
    pr_auc = next(i for i in ev.intervals if i.metric == "pr_auc")
    normal = c["true_negative"] + c["false_positive"]
    shifted = max(v for k, d in ev.drift.items() if k != "test split" for v in d.values())
    tiles = (
        ("Test PR-AUC", f"{pr_auc.value:.2f}", f"95 % CI {pr_auc.low:.2f} to {pr_auc.high:.2f}"),
        ("Failures found", f"{c['true_positive']}/{failures}", f"on {ev.test_rows:,} operations"),
        ("False alarms", f"{c['false_positive']}", f"among {normal:,} normal runs"),
        ("Drift alert", f"PSI {shifted:.2f}", "torque sensor +8 Nm"),
    )
    width, gap, x0 = 25.5, 2.4, 6
    for i, (label, number, note) in enumerate(tiles):
        x = x0 + i * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, 13),
                width,
                23,
                boxstyle="round,pad=0,rounding_size=2",
                facecolor=theme.ramp[1],
                edgecolor="none",
            )
        )
        ax.text(x + 2, 31.5, label, fontsize=11, color=theme.secondary, va="center")
        ax.text(x + 2, 24, number, fontsize=21, fontfamily=figkit.mono(), va="center")
        ax.text(x + 2, 17, note, fontsize=9, color=theme.secondary, va="center")
    ax.text(
        6,
        6,
        "MLflow  ·  scikit-learn  ·  FastAPI  ·  Docker  ·  Prometheus  ·  GitHub Actions",
        fontsize=11,
        color=theme.secondary,
        va="center",
    )
    return figure


# name -> (draw, outputs). Charts are transparent SVG and PNG; the card is an opaque PNG.
FIGURES: dict[str, tuple[Draw, Sequence[figkit.Output]]] = {
    "card": (card, figkit.CARD),
    "pipeline": (pipeline, figkit.WEB),
    "test-metrics": (test_metrics, figkit.WEB),
    "pr-curves": (pr_curves, figkit.WEB),
    "confusion": (confusion, figkit.WEB),
    "threshold": (threshold_sweep, figkit.WEB),
    "failure-modes": (failure_modes, figkit.WEB),
    "importance": (importance, figkit.WEB),
    "drift": (drift, figkit.WEB),
}


def render(ev: Evaluation, directory: Path) -> list[Path]:
    """Write all figures in both themes to ``directory / "figures"``.

    Returns:
        The written files.
    """
    out = directory / "figures"
    written = []
    for name, (draw, outputs) in FIGURES.items():
        written += figkit.render(out, name, partial(draw, ev), outputs)
    return written
