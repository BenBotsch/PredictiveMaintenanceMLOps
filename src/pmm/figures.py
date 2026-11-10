"""Figures of an evaluation, in a light and a dark theme, for the README and for the web.

The style, the fonts and the files come from ``figkit``, the figure style shared by all my
projects: every figure is written as ``<name>.svg``, ``<name>-dark.svg``, ``<name>.png`` and
``<name>-dark.png`` with a transparent background on a fixed 6.4 in canvas, and the social card as
an opaque 1200 x 630 ``card.png`` and ``card-dark.png``. Text uses the ink colours, never a series
colour, and every chart with two or more series has a legend. Tiles and boxes are a light tint
without an edge or a coloured stripe.
"""

import math
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import figkit
import matplotlib.pyplot as plt
import numpy as np
from figkit import Theme
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from pmm.drift import ALERT, WARN
from pmm.report import LABELS, Evaluation

# Colour follows the candidate, not its rank: the registered model always gets the first slot.
CANDIDATE_SLOT = {
    "boosting + features": 0,
    "boosting": 1,
    "logistic + features": 2,
    "logistic": 3,
}
# Slots 2 and 3, green and yellow, have little contrast on a light page and get a second encoding.
LINESTYLE: tuple[str | tuple[float, tuple[float, float]], ...] = ("-", "-", "--", (0, (1, 1.6)))
# Names of the inputs in figures, with their units; the code names stay in tables and the API.
FEATURE_LABELS = {
    "type": "Product type",
    "air_temp_k": "Air temperature (K)",
    "process_temp_k": "Process temperature (K)",
    "rotational_speed_rpm": "Rotational speed (rpm)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool wear (min)",
}
WIDTH = figkit.WEB_WIDTH

Draw = Callable[[Evaluation, Theme], Figure]


def _font(key: str) -> float:
    size: float = figkit.tokens()["font"][key]
    return size


def feature_label(name: str) -> str:
    """Readable name of an input with its unit, or the name itself if it has none."""
    return FEATURE_LABELS.get(name, name)


def wilson(detected: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a share; it stays inside 0..1 and holds up for few cases.

    Raises:
        ValueError: If there is no case to take a share of.
    """
    if total <= 0:
        msg = "a share needs at least one case"
        raise ValueError(msg)
    share = detected / total
    scale = 1 + z**2 / total
    centre = (share + z**2 / (2 * total)) / scale
    half = z * math.sqrt(share * (1 - share) / total + z**2 / (4 * total**2)) / scale
    return max(0.0, centre - half), min(1.0, centre + half)


def _arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float], theme: Theme) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14, color=theme.muted, linewidth=1.4
        )
    )


def pipeline(_: Evaluation, theme: Theme) -> Figure:
    """The path of the model, from the pinned dataset to the monitored API."""
    figure, ax = plt.subplots(figsize=(WIDTH, 3.3))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02)
    figkit.title(
        ax,
        "From pinned data to a monitored API",
        theme,
        subtitle="Training writes to MLflow, serving reads only the exported bundle.",
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
                ("Pinned data", "AI4I 2020,\nSHA-256 checked"),
                ("4 candidates", "MLflow runs,\nbest PR-AUC wins"),
                ("Registry", "champion alias\nonly if not worse"),
                ("Export", "skops bundle,\nno pickle"),
            ),
        ),
        (
            "Serving",
            11,
            theme.series[2],
            (
                ("Bundle", "model, threshold,\ndrift reference"),
                ("FastAPI", "/predict, /drift,\nnon-root Docker"),
                ("Monitoring", "/metrics, PSI,\nJSON request logs"),
            ),
        ),
    )
    # A tile is a light tint of its lane, without an edge or a coloured stripe.
    width, height, gap, start = 21.5, 20, 3, 5
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
                    alpha=0.12,
                    edgecolor="none",
                )
            )
            ax.text(x + 2, y + 4.6, step, fontweight="semibold", va="center")
            ax.text(
                x + 2,
                y - 3,
                detail,
                fontsize=_font("annotation"),
                color=theme.secondary,
                va="center",
                linespacing=1.4,
            )
            if i:
                _arrow(ax, (x - gap + 0.3, y), (x - 0.3, y), theme)
    # The exported bundle is what serving receives: an elbow from Export down to Bundle.
    top, bottom = lanes[0][1] - height / 2, lanes[1][1] + height / 2
    export_x = start + 3 * (width + gap) + width / 2
    bundle_x = start + width / 2
    middle = (top + bottom) / 2
    ax.plot(
        [export_x, export_x, bundle_x],
        [top - 0.5, middle, middle],
        color=theme.muted,
        linewidth=1.4,
        solid_joinstyle="round",
    )
    _arrow(ax, (bundle_x, middle), (bundle_x, bottom + 0.5), theme)
    return figure


def test_metrics(ev: Evaluation, theme: Theme) -> Figure:
    """Test metrics of the registered model with their bootstrap intervals."""
    figure, ax = plt.subplots(figsize=(WIDTH, 2.9))
    figkit.title(
        ax,
        "Test metrics with 95% bootstrap intervals",
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
    figure, ax = plt.subplots(figsize=(WIDTH, 4.0))
    figkit.title(
        ax,
        "Boosted trees with physics features win on validation",
        theme,
        subtitle="Interpolated precision per candidate, the highest PR-AUC is registered.",
    )
    for curve in sorted(ev.candidates, key=lambda c: CANDIDATE_SLOT[c.name], reverse=True):
        slot = CANDIDATE_SLOT[curve.name]
        # Interpolated precision: the best precision at this recall or more, the usual PR view.
        precision = np.maximum.accumulate(curve.precision)
        ax.plot(
            curve.recall,
            precision,
            color=theme.series[slot],
            drawstyle="steps-post",
            linestyle=LINESTYLE[slot],
            linewidth=2.4 if slot == 0 else 1.8,
            label=f"{curve.name}   PR-AUC {curve.pr_auc:.3f}",
            zorder=4 - slot,
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
    order = sorted(range(len(labels)), key=lambda i: CANDIDATE_SLOT[labels[i].split("   ")[0]])
    # The curves cross the whole plot, so the legend sits below it, in two columns.
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.2),
        ncols=2,
        handlelength=2.2,
        columnspacing=1.6,
    )
    return figure


def confusion(ev: Evaluation, theme: Theme) -> Figure:
    """Confusion matrix of the test split at the registered threshold."""
    c = ev.confusion
    found = c["true_positive"]
    failures = found + c["false_negative"]
    figure, ax = plt.subplots(figsize=(WIDTH, 3.4))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02)
    figkit.title(
        ax,
        f"{found} of {failures} failures found, {c['false_positive']} false alarms",
        theme,
        subtitle=f"Test split at threshold {ev.threshold:.3f}, chosen on validation data.",
    )
    ax.set_xlim(-0.42, 2)
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
                (col + 0.012, row + 0.025),
                0.976,
                0.95,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor=colour,
                alpha=0.14,
                edgecolor="none",
            )
        )
        # Large numbers in the text face: mono gives the comma a full cell and splits 1,447.
        ax.text(
            col + 0.5,
            row + 0.58,
            f"{count:,}",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="semibold",
            color=theme.ink,
        )
        ax.text(col + 0.5, row + 0.3, text, ha="center", va="center", color=theme.secondary)
    for x, label in ((0.5, "predicted: normal"), (1.5, "predicted: failure")):
        ax.text(x, 2.03, label, ha="center", va="bottom", color=theme.secondary)
    for y, label in ((1.5, "actual:\nnormal"), (0.5, "actual:\nfailure")):
        ax.text(-0.04, y, label, ha="right", va="center", color=theme.secondary)
    return figure


def threshold_sweep(ev: Evaluation, theme: Theme) -> Figure:
    """Precision, recall and F1 over the threshold, on validation."""
    figure, ax = plt.subplots(figsize=(WIDTH, 3.6))
    figkit.title(
        ax,
        "The threshold trades missed failures against false alarms",
        theme,
        subtitle="Validation split, serving uses the threshold with the highest F1.",
    )
    sweep = ev.sweep
    names = {"precision": LABELS["precision"], "recall": LABELS["recall"], "f1": "F1"}
    at = {key: float(np.interp(ev.threshold, sweep["threshold"], sweep[key])) for key in names}
    # At a threshold of 1 nothing is flagged and the curves fall to zero; the labels sit at the
    # right edge with the values just before that fall.
    last = 0.95
    ends = []
    for slot, (key, name) in enumerate(names.items()):
        ax.plot(sweep["threshold"], sweep[key], color=theme.series[slot], label=name)
        ax.plot(
            ev.threshold,
            at[key],
            "o",
            markersize=7,
            color=theme.series[slot],
            markeredgecolor=theme.surface,
            markeredgewidth=1.5,
            zorder=4,
        )
        ends.append((1.0, float(np.interp(last, sweep["threshold"], sweep[key])), name))
    figkit.reference_line(
        ax,
        ev.threshold,
        f"served threshold {ev.threshold:.3f}\n"
        f"precision {at['precision']:.2f}, recall {at['recall']:.2f}, F1 {at['f1']:.2f}",
        theme,
        where=0.36,
        side="left",
    )
    ax.set_xlabel("Decision threshold on the failure probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.04)
    figkit.percent(ax)
    figkit.grid(ax, "y")
    figkit.line_labels(ax, ends, theme)
    ax.legend(loc="lower left", ncols=3)
    return figure


def failure_modes(ev: Evaluation, theme: Theme) -> Figure:
    """Recall per failure mode on the test split, with Wilson intervals."""
    figure, ax = plt.subplots(figsize=(WIDTH, 2.8))
    figkit.title(
        ax,
        "Rule-based failures are found, tool wear mostly is not",
        theme,
        subtitle="Share of test failures flagged per mode, with 95% Wilson interval.",
    )
    items = list(reversed(ev.modes))
    colour = theme.series[0]
    for y, m in enumerate(items):
        if not m.total:
            # A mode without a case keeps its row, so the figure lists every recorded mode.
            ax.text(
                0.01,
                y,
                "no case in the test split",
                va="center",
                fontsize=_font("annotation"),
                color=theme.secondary,
            )
            continue
        low, high = wilson(m.detected, m.total)
        share = m.detected / m.total
        ax.plot([low, high], [y, y], color=colour, linewidth=3, alpha=0.35)
        ax.plot(share, y, "o", markersize=7, color=colour)
        figkit.value(ax, 1.1, y, f"{share:.0%}", theme, va="center", ha="right")
        figkit.value(
            ax, 1.12, y, f"{m.detected} of {m.total}", theme, va="center", color=theme.secondary
        )
    labels = [f"{m.name} ({m.mode})" if m.mode != "none" else m.name for m in items]
    ax.set_yticks(range(len(items)), labels)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines["bottom"].set_bounds(0, 1.0)
    ax.set_ylim(-0.6, len(items) - 0.4)
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
    # The spread in ink, so that it reads on the bar and beyond its end.
    ax.errorbar(
        [i.mean for i in items],
        range(len(items)),
        xerr=[i.std for i in items],
        fmt="none",
        ecolor=theme.ink,
        elinewidth=1,
        capsize=3,
    )
    reach = max(i.mean + i.std for i in items)
    for y, item in enumerate(items):
        x = item.mean + item.std + reach * 0.03
        figkit.value(ax, x, y, f"{item.mean:.2f}", theme, va="center")
    ax.set_yticks(range(len(items)), [feature_label(i.feature) for i in items])
    ax.set_xlim(0, reach * 1.15)
    figkit.category_axis(ax, "y")
    figkit.grid(ax, "x")
    return figure


def drift(ev: Evaluation, theme: Theme) -> Figure:
    """PSI per feature for the unchanged test split and for shifted torque, on a log scale."""
    shifted = next(name for name in ev.drift if name != "test split")
    worst = max(ev.drift[shifted].items(), key=lambda kv: kv[1])
    figure, ax = plt.subplots(figsize=(WIDTH, 3.1))
    figkit.title(
        ax,
        "A miscalibrated torque sensor raises a drift alert",
        theme,
        subtitle="Population stability index per input against the training data, log scale.",
    )
    features = sorted(ev.drift["test split"], key=lambda f: ev.drift[shifted][f])
    smallest = min(v for scores in ev.drift.values() for v in scores.values() if v > 0)
    low, high = min(0.002, smallest / 1.5), 1.0
    # Status zones instead of two thin lines; their labels stay in the text colour.
    ax.axvspan(WARN, ALERT, color=theme.status["warning"], alpha=0.14, linewidth=0, zorder=0)
    ax.axvspan(ALERT, high, color=theme.status["critical"], alpha=0.10, linewidth=0, zorder=0)
    for start, label in ((WARN, f"warn ≥ {WARN:g}"), (ALERT, f"alert ≥ {ALERT:g}")):
        ax.text(
            start * 1.08,
            0.99,
            label,
            transform=ax.get_xaxis_transform(),
            va="top",
            fontsize=_font("annotation"),
            color=theme.secondary,
        )
    for y, feature in enumerate(features):
        # A PSI of 0 has no place on a log axis; it is drawn at the left edge.
        test = max(ev.drift["test split"][feature], low)
        moved = max(ev.drift[shifted][feature], low)
        if moved > test * 1.05:
            ax.plot([test, moved], [y, y], color=theme.muted, linewidth=1, linestyle=(0, (2, 2)))
        ax.plot(
            test,
            y,
            "o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor=theme.series[0],
            markeredgewidth=2,
            zorder=3,
            label="test split" if y == 0 else None,
        )
        ax.plot(
            moved,
            y,
            "o",
            markersize=6,
            color=theme.status["critical"] if moved >= ALERT else theme.ink,
            zorder=4,
            label=shifted if y == 0 else None,
        )
    figkit.value(
        ax,
        worst[1] * 1.15,
        features.index(worst[0]),
        f"{worst[1]:.2f}",
        theme,
        va="center",
        ha="left",
        fontweight="semibold",
    )
    ax.set_xscale("log")
    ax.set_xlim(low, high)
    ax.xaxis.set_major_locator(FixedLocator([0.003, 0.01, 0.03, WARN, ALERT, 1.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(range(len(features)), [feature_label(f) for f in features])
    ax.set_ylim(-0.6, len(features) - 0.2)
    figkit.category_axis(ax, "y")
    figkit.grid(ax, "x")
    ax.legend(loc="lower right", handletextpad=0.4)
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
        ("Test PR-AUC", f"{pr_auc.value:.2f}", f"95% CI {pr_auc.low:.2f} to {pr_auc.high:.2f}"),
        ("Failures found", f"{c['true_positive']}/{failures}", f"on {ev.test_rows:,} operations"),
        ("False alarms", f"{c['false_positive']}", f"among {normal:,} normal runs"),
        ("Drift alert", f"PSI {shifted:.2f}", "torque sensor +8 Nm"),
    )
    width, gap, x0 = 25.5, 2.4, 6
    # The card surface is the lighter page colour, so the tiles take the other one: a quiet
    # tile without an edge, the numbers are the accent.
    tile = theme.pages[0] if theme.surface != theme.pages[0] else theme.grid
    for i, (label, number, note) in enumerate(tiles):
        x = x0 + i * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, 13),
                width,
                23,
                boxstyle="round,pad=0,rounding_size=2",
                facecolor=tile,
                edgecolor="none",
            )
        )
        ax.text(x + 2, 31.5, label, fontsize=11, color=theme.secondary, va="center")
        ax.text(x + 2, 24, number, fontsize=22, fontweight="semibold", va="center")
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
