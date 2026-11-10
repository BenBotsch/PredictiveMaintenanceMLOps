"""Figures of an evaluation, in a light and a dark theme, for the README and for the web.

Every figure is drawn once per theme and rendered from vector at each target width instead of
being scaled down, so text stays sharp at 800 px. The colours are a validated categorical palette
(adjacent pairs separable under colour-vision deficiencies, see the README); text always uses the
ink colours, never a series colour, and every chart with two or more series has a legend.
"""

import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import numpy as np

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from pmm.drift import ALERT, WARN
from pmm.report import LABELS, Evaluation

README_WIDTH = 1600
WEB_WIDTHS = (1600, 800)
CARD_SIZE = (1200, 630)
FONT = "DejaVu Sans"  # ships with matplotlib, so the images are the same on every machine


@dataclass(frozen=True, slots=True)
class Theme:
    """Surface, ink and series colours of one theme."""

    name: str
    surface: str
    primary: str
    secondary: str
    muted: str
    grid: str
    baseline: str
    series: tuple[str, str, str, str]
    warning: str = "#fab219"
    critical: str = "#d03b3b"


LIGHT = Theme(
    "light",
    surface="#fcfcfb",
    primary="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
)
DARK = Theme(
    "dark",
    surface="#1a1a19",
    primary="#f0efec",
    secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500"),
)
THEMES = (LIGHT, DARK)

# Colour follows the candidate, not its rank: the registered model always gets the first slot.
CANDIDATE_SLOT = {
    "boosting + features": 0,
    "boosting": 1,
    "logistic + features": 2,
    "logistic": 3,
}

Draw = Callable[[Figure, Evaluation, Theme], None]


def _setup(theme: Theme) -> None:
    plt.rcParams.update(
        {
            "font.family": FONT,
            "font.size": 12,
            "text.color": theme.primary,
            "axes.labelcolor": theme.secondary,
            "axes.edgecolor": theme.baseline,
            "axes.facecolor": theme.surface,
            "figure.facecolor": theme.surface,
            "xtick.color": theme.secondary,
            "ytick.color": theme.secondary,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.labelsize": 12,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "svg.hashsalt": "pmm",
        }
    )


def _header(fig: Figure, theme: Theme, title: str, subtitle: str) -> None:
    fig.text(0.035, 0.94, title, fontsize=16, fontweight="bold", va="top")
    fig.text(0.035, 0.865, subtitle, fontsize=12, color=theme.secondary, va="top")


def _axes(fig: Figure, theme: Theme, *, left: float = 0.09, grid: Literal["x", "y"] = "y") -> Axes:
    ax = fig.add_axes((left, 0.13, 0.96 - left, 0.62))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1)
    ax.tick_params(length=0, pad=6)
    ax.grid(axis=grid, color=theme.grid, linewidth=1)
    ax.set_axisbelow(True)
    return ax


def _percent(ax: Axes, axis: str = "y") -> None:
    formatter = mpl.ticker.PercentFormatter(xmax=1, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def pipeline(fig: Figure, _: Evaluation, theme: Theme) -> None:
    """The path of the model, from the pinned dataset to the monitored API."""
    _header(
        fig,
        theme,
        "From pinned data to a monitored API",
        "Training writes to MLflow; serving reads only the exported bundle.",
    )
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.78))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    lanes = (
        (
            "Training",
            36.5,
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
            11.5,
            theme.series[2],
            (
                ("Bundle", "model, threshold,\ndrift reference"),
                ("FastAPI", "/predict, /drift\nnon-root Docker"),
                ("Monitoring", "/metrics, PSI,\nJSON request logs"),
            ),
        ),
    )
    width, height, gap, start = 19.5, 19, 4, 7
    for lane, y, colour, steps in lanes:
        ax.text(
            0.5,
            y,
            lane,
            rotation=90,
            va="center",
            ha="left",
            fontsize=12,
            color=theme.secondary,
            fontweight="bold",
        )
        for i, (title, detail) in enumerate(steps):
            x = start + i * (width + gap)
            ax.add_patch(
                FancyBboxPatch(
                    (x, y - height / 2),
                    width,
                    height,
                    boxstyle="round,pad=0,rounding_size=1.6",
                    facecolor=colour,
                    alpha=0.16,
                    edgecolor="none",
                )
            )
            ax.add_patch(
                Rectangle(
                    (x, y - height / 2 + 1.6), 0.8, height - 3.2, facecolor=colour, edgecolor="none"
                )
            )
            ax.text(x + 2.4, y + 4.2, title, fontsize=13.5, fontweight="bold", va="center")
            ax.text(
                x + 2.4,
                y - 2.6,
                detail,
                fontsize=11.5,
                color=theme.secondary,
                va="center",
                linespacing=1.35,
            )
            if i:
                ax.add_patch(
                    FancyArrowPatch(
                        (x - gap + 0.6, y),
                        (x - 0.6, y),
                        arrowstyle="-|>",
                        mutation_scale=14,
                        color=theme.muted,
                        linewidth=1.6,
                    )
                )
    # The exported bundle is what serving receives: an elbow from Export down to Bundle.
    export_x = start + 3 * (width + gap) + width / 2
    bundle_x = start + width / 2
    middle = (36.5 - height / 2 + 11.5 + height / 2) / 2
    ax.plot(
        [export_x, export_x, bundle_x],
        [36.5 - height / 2 - 0.6, middle, middle],
        color=theme.muted,
        linewidth=1.6,
        solid_joinstyle="round",
    )
    ax.add_patch(
        FancyArrowPatch(
            (bundle_x, middle),
            (bundle_x, 11.5 + height / 2 + 0.6),
            arrowstyle="-|>",
            mutation_scale=14,
            color=theme.muted,
            linewidth=1.6,
        )
    )


def test_metrics(fig: Figure, ev: Evaluation, theme: Theme) -> None:
    """Test metrics of the registered model with their bootstrap intervals."""
    _header(
        fig,
        theme,
        "Test metrics with 95 % bootstrap intervals",
        f"{ev.test_rows:,} unseen rows, only {ev.test_failures} failures: "
        "the intervals are wide and belong to the result.",
    )
    ax = _axes(fig, theme, left=0.15, grid="x")
    items = list(reversed(ev.intervals))
    colour = theme.series[0]
    for y, item in enumerate(items):
        ax.plot([item.low, item.high], [y, y], color=colour, linewidth=2.5, alpha=0.45)
        ax.plot(
            item.value,
            y,
            "o",
            markersize=10,
            color=colour,
            markeredgecolor=theme.surface,
            markeredgewidth=2,
        )
        ax.text(
            item.high + 0.012, y, f"{item.value:.3f}", va="center", fontsize=12, fontweight="bold"
        )
        ax.text(
            item.high + 0.07,
            y,
            f"[{item.low:.2f}, {item.high:.2f}]",
            va="center",
            fontsize=11,
            color=theme.secondary,
        )
    ax.set_yticks(range(len(items)), [LABELS[i.metric] for i in items], fontsize=12)
    ax.tick_params(axis="y", colors=theme.primary)
    ax.set_xlim(0.5, 1.13)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.spines["left"].set_visible(False)


def pr_curves(fig: Figure, ev: Evaluation, theme: Theme) -> None:
    """Validation precision-recall curves of the four candidates."""
    _header(
        fig,
        theme,
        "Boosted trees with physics features win on validation",
        "Interpolated precision over recall per candidate; the one with the highest "
        "PR-AUC is registered.",
    )
    ax = _axes(fig, theme)
    for curve in sorted(ev.candidates, key=lambda c: CANDIDATE_SLOT[c.name], reverse=True):
        # Interpolated precision: the best precision at this recall or more, the usual PR view.
        precision = np.maximum.accumulate(curve.precision)
        ax.plot(
            curve.recall,
            precision,
            color=theme.series[CANDIDATE_SLOT[curve.name]],
            drawstyle="steps-post",
            label=f"{curve.name}  (PR-AUC {curve.pr_auc:.3f})",
            zorder=3 if CANDIDATE_SLOT[curve.name] == 0 else 2,
        )
    ax.axhline(ev.prevalence, color=theme.baseline, linewidth=1)
    ax.text(
        0.01,
        ev.prevalence + 0.025,
        f"random guess ({ev.prevalence:.1%} failures)",
        fontsize=10.5,
        color=theme.secondary,
    )
    ax.set_xlabel("Recall: share of failures found")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.04)
    _percent(ax, "x")
    _percent(ax, "y")
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: CANDIDATE_SLOT[labels[i].split("  ")[0]])
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower left",
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0, 0.06),
        labelcolor=theme.primary,
    )


def confusion(fig: Figure, ev: Evaluation, theme: Theme) -> None:
    """Confusion matrix of the test split at the registered threshold."""
    c = ev.confusion
    found = c["true_positive"]
    failures = found + c["false_negative"]
    _header(
        fig,
        theme,
        f"{found} of {failures} failures found, {c['false_positive']} false alarms",
        f"Test split at threshold {ev.threshold:.3f}, chosen on validation data.",
    )
    ax = fig.add_axes((0.2, 0.04, 0.62, 0.66))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis("off")
    good, bad = theme.series[0], theme.series[1]
    cells = (
        (0, 1, c["true_negative"], "normal runs passed", good),
        (1, 1, c["false_positive"], "false alarms", bad),
        (0, 0, c["false_negative"], "failures missed", bad),
        (1, 0, c["true_positive"], "failures found", good),
    )
    for col, row, count, text, colour in cells:
        ax.add_patch(
            FancyBboxPatch(
                (col + 0.02, row + 0.02),
                0.96,
                0.96,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor=colour,
                alpha=0.18,
                edgecolor="none",
            )
        )
        ax.text(
            col + 0.5,
            row + 0.56,
            f"{count:,}",
            ha="center",
            va="center",
            fontsize=30,
            fontweight="bold",
        )
        ax.text(
            col + 0.5,
            row + 0.3,
            text,
            ha="center",
            va="center",
            fontsize=12.5,
            color=theme.secondary,
        )
    for x, label in ((0.5, "predicted: normal"), (1.5, "predicted: failure")):
        ax.text(x, 2.04, label, ha="center", va="bottom", fontsize=12, color=theme.secondary)
    for y, label in ((1.5, "actual:\nnormal"), (0.5, "actual:\nfailure")):
        ax.text(-0.04, y, label, ha="right", va="center", fontsize=12, color=theme.secondary)


def threshold_sweep(fig: Figure, ev: Evaluation, theme: Theme) -> None:
    """Precision, recall and F1 over the threshold, on validation."""
    _header(
        fig,
        theme,
        "The threshold trades missed failures against false alarms",
        "Validation split; the threshold with the highest F1 is used in serving.",
    )
    ax = _axes(fig, theme)
    sweep = ev.sweep
    for slot, key in enumerate(("precision", "recall", "f1")):
        ax.plot(
            sweep["threshold"],
            sweep[key],
            color=theme.series[slot],
            label=LABELS[key] if key != "f1" else "F1",
        )
    ax.axvline(ev.threshold, color=theme.secondary, linewidth=1)
    ax.text(
        ev.threshold - 0.012,
        0.05,
        f"served threshold {ev.threshold:.3f}",
        ha="right",
        fontsize=11,
        color=theme.secondary,
    )
    ax.set_xlabel("Decision threshold on the failure probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.04)
    _percent(ax)
    ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=11,
        ncols=3,
        bbox_to_anchor=(0, 1.0),
        labelcolor=theme.primary,
    )


def failure_modes(fig: Figure, ev: Evaluation, theme: Theme) -> None:
    """Recall per failure mode on the test split."""
    modes = [m for m in ev.modes if m.total]
    absent = [m.mode for m in ev.modes if not m.total]
    note = f" No {', '.join(absent)} failure in the test split." if absent else ""
    _header(
        fig,
        theme,
        "Rule-based failures are found, tool wear mostly is not",
        f"Share of test failures flagged, by recorded failure mode.{note}",
    )
    ax = _axes(fig, theme, left=0.22, grid="x")
    items = list(reversed(modes))
    shares = [m.detected / m.total for m in items]
    ax.barh(range(len(items)), shares, height=0.5, color=theme.series[0])
    for y, (m, share) in enumerate(zip(items, shares, strict=True)):
        ax.text(
            share + 0.015,
            y,
            f"{share:.0%}  ({m.detected} of {m.total})",
            va="center",
            fontsize=11.5,
        )
    labels = [f"{m.name} ({m.mode})" if m.mode != "none" else m.name for m in items]
    ax.set_yticks(range(len(items)), labels, fontsize=12)
    ax.tick_params(axis="y", colors=theme.primary)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    _percent(ax, "x")
    ax.spines["left"].set_color(theme.baseline)


def importance(fig: Figure, ev: Evaluation, theme: Theme) -> None:
    """Permutation importance of the raw inputs."""
    _header(
        fig,
        theme,
        "Torque carries the most signal",
        "Drop in test PR-AUC when one input is shuffled (mean ± sd over 10 repeats).",
    )
    ax = _axes(fig, theme, left=0.24, grid="x")
    items = list(reversed(ev.importance))
    ax.barh(range(len(items)), [i.mean for i in items], height=0.5, color=theme.series[0])
    ax.errorbar(
        [i.mean for i in items],
        range(len(items)),
        xerr=[i.std for i in items],
        fmt="none",
        ecolor=theme.secondary,
        elinewidth=1.5,
        capsize=4,
    )
    for y, item in enumerate(items):
        ax.text(item.mean + item.std + 0.015, y, f"{item.mean:.2f}", va="center", fontsize=11.5)
    ax.set_yticks(
        range(len(items)), [i.feature for i in items], fontsize=12, fontfamily="DejaVu Sans Mono"
    )
    ax.tick_params(axis="y", colors=theme.primary)
    ax.set_xlim(0, max(i.mean + i.std for i in items) * 1.15)
    ax.spines["left"].set_color(theme.baseline)


def drift(fig: Figure, ev: Evaluation, theme: Theme) -> None:
    """PSI per feature for the unchanged test split and for shifted torque."""
    shifted = next(name for name in ev.drift if name != "test split")
    worst = max(ev.drift[shifted].items(), key=lambda kv: kv[1])
    _header(
        fig,
        theme,
        "A miscalibrated torque sensor raises a drift alert",
        f"Population stability index per input against the training data; "
        f"{shifted} gives PSI {worst[1]:.2f}.",
    )
    ax = _axes(fig, theme, left=0.24, grid="x")
    features = list(ev.drift["test split"])
    features.sort(key=lambda f: ev.drift[shifted][f])
    height = 0.34
    for slot, (name, offset) in enumerate((("test split", height / 2), (shifted, -height / 2))):
        values = [ev.drift[name][f] for f in features]
        ax.barh(
            [y + offset for y in range(len(features))],
            values,
            height=height - 0.04,
            color=theme.series[slot],
            label=name,
            zorder=3,
        )
    for limit, colour, label in ((WARN, theme.warning, "warn"), (ALERT, theme.critical, "alert")):
        ax.axvline(limit, color=colour, linewidth=1.5, zorder=1)
        ax.text(
            limit + 0.008,
            len(features) - 0.45,
            f"{label} {limit:g}",
            fontsize=11,
            color=theme.secondary,
        )
    y_worst = features.index(worst[0]) - height / 2
    ax.text(
        worst[1] + 0.01, y_worst, f"{worst[1]:.2f}", va="center", fontsize=11.5, fontweight="bold"
    )
    ax.set_yticks(range(len(features)), features, fontsize=12, fontfamily="DejaVu Sans Mono")
    ax.tick_params(axis="y", colors=theme.primary)
    ax.set_xlim(0, max(0.35, worst[1] * 1.15))
    ax.set_ylim(-0.6, len(features) - 0.1)
    ax.spines["left"].set_color(theme.baseline)
    ax.legend(loc="lower right", frameon=False, fontsize=11, labelcolor=theme.primary)


def card(fig: Figure, ev: Evaluation, theme: Theme) -> None:
    """Title card with the headline numbers, for the README top and social previews."""
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 63)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 61), 120, 2, facecolor=theme.series[0], edgecolor="none"))
    ax.text(6, 53, "Predictive Maintenance MLOps", fontsize=34, fontweight="bold", va="center")
    ax.text(
        6,
        45.5,
        "Machine failure prediction on AI4I 2020, from pinned data to a monitored API",
        fontsize=16,
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
        ("Failures found", f"{c['true_positive']} / {failures}", "on 1,500 unseen operations"),
        ("False alarms", f"{c['false_positive']}", f"among {normal:,} normal runs"),
        ("Drift alert", f"PSI {shifted:.2f}", "torque sensor +8 Nm"),
    )
    width, gap, x0 = 25.5, 2.4, 6
    for i, (label, value, note) in enumerate(tiles):
        x = x0 + i * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, 14),
                width,
                23,
                boxstyle="round,pad=0,rounding_size=1.2",
                facecolor=theme.series[0],
                alpha=0.12,
                edgecolor="none",
            )
        )
        ax.text(x + 1.8, 32.5, label, fontsize=14, color=theme.secondary, va="center")
        ax.text(x + 1.8, 24.5, value, fontsize=30, fontweight="bold", va="center")
        ax.text(x + 1.8, 17.5, note, fontsize=12, color=theme.secondary, va="center")
    ax.text(
        6,
        6.5,
        "MLflow  ·  scikit-learn  ·  FastAPI  ·  Docker  ·  Prometheus  ·  GitHub Actions",
        fontsize=14,
        color=theme.secondary,
        va="center",
    )


# name -> (draw, width and height in inches). Fonts are in points, so the size in inches sets
# how large text appears relative to the image.
FIGURES: dict[str, tuple[Draw, tuple[float, float]]] = {
    "card": (card, (12, 6.3)),
    "pipeline": (pipeline, (10, 5.2)),
    "test-metrics": (test_metrics, (10, 5)),
    "pr-curves": (pr_curves, (10, 5.6)),
    "confusion": (confusion, (10, 5.6)),
    "threshold": (threshold_sweep, (10, 5.6)),
    "failure-modes": (failure_modes, (10, 5)),
    "importance": (importance, (10, 5)),
    "drift": (drift, (10, 5.4)),
}


def _render(
    draw: Draw,
    size: tuple[float, float],
    ev: Evaluation,
    theme: Theme,
    width: int,
) -> Image.Image:
    _setup(theme)
    fig = plt.figure(figsize=size)
    try:
        draw(fig, ev, theme)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=width / size[0], facecolor=theme.surface)
    finally:
        plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _save_png(image: Image.Image, path: Path) -> None:
    # Flat charts survive a 256-colour palette; it makes the PNGs several times smaller.
    image.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        path, optimize=True
    )


def render(ev: Evaluation, directory: Path) -> list[Path]:
    """Write all figures in both themes.

    ``figures/<name>-<theme>.png`` is for the README, ``web/<name>-<theme>-<width>.webp`` for a
    website (``srcset``), and ``web/card-<theme>.png`` at 1200 x 630 for social previews.

    Returns:
        The written files.
    """
    readme, web = directory / "figures", directory / "web"
    readme.mkdir(parents=True, exist_ok=True)
    web.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (draw, size) in FIGURES.items():
        for theme in THEMES:
            for width in sorted({README_WIDTH, *WEB_WIDTHS}, reverse=True):
                image = _render(draw, size, ev, theme, width)
                if width == README_WIDTH:
                    path = readme / f"{name}-{theme.name}.png"
                    _save_png(image, path)
                    written.append(path)
                if width in WEB_WIDTHS:
                    path = web / f"{name}-{theme.name}-{width}.webp"
                    image.save(path, quality=88, method=6)
                    written.append(path)
            if name == "card":
                path = web / f"card-{theme.name}.png"
                _save_png(_render(draw, size, ev, theme, CARD_SIZE[0]), path)
                written.append(path)
    return written
