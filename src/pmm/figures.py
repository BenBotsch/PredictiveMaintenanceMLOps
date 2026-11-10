"""Figures of an evaluation, in a light and a dark theme, for the README and for the web.

The style, the fonts and the files come from ``figkit``, the figure style shared by all my
projects: every figure is written as ``<name>.svg``, ``<name>-dark.svg``, ``<name>.png`` and
``<name>-dark.png`` with a transparent background on a fixed 6.4 in canvas, and the social card as
an opaque 1200 x 630 ``card.png`` and ``card-dark.png``. Text uses the ink colours, never a series
colour, and every chart with two or more series has a legend. Tiles and boxes are a light tint
without an edge or a coloured stripe.

For the portfolio, every chart is also written without its title in English and German to
``figures/page``, with a ``figures.json`` that holds title, subtitle and alt text per language.
The texts come from :data:`TEXT`; the draw functions read the language that ``figkit.render``
sets, and numbers go through ``figkit.number``, so German gets ``0,95`` and ``1.500``.
"""

import math
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import figkit
import matplotlib.pyplot as plt
import numpy as np
from figkit import Theme, number
from figkit.language import speaking
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from pmm.drift import ALERT, WARN
from pmm.report import LABELS, MODE_NAMES, TORQUE_SHIFT_NM, Evaluation

# Colour follows the candidate, not its rank: the registered model always gets the first slot.
CANDIDATE_SLOT = {
    "boosting + features": 0,
    "boosting": 1,
    "logistic + features": 2,
    "logistic": 3,
}
# Slots 2 and 3, green and yellow, have little contrast on a light page and get a second encoding.
LINESTYLE: tuple[str | tuple[float, tuple[float, float]], ...] = ("-", "-", "--", (0, (1, 1.6)))
LANGUAGES = ("en", "de")
# Names of the inputs in figures, with their units; the code names stay in tables and the API.
FEATURE_LABELS = {
    "en": {
        "type": "Product type",
        "air_temp_k": "Air temperature (K)",
        "process_temp_k": "Process temperature (K)",
        "rotational_speed_rpm": "Rotational speed (rpm)",
        "torque_nm": "Torque (Nm)",
        "tool_wear_min": "Tool wear (min)",
    },
    "de": {
        "type": "Produkttyp",
        "air_temp_k": "Lufttemperatur (K)",
        "process_temp_k": "Prozesstemperatur (K)",
        "rotational_speed_rpm": "Drehzahl (1/min)",
        "torque_nm": "Drehmoment (Nm)",
        "tool_wear_min": "Werkzeugverschleiß (min)",
    },
}
CANDIDATE_LABELS = {
    "en": {name: name for name in CANDIDATE_SLOT},
    "de": {
        "boosting + features": "Boosting + Merkmale",
        "boosting": "Boosting",
        "logistic + features": "logistisch + Merkmale",
        "logistic": "logistisch",
    },
}
MODE_LABELS = {
    "en": {**MODE_NAMES, "none": "No mode recorded"},
    "de": {
        "TWF": "Werkzeugverschleiß",
        "HDF": "Wärmeabfuhr",
        "PWF": "Leistung",
        "OSF": "Überlastung",
        "RNF": "Zufall",
        "none": "Keine Ausfallart erfasst",
    },
}
# Keys of ``Evaluation.drift``: the unchanged test split, the other one is the shifted copy.
TEST_SPLIT = "test split"
SCENARIO_LABELS = {
    "en": {TEST_SPLIT: "test split", "shifted": f"torque +{TORQUE_SHIFT_NM:g} Nm"},
    "de": {TEST_SPLIT: "Testdaten", "shifted": f"Drehmoment +{TORQUE_SHIFT_NM:g} Nm"},
}
# Every other text of the figures, per language. A pipeline step is "name|detail". The German
# files carry no title, their titles and subtitles reach the page through figures.json.
TEXT: dict[str, dict[str, str]] = {
    "en": {
        "pipeline.title": "From pinned data to a monitored API",
        "pipeline.subtitle": "Training writes to MLflow, serving reads only the exported bundle.",
        "pipeline.training": "Training",
        "pipeline.serving": "Serving",
        "pipeline.data": "Pinned data|AI4I 2020,\nSHA-256 checked",
        "pipeline.candidates": "4 candidates|MLflow runs,\nbest PR-AUC wins",
        "pipeline.registry": "Registry|champion alias\nonly if not worse",
        "pipeline.export": "Export|skops bundle,\nno pickle",
        "pipeline.bundle": "Bundle|model, threshold,\ndrift reference",
        "pipeline.api": "FastAPI|/predict, /drift,\nnon-root Docker",
        "pipeline.monitoring": "Monitoring|/metrics, PSI,\nJSON request logs",
        "pipeline.alt": "Training lane from pinned data through MLflow runs and registry to an "
        "exported bundle, serving lane where a FastAPI container loads only the bundle.",
        "metrics.title": "Test metrics with 95% bootstrap intervals",
        "metrics.subtitle": "{rows} unseen rows with only {failures} failures, "
        "so the intervals are wide.",
        "metrics.interval": "[{low}, {high}]",
        "metrics.alt": "Five test metrics with 95% bootstrap intervals: {values}.",
        "pr.title": "Boosted trees with physics features win on validation",
        "pr.subtitle": "Interpolated precision per candidate, the highest PR-AUC is registered.",
        "pr.chance": "random guess ({share} failures)",
        "pr.x": "Recall: share of failures found",
        "pr.y": "Precision",
        "pr.alt": "Precision-recall curves of four candidates on validation data. {best} is "
        "highest with PR-AUC {auc}, the logistic models are dashed and dotted.",
        "confusion.title": "{found} of {failures} failures found, {alarms} false alarms",
        "confusion.subtitle": "Test split at threshold {threshold}, chosen on validation data.",
        "confusion.tn": "normal runs passed",
        "confusion.fp": "false alarms",
        "confusion.fn": "failures missed",
        "confusion.tp": "failures found",
        "confusion.predicted_normal": "predicted: normal",
        "confusion.predicted_failure": "predicted: failure",
        "confusion.actual_normal": "actual:\nnormal",
        "confusion.actual_failure": "actual:\nfailure",
        "confusion.alt": "Confusion matrix on {rows} test operations: {found} failures found, "
        "{missed} missed, {alarms} false alarms, {passed} correct negatives.",
        "threshold.title": "The threshold trades missed failures against false alarms",
        "threshold.subtitle": "Validation split, serving uses the threshold with the highest F1.",
        "threshold.served": "served threshold {threshold}\n"
        "precision {precision}, recall {recall}, F1 {f1}",
        "threshold.x": "Decision threshold on the failure probability",
        "threshold.alt": "Precision, recall and F1 over the decision threshold. F1 is flat over "
        "a wide range, the served threshold {threshold} is marked with precision {precision}, "
        "recall {recall} and F1 {f1}.",
        "modes.title": "Rule-based failures are found, tool wear mostly is not",
        "modes.subtitle": "Share of test failures flagged per mode, with 95% Wilson interval.",
        "modes.empty": "no case in the test split",
        "modes.count": "{found} of {total}",
        "modes.alt": "Recall by failure mode with 95% Wilson intervals: {values}.",
        "modes.alt_item": "{mode} {found} of {total}",
        "modes.alt_empty": "no {mode} failure in the test split",
        "importance.title": "Torque carries the most signal",
        "importance.subtitle": "Drop in test PR-AUC when one input is shuffled "
        "(mean ± sd over 10 repeats).",
        "importance.alt": "Permutation importance: {top} matters most, {bottom} hardly at all.",
        "drift.title": "A miscalibrated torque sensor raises a drift alert",
        "drift.subtitle": "Population stability index per input against the training data, "
        "log scale.",
        "drift.warn": "warn ≥ {limit}",
        "drift.alert": "alert ≥ {limit}",
        "drift.alt": "Population stability index per input on a log scale with warn and alert "
        "zones: at most {test} on test data, {worst} for torque after a {shift} Nm sensor "
        "offset, {zone}.",
        "drift.zone_alert": "inside the alert zone",
        "drift.zone_warn": "inside the warn zone",
        "drift.zone_none": "below the warn limit",
    },
    "de": {
        "pipeline.title": "Von festgeschriebenen Daten zur überwachten API",
        "pipeline.subtitle": "Das Training schreibt nach MLflow, der Betrieb liest nur das "
        "exportierte Bundle.",
        "pipeline.training": "Training",
        "pipeline.serving": "Betrieb",
        "pipeline.data": "Feste Daten|AI4I 2020,\nSHA-256 geprüft",
        "pipeline.candidates": "4 Kandidaten|MLflow-Runs,\nPR-AUC entscheidet",
        "pipeline.registry": "Registry|Alias champion,\nnie schlechter",
        "pipeline.export": "Export|skops-Bundle,\nkein Pickle",
        "pipeline.bundle": "Bundle|Modell, Schwelle,\nDrift-Referenz",
        "pipeline.api": "FastAPI|/predict, /drift,\nDocker ohne root",
        "pipeline.monitoring": "Monitoring|/metrics, PSI,\nJSON-Logs",
        "pipeline.alt": "Trainingsstrecke von festgeschriebenen Daten über MLflow-Runs und "
        "Registry zum exportierten Bundle, Betriebsstrecke, in der ein FastAPI-Container nur "
        "das Bundle lädt.",
        "metrics.title": "Testmetriken mit 95-%-Bootstrap-Intervallen",
        "metrics.subtitle": "{rows} ungesehene Zeilen mit nur {failures} Ausfällen, "
        "daher sind die Intervalle breit.",
        "metrics.interval": "{low} bis {high}",
        "metrics.alt": "Fünf Testmetriken mit 95-%-Bootstrap-Intervallen: {values}.",
        "pr.title": "Boosting mit physikalischen Merkmalen gewinnt in der Validierung",
        "pr.subtitle": "Interpolierte Precision je Kandidat, registriert wird die höchste PR-AUC.",
        "pr.chance": "Zufallsraten ({share} Ausfälle)",
        "pr.x": "Recall: Anteil gefundener Ausfälle",
        "pr.y": "Precision",
        "pr.alt": "Precision-Recall-Kurven von vier Kandidaten auf den Validierungsdaten. "
        "{best} liegt mit PR-AUC {auc} am höchsten, die logistischen Modelle sind gestrichelt "
        "und gepunktet.",
        "confusion.title": "{found} von {failures} Ausfällen erkannt, {alarms} Fehlalarme",
        "confusion.subtitle": "Testdaten bei Schwelle {threshold}, gewählt auf den "
        "Validierungsdaten.",
        "confusion.tn": "Normalbetrieb erkannt",
        "confusion.fp": "Fehlalarme",
        "confusion.fn": "Ausfälle übersehen",
        "confusion.tp": "Ausfälle erkannt",
        "confusion.predicted_normal": "Vorhersage: normal",
        "confusion.predicted_failure": "Vorhersage: Ausfall",
        "confusion.actual_normal": "tatsächlich:\nnormal",
        "confusion.actual_failure": "tatsächlich:\nAusfall",
        "confusion.alt": "Konfusionsmatrix auf {rows} Testvorgängen: {found} Ausfälle erkannt, "
        "{missed} übersehen, {alarms} Fehlalarme, {passed} Vorgänge korrekt als normal "
        "eingestuft.",
        "threshold.title": "Die Schwelle wägt übersehene Ausfälle gegen Fehlalarme ab",
        "threshold.subtitle": "Validierungsdaten, im Betrieb gilt die Schwelle mit dem "
        "höchsten F1.",
        "threshold.served": "Schwelle im Betrieb {threshold}\n"
        "Precision {precision}, Recall {recall}, F1 {f1}",
        "threshold.x": "Entscheidungsschwelle auf der Ausfallwahrscheinlichkeit",
        "threshold.alt": "Precision, Recall und F1 über der Entscheidungsschwelle. F1 verläuft "
        "über einen weiten Bereich flach, markiert ist die Schwelle im Betrieb {threshold} mit "
        "Precision {precision}, Recall {recall} und F1 {f1}.",
        "modes.title": "Regelbasierte Ausfälle werden erkannt, Werkzeugverschleiß meist nicht",
        "modes.subtitle": "Anteil erkannter Testausfälle je Ausfallart, mit 95-%-Wilson-Intervall.",
        "modes.empty": "kein Fall in den Testdaten",
        "modes.count": "{found} von {total}",
        "modes.alt": "Recall je Ausfallart mit 95-%-Wilson-Intervallen: {values}.",
        "modes.alt_item": "{mode} {found} von {total}",
        "modes.alt_empty": "{mode} ohne Fall in den Testdaten",
        "importance.title": "Das Drehmoment trägt das meiste Signal",
        "importance.subtitle": "Rückgang der Test-PR-AUC, wenn ein Eingang permutiert wird "
        "(Mittel ± SD über 10 Wiederholungen).",
        "importance.alt": "Permutationswichtigkeit: {top} ist am wichtigsten, {bottom} spielt "
        "kaum eine Rolle.",
        "drift.title": "Ein falsch kalibrierter Drehmomentsensor löst Drift-Alarm aus",
        "drift.subtitle": "Population Stability Index je Eingang gegenüber den Trainingsdaten, "
        "logarithmische Skala.",
        "drift.warn": "Warnung\n≥ {limit}",
        "drift.alert": "Alarm\n≥ {limit}",
        "drift.alt": "Population Stability Index je Eingang auf logarithmischer Skala mit Warn- "
        "und Alarmzone: höchstens {test} auf den Testdaten, {worst} für das Drehmoment nach "
        "einem Sensorversatz von {shift} Nm, {zone}.",
        "drift.zone_alert": "in der Alarmzone",
        "drift.zone_warn": "in der Warnzone",
        "drift.zone_none": "unter der Warngrenze",
    },
}
WIDTH = figkit.WEB_WIDTH

Draw = Callable[[Evaluation, Theme], Figure]


def _font(key: str) -> float:
    size: float = figkit.tokens()["font"][key]
    return size


def text(key: str, **values: object) -> str:
    """Text ``key`` of :data:`TEXT` in the language being drawn, filled with ``values``."""
    return TEXT[figkit.language()][key].format(**values)


def feature_label(name: str) -> str:
    """Readable name of an input with its unit, or the name itself if it has none."""
    return FEATURE_LABELS[figkit.language()].get(name, name)


def _shifted(ev: Evaluation) -> str:
    return next(name for name in ev.drift if name != TEST_SPLIT)


def _scenario(key: str) -> str:
    return SCENARIO_LABELS[figkit.language()][TEST_SPLIT if key == TEST_SPLIT else "shifted"]


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
    figkit.title(ax, text("pipeline.title"), theme, subtitle=text("pipeline.subtitle"))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def boxes(*keys: str) -> list[list[str]]:
        return [text(f"pipeline.{key}").split("|") for key in keys]

    lanes = (
        (
            text("pipeline.training"),
            37,
            theme.series[0],
            boxes("data", "candidates", "registry", "export"),
        ),
        (text("pipeline.serving"), 11, theme.series[2], boxes("bundle", "api", "monitoring")),
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
        text("metrics.title"),
        theme,
        subtitle=text(
            "metrics.subtitle", rows=number(ev.test_rows, ","), failures=ev.test_failures
        ),
    )
    items = list(reversed(ev.intervals))
    colour = theme.series[0]
    for y, item in enumerate(items):
        ax.plot([item.low, item.high], [y, y], color=colour, linewidth=3, alpha=0.35)
        ax.plot(item.value, y, "o", markersize=7, color=colour)
        figkit.value(ax, item.high + 0.012, y, number(item.value, ".3f"), theme, va="center")
        figkit.value(
            ax,
            item.high + 0.07,
            y,
            text("metrics.interval", low=number(item.low, ".2f"), high=number(item.high, ".2f")),
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
    figkit.title(ax, text("pr.title"), theme, subtitle=text("pr.subtitle"))
    names = CANDIDATE_LABELS[figkit.language()]
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
            label=f"{names[curve.name]}   PR-AUC {number(curve.pr_auc, '.3f')}",
            zorder=4 - slot,
        )
    figkit.reference_line(
        ax,
        ev.prevalence,
        text("pr.chance", share=number(ev.prevalence, ".1%")),
        theme,
        axis="y",
        where=0.01,
    )
    ax.set_xlabel(text("pr.x"))
    ax.set_ylabel(text("pr.y"))
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.04)
    figkit.percent(ax, "x")
    figkit.percent(ax, "y")
    figkit.grid(ax, "y")
    handles, labels = ax.get_legend_handles_labels()
    slots = {label: CANDIDATE_SLOT[name] for name, label in names.items()}
    order = sorted(range(len(labels)), key=lambda i: slots[labels[i].split("   ")[0]])
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
        text("confusion.title", found=found, failures=failures, alarms=c["false_positive"]),
        theme,
        subtitle=text("confusion.subtitle", threshold=number(ev.threshold, ".3f")),
    )
    ax.set_xlim(-0.42, 2)
    ax.set_ylim(0, 2.14)
    ax.axis("off")
    right, wrong = theme.series[0], theme.series[1]
    cells = (
        (0, 1, c["true_negative"], text("confusion.tn"), right),
        (1, 1, c["false_positive"], text("confusion.fp"), wrong),
        (0, 0, c["false_negative"], text("confusion.fn"), wrong),
        (1, 0, c["true_positive"], text("confusion.tp"), right),
    )
    for col, row, count, meaning, colour in cells:
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
            number(count, ","),
            ha="center",
            va="center",
            fontsize=20,
            fontweight="semibold",
            color=theme.ink,
        )
        ax.text(col + 0.5, row + 0.3, meaning, ha="center", va="center", color=theme.secondary)
    for x, key in ((0.5, "predicted_normal"), (1.5, "predicted_failure")):
        ax.text(x, 2.03, text(f"confusion.{key}"), ha="center", va="bottom", color=theme.secondary)
    for y, key in ((1.5, "actual_normal"), (0.5, "actual_failure")):
        label = text(f"confusion.{key}")
        ax.text(-0.04, y, label, ha="right", va="center", color=theme.secondary)
    return figure


def threshold_sweep(ev: Evaluation, theme: Theme) -> Figure:
    """Precision, recall and F1 over the threshold, on validation."""
    figure, ax = plt.subplots(figsize=(WIDTH, 3.6))
    figkit.title(ax, text("threshold.title"), theme, subtitle=text("threshold.subtitle"))
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
        text(
            "threshold.served",
            threshold=number(ev.threshold, ".3f"),
            **{key: number(value, ".2f") for key, value in at.items()},
        ),
        theme,
        where=0.36,
        side="left",
    )
    ax.set_xlabel(text("threshold.x"))
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
    figkit.title(ax, text("modes.title"), theme, subtitle=text("modes.subtitle"))
    items = list(reversed(ev.modes))
    colour = theme.series[0]
    for y, m in enumerate(items):
        if not m.total:
            # A mode without a case keeps its row, so the figure lists every recorded mode.
            ax.text(
                0.01,
                y,
                text("modes.empty"),
                va="center",
                fontsize=_font("annotation"),
                color=theme.secondary,
            )
            continue
        low, high = wilson(m.detected, m.total)
        share = m.detected / m.total
        ax.plot([low, high], [y, y], color=colour, linewidth=3, alpha=0.35)
        ax.plot(share, y, "o", markersize=7, color=colour)
        figkit.value(ax, 1.15, y, number(share, ".0%"), theme, va="center", ha="right")
        count = text("modes.count", found=m.detected, total=m.total)
        figkit.value(ax, 1.17, y, count, theme, va="center", color=theme.secondary)
    modes = MODE_LABELS[figkit.language()]
    labels = [f"{modes[m.mode]} ({m.mode})" if m.mode != "none" else modes[m.mode] for m in items]
    ax.set_yticks(range(len(items)), labels)
    ax.set_xlim(0, 1.32)
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
    figkit.title(ax, text("importance.title"), theme, subtitle=text("importance.subtitle"))
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
        figkit.value(ax, x, y, number(item.mean, ".2f"), theme, va="center")
    ax.set_yticks(range(len(items)), [feature_label(i.feature) for i in items])
    ax.set_xlim(0, reach * 1.15)
    figkit.category_axis(ax, "y")
    figkit.grid(ax, "x")
    return figure


def drift(ev: Evaluation, theme: Theme) -> Figure:
    """PSI per feature for the unchanged test split and for shifted torque, on a log scale."""
    shifted = _shifted(ev)
    worst = max(ev.drift[shifted].items(), key=lambda kv: kv[1])
    figure, ax = plt.subplots(figsize=(WIDTH, 3.1))
    figkit.title(ax, text("drift.title"), theme, subtitle=text("drift.subtitle"))
    features = sorted(ev.drift[TEST_SPLIT], key=lambda f: ev.drift[shifted][f])
    smallest = min(v for scores in ev.drift.values() for v in scores.values() if v > 0)
    low, high = min(0.002, smallest / 1.5), 1.0
    # Status zones instead of two thin lines; their labels stay in the text colour.
    ax.axvspan(WARN, ALERT, color=theme.status["warning"], alpha=0.14, linewidth=0, zorder=0)
    ax.axvspan(ALERT, high, color=theme.status["critical"], alpha=0.10, linewidth=0, zorder=0)
    for start, key in ((WARN, "drift.warn"), (ALERT, "drift.alert")):
        label = text(key, limit=number(start, "g"))
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
        test = max(ev.drift[TEST_SPLIT][feature], low)
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
            label=_scenario(TEST_SPLIT) if y == 0 else None,
        )
        ax.plot(
            moved,
            y,
            "o",
            markersize=6,
            color=theme.status["critical"] if moved >= ALERT else theme.ink,
            zorder=4,
            label=_scenario(shifted) if y == 0 else None,
        )
    figkit.value(
        ax,
        worst[1] * 1.15,
        features.index(worst[0]),
        number(worst[1], ".2f"),
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
    shifted = max(ev.drift[_shifted(ev)].values())
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
    for i, (label, headline, note) in enumerate(tiles):
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
        ax.text(x + 2, 24, headline, fontsize=22, fontweight="semibold", va="center")
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


def _short(label: str) -> str:
    """A feature label without its unit, for running text."""
    return label.split(" (", maxsplit=1)[0]


def _alt_metrics(ev: Evaluation) -> str:
    values = ", ".join(f"{LABELS[i.metric]} {number(i.value, '.3f')}" for i in ev.intervals)
    return text("metrics.alt", values=values)


def _alt_pr(ev: Evaluation) -> str:
    best = max(ev.candidates, key=lambda curve: curve.pr_auc)
    label = CANDIDATE_LABELS[figkit.language()][best.name]
    return text("pr.alt", best=label[0].upper() + label[1:], auc=number(best.pr_auc, ".3f"))


def _alt_confusion(ev: Evaluation) -> str:
    c = ev.confusion
    return text(
        "confusion.alt",
        rows=number(ev.test_rows, ","),
        found=c["true_positive"],
        missed=c["false_negative"],
        alarms=c["false_positive"],
        passed=number(c["true_negative"], ","),
    )


def _alt_threshold(ev: Evaluation) -> str:
    sweep = ev.sweep
    at = {
        key: number(float(np.interp(ev.threshold, sweep["threshold"], sweep[key])), ".2f")
        for key in ("precision", "recall", "f1")
    }
    return text("threshold.alt", threshold=number(ev.threshold, ".3f"), **at)


def _alt_modes(ev: Evaluation) -> str:
    modes = MODE_LABELS[figkit.language()]
    english = figkit.language() == "en"
    items = []
    for m in ev.modes:
        mode = modes[m.mode].lower() if english else modes[m.mode]
        if m.total:
            items.append(text("modes.alt_item", mode=mode, found=m.detected, total=m.total))
        else:
            items.append(text("modes.alt_empty", mode=mode))
    return text("modes.alt", values=", ".join(items))


def _alt_importance(ev: Evaluation) -> str:
    ranked = sorted(ev.importance, key=lambda item: item.mean)
    top, bottom = (_short(feature_label(i.feature)) for i in (ranked[-1], ranked[0]))
    if figkit.language() == "en":
        top, bottom = top.lower(), bottom.lower()
    return text("importance.alt", top=top, bottom=bottom)


def _alt_drift(ev: Evaluation) -> str:
    worst = max(ev.drift[_shifted(ev)].values())
    zone = "alert" if worst >= ALERT else "warn" if worst >= WARN else "none"
    return text(
        "drift.alt",
        test=number(max(ev.drift[TEST_SPLIT].values()), ".3f"),
        worst=number(worst, ".2f"),
        shift=number(TORQUE_SHIFT_NM, "g"),
        zone=text(f"drift.zone_{zone}"),
    )


# Figures that the portfolio shows, with the key of their texts and their alt text in the
# language being drawn; the card is only for the README and previews.
PAGE: dict[str, tuple[str, Callable[[Evaluation], str]]] = {
    "pipeline": ("pipeline", lambda _: text("pipeline.alt")),
    "test-metrics": ("metrics", _alt_metrics),
    "pr-curves": ("pr", _alt_pr),
    "confusion": ("confusion", _alt_confusion),
    "threshold": ("threshold", _alt_threshold),
    "failure-modes": ("modes", _alt_modes),
    "importance": ("importance", _alt_importance),
    "drift": ("drift", _alt_drift),
}


def captions(name: str, ev: Evaluation) -> dict[str, figkit.Caption]:
    """Title, subtitle and alt text of figure ``name`` per language, for figures.json."""
    c = ev.confusion
    key, alt = PAGE[name]
    result = {}
    for language in LANGUAGES:
        with speaking(language):
            # Every title and subtitle takes what it needs of these, as in the drawn figure.
            values = {
                "rows": number(ev.test_rows, ","),
                "failures": c["true_positive"] + c["false_negative"],
                "found": c["true_positive"],
                "alarms": c["false_positive"],
                "threshold": number(ev.threshold, ".3f"),
            }
            result[language] = figkit.Caption(
                title=text(f"{key}.title", **values),
                subtitle=text(f"{key}.subtitle", **values),
                alt=alt(ev),
            )
    return result


def _page_draw(draw: Draw, ev: Evaluation, _language: str, theme: Theme) -> Figure:
    # figkit.render has set the language already, the draw functions read it from there.
    return draw(ev, theme)


def render(ev: Evaluation, directory: Path) -> list[Path]:
    """Write all figures in both themes to ``directory / "figures"`` and the page files.

    The README files are English with their titles. ``figures/page`` holds every chart without
    its title in each language as SVG, and ``figures.json`` with the texts for the portfolio.

    Returns:
        The written files.
    """
    out = directory / "figures"
    page = out / "page"
    written = []
    entries = []
    for name, (draw, outputs) in FIGURES.items():
        written += figkit.render(out, name, partial(draw, ev), outputs)
        if name in PAGE:
            entry = figkit.page(page, name, partial(_page_draw, draw, ev), captions(name, ev))
            entries.append(entry)
            written += [
                page / entry["files"][code][mode]
                for code in LANGUAGES
                for mode in ("light", "dark")
            ]
    written.append(figkit.write_manifest(page, entries))
    return written
