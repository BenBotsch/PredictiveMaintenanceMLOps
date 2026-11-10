"""Evaluation of the served model beyond the headline metrics, for the README and the figures.

Everything here is computed from the exported bundle and the pinned dataset, so the numbers in
the README can be regenerated with ``pmm report``. The test split is scored with the bundle's
threshold, which was chosen on validation data; nothing here tunes on the test split.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve

from pmm import drift
from pmm.api import create_app
from pmm.bundle import Bundle
from pmm.data import MODES, NUMERIC, TARGET, Splits, split
from pmm.model import CANDIDATES, build, candidate_name, score

METRICS = ("pr_auc", "roc_auc", "precision", "recall", "f1")
RESAMPLES = 1000
# The shift of the drift demonstration: a torque sensor that reads 8 Nm too high.
TORQUE_SHIFT_NM = 8.0
MODE_NAMES = {
    "TWF": "Tool wear",
    "HDF": "Heat dissipation",
    "PWF": "Power",
    "OSF": "Overstrain",
    "RNF": "Random",
}

Floats = list[float]


@dataclass(frozen=True, slots=True)
class Curve:
    """Precision over recall of one candidate on validation."""

    name: str
    pr_auc: float
    recall: Floats
    precision: Floats


@dataclass(frozen=True, slots=True)
class Interval:
    """A test metric with its 95 % bootstrap interval."""

    metric: str
    value: float
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class ModeRecall:
    """How many test failures of one mode the model flags."""

    mode: str
    name: str
    detected: int
    total: int


@dataclass(frozen=True, slots=True)
class Importance:
    """Mean drop of test PR-AUC when one input column is shuffled."""

    feature: str
    mean: float
    std: float


@dataclass(frozen=True, slots=True)
class Latency:
    """In-process latency of ``/predict``, in milliseconds."""

    batch: int
    median_ms: float
    p95_ms: float


@dataclass(slots=True)
class Evaluation:
    """Everything the README and the figures show."""

    model_version: str
    threshold: float
    test_rows: int
    test_failures: int
    candidates: list[Curve]
    prevalence: float
    sweep: dict[str, Floats]
    confusion: dict[str, int]
    intervals: list[Interval]
    modes: list[ModeRecall]
    importance: list[Importance]
    drift: dict[str, dict[str, float]]
    latency: list[Latency] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        """The evaluation without the curves, for ``results/report.json``."""
        out = asdict(self)
        out["candidates"] = {c.name: round(c.pr_auc, 4) for c in self.candidates}
        out.pop("sweep")
        return out


def _probability(model: object, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
    inputs = frame.drop(columns=[TARGET, *MODES], errors="ignore")
    return np.asarray(model.predict_proba(inputs))[:, 1]  # type: ignore[attr-defined]


def _rounded(values: npt.ArrayLike, digits: int = 4) -> Floats:
    return [round(float(v), digits) for v in np.asarray(values)]


def candidate_curves(splits: Splits, seed: int = 0) -> list[Curve]:
    """Refit the training candidates and return their validation PR curves.

    The fits are deterministic, so these are the curves of the runs that ``pmm train`` logged.
    """
    features = splits.train.drop(columns=[TARGET, *MODES], errors="ignore")
    curves = []
    for kind, engineered in CANDIDATES:
        model = build(kind, engineered=engineered, seed=seed).fit(features, splits.train[TARGET])
        probability = _probability(model, splits.val)
        precision, recall, _ = precision_recall_curve(splits.val[TARGET], probability)
        pr_auc = score(splits.val[TARGET], probability, 0.5).pr_auc
        name = candidate_name(kind, engineered=engineered)
        curves.append(Curve(name, pr_auc, _rounded(recall), _rounded(precision)))
    return curves


def threshold_sweep(
    truth: npt.ArrayLike, probability: npt.NDArray[np.float64]
) -> dict[str, Floats]:
    """Precision, recall and F1 over the decision threshold."""
    precision, recall, thresholds = precision_recall_curve(truth, probability)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "threshold": _rounded(thresholds),
        "precision": _rounded(precision),
        "recall": _rounded(recall),
        "f1": _rounded(f1),
    }


def bootstrap(
    truth: npt.ArrayLike,
    probability: npt.NDArray[np.float64],
    threshold: float,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> list[Interval]:
    """95 % percentile intervals of the test metrics, resampled within each class.

    Resampling within each class keeps the number of failures fixed, so every resample has
    failures to find and the PR-AUC stays defined.
    """
    actual = np.asarray(truth).astype(bool)
    positives, negatives = np.flatnonzero(actual), np.flatnonzero(~actual)
    rng = np.random.default_rng(seed)
    samples = {m: np.empty(resamples) for m in METRICS}
    for i in range(resamples):
        rows = np.concatenate(
            [
                rng.choice(positives, len(positives)),
                rng.choice(negatives, len(negatives)),
            ]
        )
        scores = asdict(score(actual[rows], probability[rows], threshold))
        for metric in METRICS:
            samples[metric][i] = scores[metric]
    point = asdict(score(actual, probability, threshold))
    # Six digits, so that the three-digit tables never round an already rounded value.
    return [
        Interval(
            metric,
            round(point[metric], 6),
            round(float(np.quantile(samples[metric], 0.025)), 6),
            round(float(np.quantile(samples[metric], 0.975)), 6),
        )
        for metric in METRICS
    ]


def mode_recall(test: pd.DataFrame, flagged: npt.NDArray[np.bool_]) -> list[ModeRecall]:
    """Share of the failures of each mode that the model flags.

    A failure can have several modes, so the totals add up to more than the failures. AI4I also
    has failures without any mode and random failures (RNF) that are mostly labelled as no
    failure; only rows labelled as failures count, and failures without a mode get their own row.
    """
    failed = test[TARGET].to_numpy() == 1
    modes = test[list(MODES)].to_numpy() == 1
    out = []
    for i, mode in enumerate(MODES):
        rows = failed & modes[:, i]
        out.append(ModeRecall(mode, MODE_NAMES[mode], int(flagged[rows].sum()), int(rows.sum())))
    rows = failed & ~modes.any(axis=1)
    out.append(ModeRecall("none", "No mode recorded", int(flagged[rows].sum()), int(rows.sum())))
    return out


def importance(
    served: Bundle, test: pd.DataFrame, repeats: int = 10, seed: int = 0
) -> list[Importance]:
    """Permutation importance of the raw inputs, as the drop in test PR-AUC, largest first.

    The engineered features are computed inside the pipeline, so shuffling a raw input also
    breaks every feature derived from it.
    """
    inputs = test.drop(columns=[TARGET, *MODES], errors="ignore")
    result = permutation_importance(
        served.model,
        inputs,
        test[TARGET],
        scoring="average_precision",
        n_repeats=repeats,
        random_state=seed,
    )
    items = [
        Importance(column, round(float(m), 4), round(float(s), 4))
        for column, m, s in zip(
            inputs.columns, result.importances_mean, result.importances_std, strict=True
        )
    ]
    return sorted(items, key=lambda i: i.mean, reverse=True)


def drift_demo(served: Bundle, test: pd.DataFrame) -> dict[str, dict[str, float]]:
    """PSI per feature of the unchanged test split and of the same rows with shifted torque."""
    shifted = test.assign(torque_nm=test["torque_nm"] + TORQUE_SHIFT_NM)
    reference = served.metadata.reference
    return {
        name: {d.feature: d.psi for d in drift.report(reference, frame)}
        for name, frame in (("test split", test), ("torque +8 Nm", shifted))
    }


def latency(served: Bundle, test: pd.DataFrame, repeats: int = 50) -> list[Latency]:
    """Median and 95th percentile of ``/predict`` in process, for one reading and for 1,000.

    This measures validation, the model and serialisation, without network or server overhead.
    """
    client = TestClient(create_app(served))
    records = test[["type", *NUMERIC]].to_dict("records")
    out = []
    for batch in (1, 1000):
        body = {"readings": [records[i % len(records)] for i in range(batch)]}
        client.post("/predict", json=body)  # warm-up
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            client.post("/predict", json=body).raise_for_status()
            timings.append((time.perf_counter() - start) * 1000)
        out.append(
            Latency(
                batch,
                round(float(np.median(timings)), 2),
                round(float(np.quantile(timings, 0.95)), 2),
            )
        )
    return out


def evaluate(
    frame: pd.DataFrame, served: Bundle, *, seed: int = 0, resamples: int = RESAMPLES
) -> Evaluation:
    """Evaluate the served model on the split of ``frame``, which includes the mode columns."""
    splits = split(frame)
    test, val = splits.test, splits.val
    threshold = served.metadata.threshold
    probability = _probability(served.model, test)
    flagged = probability >= threshold
    actual = test[TARGET].to_numpy() == 1
    return Evaluation(
        model_version=served.metadata.version,
        threshold=round(threshold, 4),
        test_rows=len(test),
        test_failures=int(actual.sum()),
        candidates=candidate_curves(splits, seed),
        prevalence=round(float(val[TARGET].mean()), 4),
        sweep=threshold_sweep(val[TARGET], _probability(served.model, val)),
        confusion={
            "true_negative": int((~actual & ~flagged).sum()),
            "false_positive": int((~actual & flagged).sum()),
            "false_negative": int((actual & ~flagged).sum()),
            "true_positive": int((actual & flagged).sum()),
        },
        intervals=bootstrap(test[TARGET], probability, threshold, resamples, seed),
        modes=mode_recall(test, flagged),
        importance=importance(served, test, seed=seed),
        drift=drift_demo(served, test),
    )


# An en dash for ranges and empty cells in the tables.
RANGE = chr(0x2013)  # en dash between the ends of an interval, without spaces
LABELS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}


def tables(evaluation: Evaluation) -> str:
    """The README tables as Markdown."""
    lines = [
        f"Test split: {evaluation.test_rows:,} rows, {evaluation.test_failures} failures, "
        f"threshold {evaluation.threshold:.3f}, {RESAMPLES:,} stratified bootstrap resamples.",
        "",
        "| Metric | Test | 95 % interval |",
        "| --- | ---: | ---: |",
    ]
    lines += [
        f"| {LABELS[i.metric]} | {i.value:.3f} | {i.low:.3f}{RANGE}{i.high:.3f} |"
        for i in evaluation.intervals
    ]
    lines += ["", "| Failure mode | Detected | Recall |", "| --- | ---: | ---: |"]
    lines += [
        f"| {m.name}{f' ({m.mode})' if m.mode != 'none' else ''} | {m.detected} of {m.total} | "
        f"{f'{m.detected / m.total:.0%}' if m.total else 'n/a'} |"
        for m in evaluation.modes
    ]
    lines += ["", "| Input | PR-AUC drop when shuffled |", "| --- | ---: |"]
    lines += [f"| `{i.feature}` | {i.mean:.3f} ± {i.std:.3f} |" for i in evaluation.importance]
    lines += ["", "| Batch | Median | p95 |", "| ---: | ---: | ---: |"]
    lines += [
        f"| {lat.batch:,} | {lat.median_ms:.1f} ms | {lat.p95_ms:.1f} ms |"
        for lat in evaluation.latency
    ]
    return "\n".join(lines) + "\n"


def write(evaluation: Evaluation, directory: Path) -> list[Path]:
    """Write ``report.json`` and ``tables.md`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    summary = directory / "report.json"
    summary.write_text(
        json.dumps(evaluation.summary(), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    markdown = directory / "tables.md"
    markdown.write_text(tables(evaluation), encoding="utf-8", newline="\n")
    return [summary, markdown]
