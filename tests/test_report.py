import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from figkit.check import check
from PIL import Image

from pmm import bundle, drift, figures, report
from pmm.data import MODES, NUMERIC, TARGET
from tests.conftest import readings


def with_modes(frame: pd.DataFrame) -> pd.DataFrame:
    """The invented readings with failure modes that match the rules of ``readings``."""
    power = frame["torque_nm"] * frame["rotational_speed_rpm"] * 2 * np.pi / 60
    out = frame.assign(**dict.fromkeys(MODES, 0))
    out["PWF"] = (power > 8500).astype(int)
    out["TWF"] = (frame["tool_wear_min"] > 235).astype(int)
    return out


@pytest.fixture(scope="module")
def evaluation(served: bundle.Bundle) -> report.Evaluation:
    return report.evaluate(with_modes(readings(1500)), served, resamples=200)


def test_confusion_and_intervals_describe_the_test_split(evaluation: report.Evaluation) -> None:
    c = evaluation.confusion

    assert sum(c.values()) == evaluation.test_rows == 225
    assert c["true_positive"] + c["false_negative"] == evaluation.test_failures
    assert [i.metric for i in evaluation.intervals] == list(report.METRICS)
    for interval in evaluation.intervals:
        assert 0 <= interval.low <= interval.value <= interval.high <= 1


def test_modes_count_only_labelled_failures(evaluation: report.Evaluation) -> None:
    by_mode = {m.mode: m for m in evaluation.modes}

    assert by_mode["PWF"].total + by_mode["TWF"].total >= evaluation.test_failures
    assert by_mode["RNF"].total == 0
    assert by_mode["none"].total == 0
    assert all(m.detected <= m.total for m in evaluation.modes)


def test_mode_recall_has_a_row_for_failures_without_a_mode() -> None:
    test = pd.DataFrame(
        {TARGET: [1, 1, 1, 0], **{m: [0, 0, 0, 0] for m in MODES}} | {"HDF": [1, 0, 0, 0]}
    )
    flagged = np.array([True, True, False, True])

    rows = {m.mode: (m.detected, m.total) for m in report.mode_recall(test, flagged)}

    assert rows["HDF"] == (1, 1)
    assert rows["none"] == (1, 2)


def test_shifted_torque_is_the_worst_drift(evaluation: report.Evaluation) -> None:
    shifted = evaluation.drift["torque +8 Nm"]

    assert max(shifted, key=lambda f: shifted[f]) == "torque_nm"
    assert shifted["torque_nm"] > drift.ALERT
    assert set(evaluation.drift["test split"]) == set(shifted)


def test_importance_covers_every_raw_input(evaluation: report.Evaluation) -> None:
    names = [i.feature for i in evaluation.importance]

    assert sorted(names) == sorted(["type", *NUMERIC])
    assert names[0] in {"torque_nm", "rotational_speed_rpm", "tool_wear_min"}


def test_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    truth = rng.random(400) < 0.1
    probability = np.clip(truth * 0.6 + rng.random(400) * 0.5, 0, 1)

    first = report.bootstrap(truth, probability, 0.5, resamples=100, seed=1)

    assert first == report.bootstrap(truth, probability, 0.5, resamples=100, seed=1)


def test_tables_round_the_point_estimate_only_once() -> None:
    # 37 of 51 is 0.72549..., which rounds to 0.725; rounding 0.7255 again would give 0.726.
    truth = np.array([True] * 51 + [False] * 49)
    probability = np.array([0.9] * 37 + [0.1] * 14 + [0.1] * 49)

    recall = next(
        i for i in report.bootstrap(truth, probability, 0.5, resamples=10) if i.metric == "recall"
    )

    assert f"{recall.value:.3f}" == "0.725"


def test_written_summary_and_tables(evaluation: report.Evaluation, tmp_path: Path) -> None:
    evaluation.latency = [report.Latency(1, 1.5, 2.0)]

    summary, tables = report.write(evaluation, tmp_path)

    data = json.loads(summary.read_text(encoding="utf-8"))
    assert "sweep" not in data
    assert set(data["candidates"]) == set(c.name for c in evaluation.candidates)
    text = tables.read_text(encoding="utf-8")
    assert "| PR-AUC |" in text
    assert "| Random (RNF) | 0 of 0 | n/a |" in text
    assert "| 1 | 1.5 ms | 2.0 ms |" in text


def test_latency_measures_both_batch_sizes(served: bundle.Bundle) -> None:
    timings = report.latency(served, readings(50), repeats=2)

    assert [t.batch for t in timings] == [1, 1000]
    assert all(0 < t.median_ms <= t.p95_ms for t in timings)


def test_figures_are_written_in_both_themes_as_transparent_svg_and_png(
    evaluation: report.Evaluation, tmp_path: Path
) -> None:
    written = figures.render(evaluation, tmp_path)

    names = {p.relative_to(tmp_path).as_posix() for p in written}
    for name in set(figures.FIGURES) - {"card"}:
        for suffix in ("", "-dark"):
            assert {f"figures/{name}{suffix}.svg", f"figures/{name}{suffix}.png"} <= names
    assert {"figures/card.png", "figures/card-dark.png"} <= names
    with Image.open(tmp_path / "figures" / "card.png") as card:
        assert card.size == (1200, 630)
    with Image.open(tmp_path / "figures" / "pr-curves-dark.png") as chart:
        assert chart.convert("RGBA").getpixel((0, 0))[3] == 0  # type: ignore[index]
    # Pairs, transparency and file sizes, as `figkit check` tests them in CI.
    assert check(tmp_path / "figures") == []
