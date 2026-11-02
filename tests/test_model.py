import numpy as np
import pytest

from pmm.data import TARGET, Splits
from pmm.model import add_features, best_threshold, build, score
from tests.conftest import readings


def test_engineered_features_follow_the_physics() -> None:
    frame = readings(1)
    frame.loc[0, ["torque_nm", "rotational_speed_rpm", "tool_wear_min"]] = [40.0, 1500.0, 10.0]
    frame.loc[0, ["air_temp_k", "process_temp_k"]] = [300.0, 310.5]

    row = add_features(frame).iloc[0]

    assert row["power_w"] == pytest.approx(40 * 1500 * 2 * np.pi / 60)
    assert row["temp_diff_k"] == pytest.approx(10.5)
    assert row["strain_nm_min"] == pytest.approx(400)


@pytest.mark.parametrize("kind", ["logistic", "boosting"])
def test_engineered_features_help_on_a_power_rule(kind: str, splits: Splits) -> None:
    def pr_auc(engineered: bool) -> float:
        model = build(kind, engineered=engineered).fit(  # type: ignore[arg-type]
            splits.train.drop(columns=[TARGET]), splits.train[TARGET]
        )
        probability = model.predict_proba(splits.val.drop(columns=[TARGET]))[:, 1]
        return score(splits.val[TARGET], probability, 0.5).pr_auc

    assert pr_auc(engineered=True) > pr_auc(engineered=False) - 0.01
    assert pr_auc(engineered=True) > 0.8


def test_threshold_maximises_f1() -> None:
    truth = np.array([0, 0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.6, 0.7, 0.9])

    threshold = best_threshold(truth, probability)

    assert threshold == pytest.approx(0.7)
    assert score(truth, probability, threshold).f1 == pytest.approx(1.0)


def test_score_without_positive_predictions_is_zero() -> None:
    result = score(np.array([0, 1]), np.array([0.1, 0.2]), threshold=0.9)

    assert (result.precision, result.recall, result.f1) == (0, 0, 0)
