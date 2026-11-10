from pathlib import Path

import numpy as np
import pytest

from pmm.bundle import reference_of
from pmm.cli import main
from pmm.drift import ALERT, MIN_BATCH, WARN, psi, report
from tests.conftest import readings


def test_same_distribution_is_stable_and_a_shift_alerts() -> None:
    reference = reference_of(readings(5000, seed=0))
    current = readings(2000, seed=1)
    current["torque_nm"] += 15

    by_feature = {d.feature: d for d in report(reference, current)}

    assert by_feature["torque_nm"].level == "alert"
    assert by_feature["torque_nm"].psi > ALERT
    assert all(d.psi < WARN for name, d in by_feature.items() if name != "torque_nm")
    assert report(reference, current)[0].feature == "torque_nm"


def test_values_outside_every_bin_stay_finite() -> None:
    reference = {"edges": [1.0, 2.0], "shares": [0.0, 0.5, 0.5]}

    value = psi(reference, np.array([5.0, 6.0]))

    assert np.isfinite(value) and value > ALERT


def test_the_smallest_batch_without_drift_does_not_warn() -> None:
    # As many reference rows as the training split of AI4I. At 100 readings most batches would warn.
    reference = reference_of(readings(7000, seed=0))

    worst = [report(reference, readings(MIN_BATCH, seed=seed))[0].psi for seed in range(1, 51)]

    assert max(worst) < WARN


def test_cli_rejects_a_batch_below_the_minimum(tmp_path: Path) -> None:
    csv = tmp_path / "readings.csv"
    readings(MIN_BATCH - 1).to_csv(csv, index=False)

    with pytest.raises(SystemExit, match=f"at least {MIN_BATCH} readings"):
        main(["drift", str(csv)])
