from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pmm import bundle, train
from pmm.api import create_app
from pmm.data import TARGET, Splits
from tests.conftest import readings


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    # MLflow writes artifacts relative to the working directory.
    monkeypatch.chdir(tmp_path)
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


@pytest.fixture
def served(splits: Splits, tracking: str, tmp_path: Path) -> bundle.Bundle:
    train.run(splits, tracking, log=lambda _: None)
    return bundle.load(train.export(tracking, splits.train, tmp_path / "bundle"))


def test_training_registers_the_best_candidate_and_promotes_it(
    splits: Splits, tracking: str
) -> None:
    outcome = train.run(splits, tracking, log=lambda _: None)

    assert len(outcome.candidates) == len(train.CANDIDATES)
    assert outcome.best.val.pr_auc == max(c.val.pr_auc for c in outcome.candidates)
    assert (outcome.version, outcome.promoted) == ("1", True)


def test_a_worse_retraining_keeps_the_champion(splits: Splits, tracking: str) -> None:
    train.run(splits, tracking, log=lambda _: None)
    noisy = Splits(
        splits.train.sample(frac=1.0, random_state=1).assign(
            **{TARGET: splits.train[TARGET].sample(frac=1.0, random_state=2).to_numpy()}
        ),
        splits.val,
        splits.test,
    )

    outcome = train.run(noisy, tracking, log=lambda _: None)

    assert (outcome.version, outcome.promoted) == ("2", False)
    assert train.export(tracking, splits.train, Path("bundle")).joinpath("metadata.json").is_file()
    assert bundle.load(Path("bundle")).metadata.version == "1"


def test_bundle_keeps_threshold_and_test_metrics(served: bundle.Bundle) -> None:
    assert 0 < served.metadata.threshold < 1
    assert {"pr_auc", "recall", "precision"} <= set(served.metadata.test_metrics)


READING = {
    "type": "M",
    "air_temp_k": 300.0,
    "process_temp_k": 310.0,
    "rotational_speed_rpm": 1500.0,
    "torque_nm": 40.0,
    "tool_wear_min": 10.0,
}


def test_predict_flags_a_high_power_reading(served: bundle.Bundle) -> None:
    client = TestClient(create_app(served))
    risky = READING | {"torque_nm": 70.0, "rotational_speed_rpm": 1500.0}

    response = client.post("/predict", json={"readings": [READING, risky]})

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "1"
    assert [p["failure"] for p in body["predictions"]] == [False, True]


@pytest.mark.parametrize(
    "change",
    [{"type": "X"}, {"torque_nm": -1}, {"rotational_speed_rpm": "fast"}, {"extra": 1}],
)
def test_predict_rejects_invalid_readings(served: bundle.Bundle, change: dict[str, object]) -> None:
    response = TestClient(create_app(served)).post(
        "/predict", json={"readings": [READING | change]}
    )

    assert response.status_code == 422


@pytest.mark.parametrize("count", [0, 1001])
def test_predict_rejects_empty_and_oversized_batches(served: bundle.Bundle, count: int) -> None:
    response = TestClient(create_app(served)).post("/predict", json={"readings": [READING] * count})

    assert response.status_code == 422


def test_drift_needs_a_batch_and_reports_a_shift(served: bundle.Bundle) -> None:
    client = TestClient(create_app(served))
    shifted = readings(300, seed=5)
    shifted["air_temp_k"] += 8

    too_small = client.post("/drift", json={"readings": [READING]})
    response = client.post(
        "/drift", json={"readings": shifted.drop(columns=[TARGET]).to_dict("records")}
    )

    assert too_small.status_code == 422
    assert response.json()["alert"] is True
    assert response.json()["features"][0]["feature"] == "air_temp_k"


def test_health_and_metadata(served: bundle.Bundle) -> None:
    client = TestClient(create_app(served))

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/metadata").json()["name"] == train.MODEL_NAME
