"""Train the candidates with MLflow tracking, register the best and export the champion.

Every candidate becomes one MLflow run with its parameters, validation metrics and model. The
best run by validation PR-AUC is registered; it gets the ``champion`` alias only if it is at
least as good as the current champion, so a worse retraining never replaces a better model.
The test split is scored once, for the registered model only.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from pmm import bundle
from pmm.data import TARGET, ZIP_SHA256, Splits
from pmm.model import TRUSTED_TYPES, Kind, Scores, best_threshold, build, score

MODEL_NAME = "ai4i-machine-failure"
ALIAS = "champion"
CANDIDATES: tuple[tuple[Kind, bool], ...] = (
    ("logistic", False),
    ("logistic", True),
    ("boosting", False),
    ("boosting", True),
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One trained candidate and where MLflow keeps it."""

    name: str
    run_id: str
    model_uri: str
    val: Scores


@dataclass(frozen=True, slots=True)
class Outcome:
    """Result of one training: all candidates, the registered version and its test scores."""

    candidates: list[Candidate]
    best: Candidate
    version: str
    test: Scores
    promoted: bool


def _probability(model: object, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(frame.drop(columns=[TARGET])))[:, 1]  # type: ignore[attr-defined]


def run(
    splits: Splits,
    tracking_uri: str,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> Outcome:
    """Train all candidates, register the best and promote it if it beats the champion."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MODEL_NAME)
    features = splits.train.drop(columns=[TARGET])
    candidates = []
    for kind, engineered in CANDIDATES:
        name = f"{kind}{' + features' if engineered else ''}"
        with mlflow.start_run(run_name=name) as active:
            model = build(kind, engineered=engineered, seed=seed).fit(
                features, splits.train[TARGET]
            )
            probability = _probability(model, splits.val)
            val = score(
                splits.val[TARGET],
                probability,
                best_threshold(splits.val[TARGET], probability),
            )
            mlflow.log_params(
                {
                    "kind": kind,
                    "engineered": engineered,
                    "seed": seed,
                    "data_sha256": ZIP_SHA256,
                }
            )
            mlflow.log_metrics({f"val_{k}": v for k, v in val.to_dict().items()})
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=features.head(3),
                skops_trusted_types=TRUSTED_TYPES,
            )
            candidates.append(Candidate(name, active.info.run_id, info.model_uri, val))
            log(f"{name:<22} val PR-AUC {val.pr_auc:.3f}  F1 {val.f1:.3f}")

    best = max(candidates, key=lambda c: c.val.pr_auc)
    model = mlflow.sklearn.load_model(best.model_uri)
    probability = _probability(model, splits.test)
    test = score(splits.test[TARGET], probability, best.val.threshold)
    with mlflow.start_run(run_id=best.run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in test.to_dict().items()})

    client = MlflowClient()
    version = mlflow.register_model(best.model_uri, MODEL_NAME).version
    client.set_model_version_tag(MODEL_NAME, version, "val_pr_auc", str(best.val.pr_auc))
    client.set_model_version_tag(MODEL_NAME, version, "threshold", str(best.val.threshold))
    champion = _champion_pr_auc(client)
    promoted = champion is None or best.val.pr_auc >= champion
    if promoted:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    status = "promoted" if promoted else "kept the champion"
    log(f"registered {best.name} as version {version}, {status}")
    return Outcome(candidates, best, str(version), test, promoted)


def _champion_pr_auc(client: MlflowClient) -> float | None:
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except MlflowException:
        return None
    return float(version.tags["val_pr_auc"])


def export(tracking_uri: str, train: pd.DataFrame, directory: Path) -> Path:
    """Write the current champion as a serving bundle, with the drift reference of ``train``.

    Raises:
        MlflowException: If no model has the champion alias.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    run_id = str(version.run_id)
    metrics = client.get_run(run_id).data.metrics
    metadata = bundle.Metadata(
        name=MODEL_NAME,
        version=str(version.version),
        run_id=run_id,
        threshold=float(version.tags["threshold"]),
        test_metrics={
            k.removeprefix("test_"): v for k, v in metrics.items() if k.startswith("test_")
        },
        reference=bundle.reference_of(train),
    )
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    return bundle.save(directory, model, metadata)
