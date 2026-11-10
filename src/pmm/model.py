"""Features, candidate models, decision threshold and metrics."""

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from pmm.data import NUMERIC, TYPES

Kind = Literal["logistic", "boosting"]
# The candidates of every training: a linear baseline and boosted trees, each without and with
# the engineered features.
CANDIDATES: tuple[tuple[Kind, bool], ...] = (
    ("logistic", False),
    ("logistic", True),
    ("boosting", False),
    ("boosting", True),
)
ENGINEERED = ("power_w", "temp_diff_k", "strain_nm_min")
# Types skops does not trust by default: own code inside the pipeline, and the boosting trees,
# whose raw node indices are safe here because only models trained by this project are loaded.
TRUSTED_TYPES = [
    "pmm.model.add_features",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add mechanical power, the temperature difference and the tool strain.

    These follow from the physics of the process: power from torque and speed, heat dissipation
    from the temperature gap, wear from torque times tool time.
    """
    out = frame.copy()
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * 2 * math.pi / 60
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["strain_nm_min"] = out["torque_nm"] * out["tool_wear_min"]
    return out


def candidate_name(kind: Kind, *, engineered: bool) -> str:
    """The name of a candidate in MLflow and in the results, e.g. ``boosting + features``."""
    return f"{kind}{' + features' if engineered else ''}"


def build(kind: Kind, *, engineered: bool, seed: int = 0) -> Pipeline:
    """A pipeline from raw readings to a failure probability."""
    numeric = [*NUMERIC, *(ENGINEERED if engineered else ())]
    scale = StandardScaler() if kind == "logistic" else "passthrough"
    columns = ColumnTransformer(
        [
            ("type", OneHotEncoder(categories=[list(TYPES)]), ["type"]),
            ("numeric", scale, numeric),
        ]
    )
    if kind == "logistic":
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        classifier = HistGradientBoostingClassifier(random_state=seed)
    steps = [("features", FunctionTransformer(add_features))] if engineered else []
    return Pipeline([*steps, ("columns", columns), ("classifier", classifier)])


@dataclass(frozen=True, slots=True)
class Scores:
    """Metrics of failure probabilities against the truth, at one decision threshold."""

    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float

    def to_dict(self) -> dict[str, float]:
        """The metrics as a flat dictionary, e.g. for MLflow."""
        return {k: round(v, 4) for k, v in asdict(self).items()}


def best_threshold(truth: npt.ArrayLike, probability: npt.NDArray[np.float64]) -> float:
    """The threshold with the highest F1, chosen on validation data only."""
    precision, recall, thresholds = precision_recall_curve(truth, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def score(truth: npt.ArrayLike, probability: npt.NDArray[np.float64], threshold: float) -> Scores:
    """PR-AUC, ROC-AUC and precision, recall and F1 at ``threshold``."""
    actual = np.asarray(truth).astype(bool)
    predicted = probability >= threshold
    hits = int((actual & predicted).sum())
    precision = hits / max(int(predicted.sum()), 1)
    recall = hits / max(int(actual.sum()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return Scores(
        pr_auc=float(average_precision_score(actual, probability)),
        roc_auc=float(roc_auc_score(actual, probability)),
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=threshold,
    )
