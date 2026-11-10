"""Input drift: the population stability index (PSI) of each numeric feature."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from pmm.data import NUMERIC

# Common rule of thumb: below 0.1 stable, 0.1 to 0.25 a moderate shift, above 0.25 a large one.
WARN = 0.1
ALERT = 0.25
_EPSILON = 1e-4

# Fewest readings a batch needs. Without any drift a batch of n readings already has an expected PSI
# of about (bins - 1) * (1/n + 1/N) against N training rows, from sampling noise alone: 0.091 for
# n = 100 and N = 7,000, which is almost the warning threshold, and 0.019 for n = 500. On AI4I,
# random batches of 100 readings without drift warned for at least one of the five features in 91%
# of 1,000 trials, batches of 500 in none.
MIN_BATCH = 500


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """PSI of one feature and its level."""

    feature: str
    psi: float
    level: Literal["ok", "warn", "alert"]


def psi(reference: dict[str, list[float]], values: np.ndarray) -> float:
    """PSI of ``values`` against the bins of the reference.

    Empty bins get a small share, so the logarithm stays finite.
    """
    edges = np.asarray(reference["edges"])
    expected = np.maximum(np.asarray(reference["shares"]), _EPSILON)
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    actual = np.maximum(counts / max(len(values), 1), _EPSILON)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def report(reference: dict[str, dict[str, list[float]]], frame: pd.DataFrame) -> list[FeatureDrift]:
    """PSI per numeric feature of a batch of readings, worst first."""
    drifts = []
    for feature in NUMERIC:
        value = psi(reference[feature], frame[feature].to_numpy(dtype=float))
        level: Literal["ok", "warn", "alert"] = (
            "alert" if value > ALERT else "warn" if value > WARN else "ok"
        )
        drifts.append(FeatureDrift(feature, round(value, 4), level))
    return sorted(drifts, key=lambda d: d.psi, reverse=True)
