"""The serving bundle: the model, its threshold and the reference for drift checks.

The API and the container read only this directory, so serving needs neither MLflow nor the
training data.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io
from sklearn.pipeline import Pipeline

from pmm.data import NUMERIC
from pmm.model import TRUSTED_TYPES

# skops instead of pickle: loading runs no code beyond the trusted types.
MODEL_FILE = "model.skops"
METADATA_FILE = "metadata.json"
BINS = 10


@dataclass(frozen=True, slots=True)
class Metadata:
    """What the API reports about the model it serves."""

    name: str
    version: str
    run_id: str
    threshold: float
    test_metrics: dict[str, float]
    # Per numeric feature: inner decile edges and the share of training rows in each bin.
    reference: dict[str, dict[str, list[float]]]


@dataclass(frozen=True, slots=True)
class Bundle:
    """A loaded model with its metadata."""

    model: Pipeline
    metadata: Metadata


def reference_of(train: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    """Decile edges and bin shares of each numeric feature in the training data."""
    reference = {}
    for column in NUMERIC:
        values = train[column].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, BINS + 1)[1:-1]))
        counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
        reference[column] = {
            "edges": [float(e) for e in edges],
            "shares": [float(c) / len(values) for c in counts],
        }
    return reference


def save(directory: Path, model: Pipeline, metadata: Metadata) -> Path:
    """Write the model and its metadata to ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    skops.io.dump(model, directory / MODEL_FILE)
    (directory / METADATA_FILE).write_text(
        json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8"
    )
    return directory


def load(directory: Path) -> Bundle:
    """Read a bundle written by :func:`save`.

    Raises:
        FileNotFoundError: If the model or its metadata is missing.
    """
    metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    model = skops.io.load(directory / MODEL_FILE, trusted=TRUSTED_TYPES)
    return Bundle(model, Metadata(**metadata))
