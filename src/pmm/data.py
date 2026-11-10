"""The AI4I 2020 predictive maintenance dataset: download, schema and splits.

AI4I 2020 (Matzka, 2020) is a synthetic dataset of 10,000 milling operations, published by the UCI
Machine Learning Repository under CC BY 4.0. A machine failure is the union of five failure modes;
the columns of those modes are the target in disguise and never reach a model.
"""

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
ZIP_SHA256 = "f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252"
CSV_NAME = "ai4i2020.csv"

COLUMNS = {
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "failure",
}
TYPES = ("L", "M", "H")
NUMERIC = (
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)
TARGET = "failure"
# Tool wear, heat dissipation, power, overstrain and random failures. Their union is the target,
# so they are dropped with the identifiers UDI and Product ID unless explicitly requested.
MODES = ("TWF", "HDF", "PWF", "OSF", "RNF")


class DatasetError(Exception):
    """The dataset is missing, changed or has an unexpected schema."""


@dataclass(frozen=True, slots=True)
class Splits:
    """Stratified train, validation and test frames."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def download(root: Path, client: httpx.Client | None = None) -> Path:
    """Download the archive, check its SHA-256 and extract the CSV into ``root``.

    Returns:
        The path of the CSV; an existing file is kept.

    Raises:
        DatasetError: If the archive does not match its pinned SHA-256.
    """
    target = root / CSV_NAME
    if target.is_file():
        return target
    own = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=True)
    try:
        response = client.get(URL)
        response.raise_for_status()
    finally:
        if own:
            client.close()
    if hashlib.sha256(response.content).hexdigest() != ZIP_SHA256:
        raise DatasetError(f"{URL} does not match its pinned SHA-256")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        content = archive.read(CSV_NAME)
    root.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def load(path: Path, *, with_modes: bool = False) -> pd.DataFrame:
    """Read the CSV with short column names and without identifiers and failure modes.

    Args:
        path: The CSV written by :func:`download`.
        with_modes: Keep the failure-mode columns (``MODES``) after the target. Only for
            evaluation by mode; they define the target and must never reach a model.

    Raises:
        DatasetError: If the file is missing or a column is missing.
    """
    if not path.is_file():
        raise DatasetError(f"{path} is missing; run `pmm download` first")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {*COLUMNS, *(MODES if with_modes else ())}
    missing = required - set(frame.columns)
    if missing:
        raise DatasetError(f"{path} lacks the columns {sorted(missing)}")
    frame = frame.rename(columns=COLUMNS)
    kept = [*COLUMNS.values(), *(MODES if with_modes else ())]
    # Floats throughout, so the model signature accepts any reading, not just whole numbers.
    return frame[kept].astype(dict.fromkeys(NUMERIC, float))


def split(frame: pd.DataFrame, seed: int = 0) -> Splits:
    """Split 70/15/15, stratified by the target, so each part has about 3.4 % failures."""
    train, rest = train_test_split(frame, test_size=0.3, stratify=frame[TARGET], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=seed)
    return Splits(
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
