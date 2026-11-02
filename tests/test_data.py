import hashlib
import io
import zipfile
from pathlib import Path

import httpx
import pytest

from pmm import data
from pmm.data import TARGET, DatasetError, download, load, split
from tests.conftest import readings

HEADER = (
    "UDI,Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],"
    "Torque [Nm],Tool wear [min],Machine failure,TWF,HDF,PWF,OSF,RNF\n"
)


def archive(csv: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as file:
        file.writestr(data.CSV_NAME, "﻿" + csv)
    return buffer.getvalue()


def hub(content: bytes) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))
    )


def test_download_checks_the_hash_and_extracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = archive(HEADER + "1,M1,M,298.1,308.6,1551,42.8,0,0,0,0,0,0,0\n")
    monkeypatch.setattr(data, "ZIP_SHA256", hashlib.sha256(content).hexdigest())

    path = download(tmp_path, hub(content))

    assert load(path).iloc[0].to_dict() == {
        "type": "M",
        "air_temp_k": 298.1,
        "process_temp_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 0,
        "failure": 0,
    }


def test_download_refuses_a_changed_archive(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="SHA-256"):
        download(tmp_path, hub(archive(HEADER)))
    assert not (tmp_path / data.CSV_NAME).exists()


def test_failure_modes_never_reach_the_features(tmp_path: Path) -> None:
    path = tmp_path / data.CSV_NAME
    path.write_text(HEADER + "1,M1,M,298.1,308.6,1551,42.8,0,1,1,0,0,0,0\n", encoding="utf-8")

    assert not {"TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"} & set(load(path).columns)


def test_missing_columns_are_an_error(tmp_path: Path) -> None:
    path = tmp_path / data.CSV_NAME
    path.write_text("Type,Torque [Nm]\nM,40\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="lacks"):
        load(path)


def test_split_is_stratified_and_disjoint() -> None:
    frame = readings(2000)

    parts = split(frame)

    sizes = [len(parts.train), len(parts.val), len(parts.test)]
    assert sizes == [1400, 300, 300]
    rate = frame[TARGET].mean()
    for part in (parts.train, parts.val, parts.test):
        assert part[TARGET].mean() == pytest.approx(rate, abs=0.01)
