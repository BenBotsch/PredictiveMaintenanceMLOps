import numpy as np
import pandas as pd
import pytest

from pmm import bundle, train
from pmm.data import Splits, split


def readings(rows: int, seed: int = 0) -> pd.DataFrame:
    """Invented readings in the ranges of AI4I, with failures from high power or worn tools."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "type": rng.choice(["L", "M", "H"], rows),
            "air_temp_k": rng.normal(300, 2, rows),
            "process_temp_k": rng.normal(310, 1.5, rows),
            "rotational_speed_rpm": rng.normal(1540, 180, rows),
            "torque_nm": rng.normal(40, 10, rows),
            "tool_wear_min": rng.uniform(0, 250, rows),
        }
    )
    power = frame["torque_nm"] * frame["rotational_speed_rpm"] * 2 * np.pi / 60
    frame["failure"] = ((power > 8500) | (frame["tool_wear_min"] > 235)).astype(int)
    return frame


@pytest.fixture
def splits() -> Splits:
    return split(readings(1500))


@pytest.fixture(scope="session")
def served(tmp_path_factory: pytest.TempPathFactory) -> bundle.Bundle:
    """One trained and exported bundle, shared by all tests that only read it."""
    root = tmp_path_factory.mktemp("served")
    tracking = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    parts = split(readings(1500))
    # MLflow writes artifacts relative to the working directory.
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        train.run(parts, tracking, log=lambda _: None)
        return bundle.load(train.export(tracking, parts.train, root / "bundle"))
