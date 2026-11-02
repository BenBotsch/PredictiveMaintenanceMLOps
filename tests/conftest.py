import numpy as np
import pandas as pd
import pytest

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
