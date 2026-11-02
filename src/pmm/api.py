"""HTTP API that serves a bundle: predictions, model metadata and an input drift check."""

from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from pmm import drift
from pmm.bundle import Bundle


class Reading(BaseModel):
    """One machine state. The bounds reject impossible values, not unusual ones."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["L", "M", "H"] = Field(description="Product quality variant: low, medium, high.")
    air_temp_k: float = Field(gt=200, lt=400)
    process_temp_k: float = Field(gt=200, lt=500)
    rotational_speed_rpm: float = Field(gt=0, lt=10_000)
    torque_nm: float = Field(ge=0, lt=500)
    tool_wear_min: float = Field(ge=0, lt=10_000)


class PredictIn(BaseModel):
    """Readings to score."""

    readings: list[Reading] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    """Failure probability and the decision at the model's threshold."""

    failure_probability: float
    failure: bool


class PredictOut(BaseModel):
    """Predictions in the order of the readings."""

    model_version: str
    threshold: float
    predictions: list[Prediction]


class DriftIn(BaseModel):
    """A batch of recent readings. Fewer than 100 make the PSI too noisy to read."""

    readings: list[Reading] = Field(min_length=100, max_length=100_000)


class FeatureDriftOut(BaseModel):
    """PSI of one feature against the training data."""

    feature: str
    psi: float
    level: Literal["ok", "warn", "alert"]


class DriftOut(BaseModel):
    """PSI per feature, worst first, and whether any feature crossed the alert level."""

    alert: bool
    features: list[FeatureDriftOut]


class MetadataOut(BaseModel):
    """The served model."""

    name: str
    version: str
    run_id: str
    threshold: float
    test_metrics: dict[str, float]


def _frame(readings: list[Reading]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in readings])


def create_app(served: Bundle) -> FastAPI:
    """The API for one loaded bundle."""
    app = FastAPI(title="Predictive maintenance", version=served.metadata.version)
    meta = served.metadata

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metadata")
    def metadata() -> MetadataOut:
        return MetadataOut(
            name=meta.name,
            version=meta.version,
            run_id=meta.run_id,
            threshold=meta.threshold,
            test_metrics=meta.test_metrics,
        )

    @app.post("/predict")
    def predict(body: PredictIn) -> PredictOut:
        probability = served.model.predict_proba(_frame(body.readings))[:, 1]
        return PredictOut(
            model_version=meta.version,
            threshold=meta.threshold,
            predictions=[
                Prediction(
                    failure_probability=round(float(p), 6),
                    failure=bool(p >= meta.threshold),
                )
                for p in probability
            ],
        )

    @app.post("/drift")
    def check_drift(body: DriftIn) -> DriftOut:
        features = drift.report(meta.reference, _frame(body.readings))
        return DriftOut(
            alert=any(f.level == "alert" for f in features),
            features=[
                FeatureDriftOut(feature=f.feature, psi=f.psi, level=f.level) for f in features
            ],
        )

    return app
