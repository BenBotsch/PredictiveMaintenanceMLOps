"""HTTP API that serves a bundle: predictions, model metadata and an input drift check.

Every request is counted and timed for Prometheus at ``/metrics`` and written as one JSON log
line with a request ID. Logs and metrics carry no readings, only paths, status codes, timings and
counts, so nothing about a machine leaves the service through them.
"""

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Literal

import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from pmm import drift
from pmm.bundle import Bundle, Metadata


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


LOG = logging.getLogger("pmm.api")
REQUEST_ID = "X-Request-ID"
# A client-supplied request ID is echoed only if it is short and plain, so it cannot inject
# anything into the logs.
_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


class _Metrics:
    """Prometheus metrics of one app, in their own registry so that apps do not share them."""

    def __init__(self, meta: Metadata) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "pmm_requests",
            "HTTP requests by route and status code.",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "pmm_request_duration_seconds",
            "Time to answer a request, by route.",
            ["route"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
            registry=self.registry,
        )
        self.readings = Counter(
            "pmm_readings_scored", "Readings scored by /predict.", registry=self.registry
        )
        self.failures = Counter(
            "pmm_failures_predicted",
            "Readings above the threshold, i.e. predicted failures.",
            registry=self.registry,
        )
        self.psi = Gauge(
            "pmm_drift_psi",
            "PSI of each feature in the last /drift batch.",
            ["feature"],
            registry=self.registry,
        )
        Info("pmm_model", "The served model.", registry=self.registry).info(
            {"name": meta.name, "version": meta.version, "run_id": meta.run_id}
        )


def _request_id(request: Request) -> str:
    given = request.headers.get(REQUEST_ID, "")
    if 0 < len(given) <= 64 and set(given) <= _ID_CHARS:
        return given
    return uuid.uuid4().hex


def create_app(served: Bundle) -> FastAPI:
    """The API for one loaded bundle."""
    app = FastAPI(title="Predictive maintenance", version=served.metadata.version)
    meta = served.metadata
    metrics = _Metrics(meta)

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = _request_id(request)
        start = time.perf_counter()
        response = await call_next(request)
        seconds = time.perf_counter() - start
        # The route template, not the raw path, so that unknown paths cannot inflate the labels.
        route = request.scope.get("route")
        template = getattr(route, "path", "unmatched")
        metrics.requests.labels(request.method, template, str(response.status_code)).inc()
        metrics.latency.labels(template).observe(seconds)
        response.headers[REQUEST_ID] = request_id
        LOG.info(
            json.dumps(
                {
                    "event": "request",
                    "request_id": request_id,
                    "method": request.method,
                    "route": template,
                    "status": response.status_code,
                    "duration_ms": round(seconds * 1000, 2),
                }
            )
        )
        return response

    @app.get("/metrics", include_in_schema=False)
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

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
        metrics.readings.inc(len(probability))
        metrics.failures.inc(int((probability >= meta.threshold).sum()))
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
        for f in features:
            metrics.psi.labels(f.feature).set(f.psi)
        return DriftOut(
            alert=any(f.level == "alert" for f in features),
            features=[
                FeatureDriftOut(feature=f.feature, psi=f.psi, level=f.level) for f in features
            ],
        )

    return app
