# Predictive maintenance MLOps

Predicts machine failures in milling operations on the AI4I 2020 dataset and carries the model
through the whole lifecycle: a pinned dataset, tracked training runs, a model registry with a
promotion rule, an exported serving bundle, an HTTP API in a container and a drift check on
incoming readings. The model is deliberately simple; the point is the path around it.

```text
pmm download   UCI archive, checked against a pinned SHA-256
pmm train      4 candidates as MLflow runs; best by validation PR-AUC is registered
               and gets the alias "champion" only if it is at least as good as the current one
pmm export     champion + threshold + drift reference -> bundle/ (skops, no pickle)
pmm serve      FastAPI on bundle/: /predict, /drift, /metadata, /health
docker build   serving image without MLflow and without training data
```

## Results

Stratified split 70/15/15 with seed 0; the test split has 1,500 rows, 51 of them failures. The
threshold is the one with the best F1 on validation, the test split is scored once, for the
registered model only ([results/training.json](results/training.json)).

| Candidate | Val PR-AUC | Val F1 |
| --- | ---: | ---: |
| Logistic regression | 0.421 | 0.473 |
| Logistic regression + features | 0.394 | 0.436 |
| Gradient boosting | 0.775 | 0.758 |
| **Gradient boosting + features** | **0.859** | **0.832** |

Test, registered model: PR-AUC 0.882, ROC-AUC 0.973, precision 0.949, recall 0.725, F1 0.822 at
threshold 0.781. That is 37 of 51 failures found, with 2 false alarms.

The features are mechanical power (torque times angular speed), the gap between process and air
temperature, and torque times tool wear.

### What these numbers do and do not say

AI4I 2020 is synthetic. Its failures come from fixed rules on exactly the quantities the features
compute (power outside a band, a small temperature gap at low speed, wear times torque above a
limit), plus random failures that no input can predict. The gain from the features therefore shows
that the right physical quantities matter; it does not show how much they would help on a real
machine, where failure mechanisms are unknown, overlap and drift. The five failure-mode columns
(TWF, HDF, PWF, OSF, RNF) are dropped before training, because they define the target.

## Run it

Needs [uv](https://docs.astral.sh/uv/) and, for the container, Docker.

```sh
uv sync
uv run pmm download
uv run pmm train          # MLflow tracking and registry in mlflow.db
uv run pmm export         # writes bundle/
uv run pmm serve          # http://127.0.0.1:8000/docs
uv run pytest             # runs on invented data, no download needed
```

Inspect the runs with `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`. A second
`pmm train` registers a new version and keeps the champion unless the new one is at least as good.

```sh
docker build -t pmm .
docker run --rm -p 8000:8000 pmm
curl -s http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d '{"readings": [{"type": "L", "air_temp_k": 302.0, "process_temp_k": 310.5,
       "rotational_speed_rpm": 1300, "torque_nm": 68.0, "tool_wear_min": 220}]}'
```

```json
{"model_version": "1", "threshold": 0.781,
 "predictions": [{"failure_probability": 0.999906, "failure": true}]}
```

## Drift

The bundle keeps the decile edges of every numeric feature in the training data. `/drift` and
`pmm drift <csv>` compare a batch of at least 100 readings against them with the population
stability index: below 0.1 is ok, from 0.1 a warning, from 0.25 an alert. On the test split every
feature stays below 0.01; adding 8 Nm to every torque reading gives a PSI of 0.59 for torque and an
alert, while the other features stay unchanged.

The check only sees the inputs. Whether the model still predicts well needs labels, i.e. failures
that actually happened, and a scheduled comparison against them; that is not part of this project.

## Design decisions

- **Serving without MLflow.** MLflow is in the `train` dependency group. The image installs only
  the main dependencies and reads the bundle: model, threshold, test metrics and drift reference.
- **skops instead of pickle.** Loading a bundle runs no code beyond an explicit list of trusted
  types (the feature function and the boosting trees).
- **Promotion by rule, not by hand.** A worse retraining is registered but does not move the
  alias, so serving never silently gets worse. A test retrains on noisy data to check that.
- **Strict input validation.** Unknown fields, impossible values and batches over 1,000 readings
  are rejected with 422.

## Layout

```text
src/pmm/       data, model, train (MLflow), bundle, drift, api, cli
tests/         unit tests and an end-to-end test on invented data
Dockerfile     serving image, non-root
.github/       CI: lint, types, tests; then the full pipeline on the real data up to a request
               against the running container
```

## Data and citation

AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, licensed CC BY 4.0
(<https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>). The data is
downloaded at run time and not part of this repository.

S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications", Third
International Conference on Artificial Intelligence for Industries (AI4I), 2020.

## License

MIT, see [LICENSE](LICENSE).
