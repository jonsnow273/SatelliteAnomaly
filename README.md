# Satellite Anomaly Detection

Multivariate time-series anomaly detection on real spacecraft telemetry,
built on NASA JPL's SMAP (satellite) and MSL (Mars rover) datasets.

## Problem

Spacecraft stream dozens of telemetry channels (power, radiation, temperature)
back to mission control. Catching abnormal behavior early — before it becomes
a failure — is a hard, imbalanced time-series problem: anomalies are rare,
often subtle (contextual, not just spikes), and every channel behaves
differently.

## Approach

- Baseline: per-channel statistical anomaly detection (rolling z-score / Isolation Forest)
- Core model: LSTM trained to predict the next telemetry value from recent
  history; large prediction error = anomaly signal
- Dynamic, nonparametric thresholding per channel (rather than a fixed cutoff)
- Evaluated against expert-labeled anomaly windows

## Benchmark

Reference implementation: Hundman et al. (2018), "Detecting Spacecraft
Anomalies Using LSTMs and Nonparametric Dynamic Thresholding" (KDD 2018).
Reproduced their results locally as a baseline to compare against:

| Metric    | Paper (reported) | Reproduced locally |
|-----------|-------------------|---------------------|
| Precision | 87.5%             | 87%                 |
| Recall    | 80.0%             | 83%                 |

## Dataset

- NASA SMAP + MSL telemetry (Kaggle mirror: patrickfleith/nasa-anomaly-detection-dataset-smap-msl)
- 82 channels, 105 labeled anomaly sequences, ~496K telemetry values

## Stack

Python, PyTorch/TensorFlow, scikit-learn, pandas/numpy, FastAPI

## Project structure

See `src/` for the pipeline, `api/` for the serving layer.

---

made by --- pranit bharat more 🐐