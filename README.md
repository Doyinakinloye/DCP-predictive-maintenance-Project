# Predictive Maintenance on NASA C-MAPSS (FD003)

A predictive-maintenance proof of concept that pairs **Remaining Useful Life (RUL) regression** with **unsupervised anomaly detection**, then fuses both into a single composite risk score — built on NASA's C-MAPSS turbofan degradation dataset (FD003 subset), with sensor channels relabeled to cement-plant equipment analogues (kiln, mill, crusher, separator) as a conceptual proxy for real DCP plant sensors.

> **Note on the dataset:** C-MAPSS is simulated turbofan engine degradation data, not real cement-plant telemetry. The sensor renaming (e.g. `sensor_3` → `Kiln_Burn_Zone_Temp`) is an interpretive mapping used to reason about the approach in a cement-plant context — it demonstrates the *methodology*, not validated cement-equipment behavior.

## Approach

**1. Data preparation**
- Dropped near-constant sensor columns (fewer than 18 unique values) to remove non-informative noise.
- Computed ground-truth RUL for training data as `max_cycle − time_cycle` per engine unit; merged the provided `RUL_FD003` file into the test set the same way.
- Engineered rolling-window features (mean, std, deviation-from-baseline) with a window of 10 cycles per sensor, back-filling the resulting NaNs at the start of each unit's series.
- Checked for outliers via per-unit z-scores (`|z| > 3`); found the outlier rate negligible relative to sample size and proceeded without removal.
- Scaled all features with `RobustScaler` (chosen over standard scaling for resilience to the outliers found above).
- Split by **engine unit**, not by row, before train/validation partitioning — this avoids leaking a single engine's degradation trajectory across both splits.

**2. RUL regression — LSTM**
- Sliding-window sequences of `SEQUENCE_LENGTH = 50` cycles per engine, using operational settings + all sensor/rolling features (54 total) as input.
- Architecture: `LSTM(64) → Dropout(0.3) → Dense(32, relu) → Dropout(0.3) → Dense(1)`.
- Training data was capped at `RUL > 160` so the model concentrates learning capacity on the degradation region rather than the long, near-constant healthy-life plateau.
- Trained 10 epochs with early stopping on validation loss.

**3. Anomaly detection — One-Class Conv1D Autoencoder**
- Same 50-cycle sequence windows, but trained only to reconstruct its own input (no RUL label involved).
- Architecture: `Conv1D(32) → MaxPool → Conv1D(64) → MaxPool → Dense(32) bottleneck → Dense → Reshape → Conv1DTranspose(64) → Conv1DTranspose(32) → Conv1D(output)`, with a linear final activation to match the scaled (non-[0,1]) feature range.
- Anomaly threshold set at the 95th percentile of reconstruction error (MSE), giving a data-driven cutoff rather than a fixed guess.

**4. Composite risk scoring**
- Combines the two model outputs: `risk = 0.6 × (1 − normalized RUL) + 0.4 × normalized reconstruction error`.
- Risk is bucketed into three tiers — **Healthy** (< 0.3), **Warning** (0.3–0.7), **Critical** (> 0.7) — turning two separate numeric outputs into a single actionable status per asset.

## Results

| Model | Metric | Value |
|---|---|---|
| LSTM (RUL regression) | RMSE | 17.80 cycles |
| LSTM (RUL regression) | MAE | 12.70 cycles |
| LSTM (RUL regression) | R² | 0.81 |
| LSTM (RUL regression) | NASA asymmetric scoring function | ~956 |
| CNN autoencoder | Reconstruction-error threshold (95th pct) | 0.90 |

## Tech stack

`Python` · `pandas` / `numpy` · `scikit-learn` (RobustScaler, train_test_split) · `TensorFlow / Keras` (LSTM, Conv1D autoencoder) · `matplotlib` / `seaborn`

## Dataset

[NASA C-MAPSS Turbofan Engine Degradation Simulation](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) — FD003 subset (`train_FD003.csv`, `test_FD003.csv`, `RUL_FD003.csv`).
