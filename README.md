# VERF

Implementation of **VERF: Versioned Evolving Fuzzy Rules with Delayed-Label Rollback across Manufacturing Tools**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0, pandas 2.2.3. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Generate a controlled rotating-boundary stream with tool offsets and delayed labels:

```bash
verf generate --output data/synthetic.npz --samples 20000 --tools 4 --features 8 --drift-interval 5000 --delay 100 --delay-kind fixed --seed 13
```

Recorded CSV uses `time,tool,label,delay,sensor_1,sensor_2,drift`: increasing integer time, nonnegative delay, zero-based labels, finite features and optional known drift marks. Warmup must contain at least two tools and every class.

```bash
verf prepare --csv data/stream.csv --features sensor_1 sensor_2 --output data/stream.npz
```

## Training and inference

```bash
verf run --data data/synthetic.npz --output outputs/synthetic --warmup 1000 --inject-every 2000 --seed 13
```

## Evaluation

Evaluation is prequential: prediction precedes feedback. Generated audit labels score rollback effects; recorded data can supply independent `--version-truth` annotations.

## Comparisons and ablations

```bash
verf run --data data/synthetic.npz --output outputs/efs --variant no_versions --seed 13
verf run --data data/synthetic.npz --output outputs/no_rollback --variant no_rollback --seed 13
verf run --data data/synthetic.npz --output outputs/global --variant global_rollback --seed 13
verf run --data data/synthetic.npz --output outputs/no_canary --variant no_canary --seed 13
verf run --data data/synthetic.npz --output outputs/no_compensation --variant no_compensation --seed 13
```

## Outputs

`metrics.json`, `history.json`, `versions.json`, `predictions.json` and `probabilities.npz` retain metrics, state size, lineage, issued predictions and probabilities/regret.

```bash
verf plot --runs outputs/synthetic outputs/efs outputs/no_rollback --output outputs/figures
```
