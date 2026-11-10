# Calibrated Tabular Lab

Development version 0.1.0. An offline Python package and CLI for inspectable tabular classification experiments with four separate roles: training, probability calibration, threshold selection and final evaluation.

Scikit-learn fits a fixed random forest, a logistic baseline and a prior baseline. A frozen forest is calibrated on held-out data. Original code audits deterministic splits, checks metrics and threshold costs independently, and writes bounded local reports. Calibration may worsen held-out scores; the report preserves both raw and calibrated values.

## Install and run

Qualified platform: Linux amd64, CPython 3.13, with the exact dependencies in the hash locks. No compiler, GPU, service, model download or API is needed at runtime.

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-build-isolation --no-deps .
calibrated-tabular-lab demo --assert --output report-demo
python -m unittest discover -s tests -v
```

The output parent directory must exist. A report destination is never overwritten. Normal CLI commands write one JSON outcome; `--help` prints ordinary usage text. Check exit status and the complete outcome before trusting artifacts.

Presets are `stratified`, `group` and `chronological`. They use the same fixed 320-row, eight-feature synthetic process-deviation dataset. Group mode keeps batches apart; chronological mode keeps timestamp buckets ordered and uses an embargo. These are distinct assurances: chronological data can still share groups.

```sh
calibrated-tabular-lab demo --assert --preset chronological --output report-time
```

## Python API

```python
from calibrated_tabular_lab import (
    ExperimentConfig, create_split_plan, make_synthetic_process_dataset,
    run_experiment,
)

dataset = make_synthetic_process_dataset()
config = ExperimentConfig(seed=42, fp_cost=1, fn_cost=4)
plan = create_split_plan(dataset, seed=42)
result = run_experiment(dataset, config=config, plan=plan)
print(result.test_evaluations["calibrated_forest"].brier_loss)
```

The frozen plan contains canonical row IDs, not a mandatory label hash. Reusing it with altered final-test labels leaves training, calibration, predictions and policy selection unchanged. Initial stratification legitimately uses labels. An explicit config must match the plan's strategy, seed and embargo; stricter assurances are checked.

## Analyze a local input

Create a complete synthetic JSON input from the installed package:

```sh
python - <<'PY'
import json
from pathlib import Path
from calibrated_tabular_lab import make_synthetic_process_dataset
d = make_synthetic_process_dataset()
rows = [dict(row_id=r.row_id, features=list(r.features), label=r.label,
             group_id=r.group_id, timestamp=r.timestamp) for r in d.observations]
Path("synthetic.json").write_text(json.dumps(dict(rows=rows, feature_names=list(d.feature_names))))
PY
calibrated-tabular-lab analyze --input synthetic.json --seed 42 --output report-input
```

`--input -` reads bounded stdin. The closed JSON schema has `rows` and optional `feature_names`; each row has `row_id`, `features`, `label`, and optional `group_id`/`timestamp`. Metadata must be omitted or have its declared type. Duplicate keys, unknown fields, nonfinite numbers, symlink/FIFO inputs and unsupported types are rejected. No expressions, plugins or model files are accepted.

Reports contain strict JSON with aligned replay labels/probabilities, all reliability-bin rows in CSV, every threshold candidate in CSV, escaped self-contained HTML and a per-file SHA256 manifest. There are no external report assets or scripts.

## Bounds and interpretation

Inputs contain 160–4096 rows and 1–16 features, each finite with magnitude at most 1,000,000 or `null` for missing. IDs are unique printable ASCII strings of at most 64 bytes. Every role needs at least eight rows of each label. Input and the aggregate five-file report are each bounded to 4 MiB; callers can lower output admission with `--max-output-bytes`.

The forest has 64 trees of depth at most six; logistic fitting has at most 200 iterations. No parameter search or seed retry occurs. Threshold penalties are hypothetical integer units, not money or a deployment recommendation. Brier and log loss assess more than calibration alone; sparse reliability bins do not certify probabilities. See [the exact contract](docs/contract.md).

CLI outcomes preserve `INVALID_INPUT`, `INVALID_SPLIT`, `FIT_FAILED` and `UNKNOWN_BUDGET`. Filesystem collisions, precommit failures and visible commits with uncertain durability have separate outcomes. A timeout, OOM or dependency problem is operationally incomplete, never a mathematical rejection. No real-time, power-loss or hostile-filesystem guarantee is made.

These are synthetic laboratory examples, without medical, credit, hiring or safety-critical decisions, real customer data, production-readiness claims or accuracy certification.

## Reproduce isolated gates

```sh
sh scripts/ci.sh
```

Only locked official-wheel acquisition has network access. Tests and a new installed-artifact consumer run with no network or host binds, UID/GID 1000, dropped capabilities, no-new-privileges, PID limit 256, 1 GiB and two CPUs. The consumer has no source checkout, build package or compiler. Hosted CI evidence exists only after an actual run.

The gate exports real wheel/source distributions and synthetic sample reports to the ignored `dist/` directory.

Original code is MIT © 2026 [Lofrano Tecnologia](https://lofranotecnologia.com.br). Dependencies retain their own licenses; see [exact notices](THIRD_PARTY_NOTICES.md). [Português brasileiro](README.pt-BR.md).
