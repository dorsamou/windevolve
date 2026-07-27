# windevolve_sdwpf

Evolutionary model discovery for wind power forecasting on the **SDWPF / Baidu KDD Cup 2022** dataset.

This folder contains the evaluation harness, a tiered set of baselines, and the OpenEvolve configuration used to search for forecasting programs. Every method here is scored with the **official competition metric**.

---

## Layout

```
windevolve_sdwpf/
├── metrics.py                  # official scorer  (do not modify)
├── test_data.py                # official window loader  (do not modify)
├── evaluator.py                # OpenEvolve-facing evaluator
├── initial_program.py          # persistence seed, evolved by OpenEvolve
├── run_baselines.py            # score every baseline into one table
├── config.yaml                 # OpenEvolve configuration
└── baseline_models/
    ├── naive/              # no training required
    │   ├── baseline_naive.py
    └── gru/
        ├── gru_baseline.py
        └── checkpoints/        # model_0.pt ... model_133.pt
```

Each model has a `.md` beside its `.py`. Trained artefacts live next to the model
that produces them, so `gbdt_model.joblib` and `checkpoints/` resolve relative to
their own file and need no configuration.

## Contents

### Evaluation harness

| File | Purpose |
|---|---|
| `metrics.py` | opy of the official scorer: masking rules, kW→MW, per-turbine MAE/RMSE, sum across turbines, degeneracy guards. Verified bit-for-bit against the original. |
| `test_data.py` | Faithful copy of the official per-window loader. Slices one window CSV into 134 per-turbine blocks.|
| `evaluator.py` | The OpenEvolve-facing evaluator. Loops the test windows, calls a program's `forecast(envs)`, scores with `metrics.py`, and returns a metrics dict. |

### Forecasting programs

Every program exposes the same contract, so all are scored identically:

```python
forecast(envs) -> np.ndarray   # shape (134, 288, 1), active power in kW
```

`envs["path_to_test_x"]` is the path to one test window; the program reads it itself. `envs` also carries `path_to_train` (the 245-day CSV) and `path_to_locations` (turbine x/y) for methods that fit a model.


### Documentation

Each model has a `.md` beside its `.py` explaining the method, its parameters, and its caveats. `baseline_models/gru/gru_baseline.md` covers the GRU in depth.

---

## Setup

The data is **not** in this repo. Point the harness at it with environment variables (or edit the constants at the top of `evaluator.py`):

```bash
export SDWPF_TEST_DIR="/path/to/sdwpf_kddcup/final_phase_test"
export SDWPF_TRAIN_FILE="/path/to/sdwpf_kddcup/sdwpf_245days_v1.csv"
export SDWPF_LOCATION_FILE="/path/to/sdwpf_kddcup/sdwpf_baidukddcup2022_turb_location.csv"
```

`final_phase_test/` must contain `infile/*in.csv` and `outfile/*out.csv` (142 windows).

Dependencies: `numpy`, `pandas`, `scikit-learn`, `joblib`, `torch` (GRU only), `matplotlib` (notebooks).

---

## Usage

All commands are run from the package root (`windevolve_sdwpf/`).

**Score one program:**

```bash
python evaluator.py initial_program.py                              # all windows
python evaluator.py initial_program.py "" 10                        # first 10 windows
python evaluator.py baseline_models/classical/baseline_expsmooth.py
```

**Train the models that need it:**

```bash
python baseline_models/tree/baseline_gbdt.py --origins 150
python baseline_models/gru/gru_baseline.py --turbines 0-133 --device cuda
```

Both are resumable: the GRU skips turbines whose checkpoints already exist.


**Run the evolutionary search:**

```bash
export OPENAI_API_KEY="<key>"
python -m openevolve.cli initial_program.py evaluator.py --config config.yaml
```

---

## The metric

For each turbine, ground-truth steps that are missing or invalid are **excluded** (not zeroed):

- any NaN in the row
- `Patv < 0`
- `Patv == 0` while `Wspd > 2.5` — a stopped turbine
- any pitch angle `Pab* > 89` — blades feathered
- `|Wdir| > 180` or `|Ndir| > 720` — sensor out of range

On the surviving steps, MAE and RMSE are computed in **megawatts**, then **summed across the 134 turbines** to score one window, then **averaged across the 142 windows**:

```
score = 0.5 * (RMSE + MAE)      # lower is better
```

Because errors are summed over turbines, a method must work well on **all** turbines — a few bad ones dominate the total.

**Sign convention:** `evaluator.py` returns `combined_score = -competition_score` because OpenEvolve maximizes. A valid score is therefore **negative** (roughly −40 to −70). Failures return `−1e6`, always by returning rather than raising, since a raised exception would be replaced by fitness `0.0` and wrongly outrank real programs.

**Degeneracy guards:** the official scorer rejects predictions that are all-zero or (near-)constant across the farm. A flat per-turbine mean is rejected outright — which is why `baseline_climatology.py` varies by time of day rather than emitting a single number.

---

## Citation

Scoring and program contract follow the official Baidu KDD Cup 2022 SDWPF evaluator:
PaddlePaddle/PaddleSpatial, `apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation`
<https://github.com/PaddlePaddle/PaddleSpatial>

Dataset:

- Zhou et al. "SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting Challenge at KDD Cup 2022." arXiv:2208.04360, 2022.
- Zhou et al. "A Dataset for Spatial Dynamic Wind Power Forecasting over a Large Turbine Array." *Scientific Data* 11:649, 2024. doi:10.1038/s41597-024-03427-5
