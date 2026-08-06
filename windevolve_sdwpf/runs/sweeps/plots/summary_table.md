### 5.4 Cross-model summary

Scores are 0.5(RMSE+MAE) in MW on the validation split, lower is better.

| Model | Reference config | Reference | Best tuned config | Best tuned | Tuning headroom | Best shrinkage config | Shrunk score | Shrinkage gain | Noise floor (1 sd) |
|---|---|---|---|---|---|---|---|---|---|
| ARMA | `order=1,0,1` | 44.976 | `order=1,0,1 fit_days=30` | 44.324 | 0.652 | `order=1,0,1 scale=0.8` | 44.517 | 0.459 | deterministic |
| GRU | `hidden=48 layers=2 (default)` | 45.136 | `40 epochs pat=8` | 44.537 | 0.598 | `default scale=0.8` | 44.642 | 0.493 | deterministic |
| XGBoost | `seed=42` | 46.139 | `trees=200` | 45.445 | 0.694 | `default scale=0.8` | 45.526 | 0.613 | 0.396 |
| Climatology | `n_days=14` | 47.768 | `n_days=10` | 47.047 | 0.721 | `n_days=14 scale=0.8` | 46.940 | 0.828 | deterministic |
| Seasonal naive | `lag=144` | -- | `lag=432` | 59.424 | -- | `--` | -- | -- | deterministic |

**Tuning headroom** = reference minus best tuned. This is the bar the evolutionary search has to clear before *discovery* is the right word for what it did.

**Shrinkage gain** = reference minus the best point on the scale axis. Expect this to be large for the poor forecasters and near zero for the good ones: a well-calibrated forecast has nothing to shrink toward. If that pattern holds it is the cleanest evidence available that shrinkage is metric calibration, not forecasting skill.

Grid points scored: 102. Rejected by the official guards: 17.
