
### ARMA

| axis | setting | score (MW) | RMSE | MAE | valid |
|---|---|---|---|---|---|
| order | order=1,0,1 | 44.976 | 48.853 | 41.099 | yes |
| order | order=1,0,0 | 45.063 | 48.945 | 41.181 | yes |
| order | order=2,0,1 | 45.537 | 49.498 | 41.575 | yes |
| order | order=2,0,2 | 44.883 | 48.751 | 41.015 | yes |
| order | order=1,1,0 | -- | -- | -- | MetricsError: 92.54% turbines with (almost) identical predic |
| order | order=1,1,1 | 56.631 | 61.433 | 51.829 | yes |
| fit_days | order=1,0,1 fit_days=30 | 44.324 | 48.705 | 39.943 | yes |
| fit_days | order=1,0,1 fit_days=90 | 44.681 | 48.670 | 40.691 | yes |
| fit_days | order=1,0,1 fit_days=197 | 44.976 | 48.853 | 41.099 | yes |
| apply_steps | order=1,0,1 apply=144 | 44.976 | 48.853 | 41.099 | yes |
| apply_steps | order=1,0,1 apply=288 | 44.976 | 48.853 | 41.099 | yes |
| apply_steps | order=1,0,1 apply=432 | 44.976 | 48.853 | 41.099 | yes |
| apply_steps | order=1,0,1 apply=1008 | 44.976 | 48.853 | 41.099 | yes |
| apply_steps | order=1,0,1 apply=2016 | 44.976 | 48.853 | 41.099 | yes |
| scale | order=1,0,1 scale=1.0 | 44.976 | 48.853 | 41.099 | yes |
| scale | order=1,0,1 scale=0.9 | 44.561 | 48.712 | 40.410 | yes |
| scale | order=1,0,1 scale=0.8 | 44.517 | 48.980 | 40.053 | yes |
| scale | order=1,0,1 scale=0.7 | 44.880 | 49.694 | 40.066 | yes |
| scale | order=1,0,1 scale=0.6 | 45.712 | 50.889 | 40.536 | yes |
| scale | order=1,0,1 scale=0.5 | 47.077 | 52.598 | 41.555 | yes |
| scale | order=1,0,1 scale=0.4 | 49.028 | 54.861 | 43.196 | yes |
| scale | order=1,0,1 scale=0.3 | 51.612 | 57.698 | 45.525 | yes |
| scale | order=1,0,1 scale=0.2 | 54.813 | 61.048 | 48.579 | yes |

### Climatology

| axis | setting | score (MW) | RMSE | MAE | valid |
|---|---|---|---|---|---|
| n_days | n_days=14 | 47.768 | 52.665 | 42.871 | yes |
| n_days | n_days=1 | -- | -- | -- | MetricsError: 66.42% turbines with zero predicted values (in |
| n_days | n_days=2 | 62.538 | 68.025 | 57.051 | yes |
| n_days | n_days=3 | 54.079 | 59.645 | 48.512 | yes |
| n_days | n_days=5 | 50.553 | 55.855 | 45.251 | yes |
| n_days | n_days=7 | 49.148 | 54.444 | 43.853 | yes |
| n_days | n_days=10 | 47.047 | 52.064 | 42.031 | yes |
| halflife | halflife=0.5 | 64.147 | 70.052 | 58.242 | yes |
| halflife | halflife=1 | 56.687 | 62.191 | 51.182 | yes |
| halflife | halflife=2 | 51.134 | 56.444 | 45.823 | yes |
| halflife | halflife=3 | 49.345 | 54.549 | 44.141 | yes |
| halflife | halflife=5 | 48.248 | 53.327 | 43.169 | yes |
| halflife | halflife=10 | 47.790 | 52.768 | 42.813 | yes |
| agg | median n_days=3 | 57.218 | 63.624 | 50.813 | yes |
| agg | median n_days=7 | 52.092 | 57.932 | 46.252 | yes |
| agg | median n_days=14 | 50.455 | 56.177 | 44.732 | yes |
| scale | n_days=14 scale=1.0 | 47.768 | 52.665 | 42.871 | yes |
| scale | n_days=14 scale=0.9 | 47.167 | 52.221 | 42.114 | yes |
| scale | n_days=14 scale=0.8 | 46.940 | 52.163 | 41.717 | yes |
| scale | n_days=14 scale=0.7 | 47.130 | 52.533 | 41.726 | yes |
| scale | n_days=14 scale=0.6 | 47.787 | 53.380 | 42.193 | yes |
| scale | n_days=14 scale=0.5 | 48.955 | 54.748 | 43.161 | yes |
| scale | n_days=14 scale=0.4 | 50.663 | 56.662 | 44.664 | yes |
| scale | n_days=14 scale=0.3 | 52.920 | 59.101 | 46.740 | yes |
| scale | n_days=14 scale=0.2 | 55.708 | 61.995 | 49.422 | yes |

### XGBoost

| axis | setting | score (MW) | RMSE | MAE | valid |
|---|---|---|---|---|---|
| seed | seed=42 | 46.139 | 50.448 | 41.830 | yes |
| seed | seed=43 | 46.922 | 51.203 | 42.641 | yes |
| seed | seed=44 | 46.426 | 50.645 | 42.207 | yes |
| features | features=power | 45.834 | 50.155 | 41.512 | yes |
| features | features=base | 46.139 | 50.448 | 41.830 | yes |
| features | features=turbine | 48.151 | 52.550 | 43.752 | yes |
| features | features=farm | 47.582 | 51.957 | 43.207 | yes |
| features | features=neighbour | 47.789 | 52.103 | 43.474 | yes |
| features | features=xy | 47.609 | 51.916 | 43.303 | yes |
| trees | trees=200 | 45.445 | 49.692 | 41.198 | yes |
| trees | trees=400 | 46.139 | 50.448 | 41.830 | yes |
| trees | trees=800 | 46.683 | 51.039 | 42.327 | yes |
| depth | depth=4 | 45.512 | 49.724 | 41.300 | yes |
| depth | depth=6 | 46.139 | 50.448 | 41.830 | yes |
| depth | depth=9 | 47.179 | 51.645 | 42.714 | yes |
| lr | lr=0.03 | 45.759 | 49.966 | 41.552 | yes |
| lr | lr=0.05 | 46.139 | 50.448 | 41.830 | yes |
| lr | lr=0.1 | 47.231 | 51.603 | 42.858 | yes |
| objective | objective=squarederror | 46.139 | 50.448 | 41.830 | yes |
| objective | objective=absoluteerror | 49.104 | 54.243 | 43.965 | yes |
| scale | default scale=1.0 | 46.139 | 50.448 | 41.830 | yes |
| scale | default scale=0.9 | 45.650 | 50.105 | 41.194 | yes |
| scale | default scale=0.8 | 45.526 | 50.156 | 40.897 | yes |
| scale | default scale=0.7 | 45.819 | 50.658 | 40.980 | yes |
| scale | default scale=0.6 | 46.583 | 51.681 | 41.486 | yes |
| scale | default scale=0.5 | 47.876 | 53.281 | 42.470 | yes |
| scale | default scale=0.4 | 49.749 | 55.483 | 44.015 | yes |
| scale | default scale=0.3 | 52.214 | 58.242 | 46.185 | yes |
| scale | default scale=0.2 | 55.243 | 61.447 | 49.040 | yes |

### GRU

| axis | setting | score (MW) | RMSE | MAE | valid |
|---|---|---|---|---|---|
| capacity | hidden=48 layers=2 (default) | 45.136 | 49.038 | 41.233 | yes |
| capacity | hidden=96 | 44.833 | 48.763 | 40.902 | yes |
| capacity | hidden=128 | 44.745 | 48.683 | 40.806 | yes |
| capacity | layers=1 | 45.118 | 48.999 | 41.238 | yes |
| capacity | hidden=96 layers=1 | 44.927 | 48.853 | 41.000 | yes |
| input_len | input_len=288 | 45.117 | 49.042 | 41.192 | yes |
| input_len | input_len=432 | 45.093 | 49.021 | 41.165 | yes |
| optim | lr=3e-4 | 44.642 | 48.576 | 40.708 | yes |
| optim | lr=1e-3 | 44.667 | 48.666 | 40.667 | yes |
| optim | lr=1e-4 flat | 44.604 | 48.558 | 40.649 | yes |
| optim | lr=1e-3 decay=0.9 | 44.778 | 48.807 | 40.749 | yes |
| budget | 20 epochs pat=5 | 44.597 | 48.552 | 40.642 | yes |
| budget | 40 epochs pat=8 | 44.537 | 48.499 | 40.576 | yes |
| provenance | official loop (defects) | 44.802 | 48.818 | 40.786 | yes |
| scale | default scale=1.0 | 45.136 | 49.038 | 41.233 | yes |
| scale | default scale=0.9 | 44.709 | 48.878 | 40.541 | yes |
| scale | default scale=0.8 | 44.642 | 49.122 | 40.163 | yes |
| scale | default scale=0.7 | 44.991 | 49.816 | 40.166 | yes |
| scale | default scale=0.6 | 45.825 | 51.008 | 40.642 | yes |
| scale | default scale=0.5 | 47.210 | 52.741 | 41.679 | yes |
| scale | default scale=0.4 | 49.190 | 55.042 | 43.338 | yes |
| scale | default scale=0.3 | 51.781 | 57.897 | 45.664 | yes |
| scale | default scale=0.2 | 54.951 | 61.212 | 48.689 | yes |

### Seasonal naive

| axis | setting | score (MW) | RMSE | MAE | valid |
|---|---|---|---|---|---|
| lag | lag=144 | -- | -- | -- | MetricsError: 66.42% turbines with zero predicted values (in |
| lag | lag=2 | -- | -- | -- | MetricsError: 76.87% turbines with zero predicted values (in |
| lag | lag=6 | -- | -- | -- | MetricsError: 54.48% turbines with zero predicted values (in |
| lag | lag=18 | -- | -- | -- | MetricsError: 100.00% turbines with (almost) identical predi |
| lag | lag=36 | -- | -- | -- | MetricsError: 19.40% turbines with zero predicted values (in |
| lag | lag=72 | -- | -- | -- | MetricsError: 19.40% turbines with zero predicted values (in |
| lag | lag=288 | 68.653 | 75.380 | 61.925 | yes |
| lag | lag=432 | 59.424 | 66.321 | 52.527 | yes |
| lag | lag=1008 | 63.115 | 70.587 | 55.642 | yes |
| lag | lag=2016 | 62.760 | 68.957 | 56.564 | yes |
| scale | lag=144 scale=1.0 | -- | -- | -- | MetricsError: 66.42% turbines with zero predicted values (in |
| scale | lag=144 scale=0.9 | -- | -- | -- | MetricsError: 66.42% turbines with zero predicted values (in |
| scale | lag=144 scale=0.8 | -- | -- | -- | MetricsError: 66.42% turbines with zero predicted values (in |
| scale | lag=144 scale=0.7 | -- | -- | -- | MetricsError: 66.42% turbines with zero predicted values (in |
| scale | lag=144 scale=0.6 | -- | -- | -- | MetricsError: 66.42% turbines with zero predicted values (in |
| scale | lag=144 scale=0.5 | -- | -- | -- | MetricsError: 66.42% turbines with zero predicted values (in |
| scale | lag=144 scale=0.4 | -- | -- | -- | MetricsError: 66.42% turbines with zero predicted values (in |
| scale | lag=144 scale=0.3 | -- | -- | -- | MetricsError: 66.42% turbines with zero predicted values (in |
| scale | lag=144 scale=0.2 | -- | -- | -- | MetricsError: 66.42% turbines with zero predicted values (in |
