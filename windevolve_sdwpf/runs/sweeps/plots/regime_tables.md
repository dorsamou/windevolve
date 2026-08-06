### Mean |error| (MW) by power regime and horizon

| regime | n | model | 0-6h | 6-12h | 12-24h | 24-36h | 36-48h |
|---|---|---|---|---|---|---|---|
| parked (<30 kW) | 213 | Seasonal naive | 0.2019 | 0.2870 | 0.7257 | 0.7486 | 0.4404 |
| parked (<30 kW) | 213 | Climatology | 0.3360 | 0.3476 | 0.6675 | 0.6395 | 0.3459 |
| parked (<30 kW) | 213 | ARMA | 0.2847 | 0.2365 | 0.6239 | 0.5856 | 0.3136 |
| parked (<30 kW) | 213 | XGBoost | 0.3838 | 0.2850 | 0.7427 | 0.6729 | 0.3064 |
| parked (<30 kW) | 213 | GRU | 0.2733 | 0.2363 | 0.6379 | 0.6071 | 0.3177 |
| low (30-300) | 50 | Seasonal naive | 0.2200 | 0.3840 | 0.7464 | 0.6748 | 0.3622 |
| low (30-300) | 50 | Climatology | 0.5075 | 0.5028 | 0.7144 | 0.7683 | 0.2929 |
| low (30-300) | 50 | ARMA | 0.4179 | 0.3563 | 0.6606 | 0.6530 | 0.2653 |
| low (30-300) | 50 | XGBoost | 0.4929 | 0.4618 | 0.8174 | 0.7622 | 0.2927 |
| low (30-300) | 50 | GRU | 0.4267 | 0.3684 | 0.6804 | 0.6722 | 0.2691 |
| mid (300-800) | 13 | Seasonal naive | 0.3079 | 0.4248 | 0.7486 | 0.6438 | 0.3524 |
| mid (300-800) | 13 | Climatology | 0.5750 | 0.7097 | 0.6532 | 0.8862 | 0.3983 |
| mid (300-800) | 13 | ARMA | 0.3577 | 0.4564 | 0.6111 | 0.8102 | 0.3168 |
| mid (300-800) | 13 | XGBoost | 0.4099 | 0.4628 | 0.7600 | 0.9520 | 0.3472 |
| mid (300-800) | 13 | GRU | 0.3596 | 0.3989 | 0.6355 | 0.8422 | 0.3246 |
| high (800-1200) | 54 | Seasonal naive | 0.9019 | 0.8299 | 0.6833 | 0.1970 | 0.4996 |
| high (800-1200) | 54 | Climatology | 1.0921 | 1.1431 | 0.3073 | 0.0568 | 0.3797 |
| high (800-1200) | 54 | ARMA | 0.4901 | 0.7858 | 0.3098 | 0.1867 | 0.2900 |
| high (800-1200) | 54 | XGBoost | 0.3097 | 0.5417 | 0.4380 | 0.3130 | 0.2592 |
| high (800-1200) | 54 | GRU | 0.3964 | 0.5417 | 0.3540 | 0.2431 | 0.2806 |
| near rated (>1200) | 72 | Seasonal naive | 1.0640 | 0.9177 | 0.8158 | 0.2390 | 0.5829 |
| near rated (>1200) | 72 | Climatology | 1.2977 | 1.2915 | 0.3328 | 0.0485 | 0.3733 |
| near rated (>1200) | 72 | ARMA | 0.4970 | 0.8670 | 0.3349 | 0.2040 | 0.2839 |
| near rated (>1200) | 72 | XGBoost | 0.3871 | 0.7500 | 0.4214 | 0.2187 | 0.2609 |
| near rated (>1200) | 72 | GRU | 0.5120 | 0.6117 | 0.4070 | 0.2798 | 0.2706 |

### Crossover: first horizon where each model overtakes the best-at-h=1 model

| regime | leader at h=1 | overtaken by | at horizon |
|---|---|---|---|
| parked (<30 kW) | ARMA | GRU | 1.0 h |
| low (30-300) | ARMA | Seasonal naive | 1.2 h |
| mid (300-800) | ARMA | GRU | 1.0 h |
| high (800-1200) | XGBoost | GRU | 4.0 h |
| near rated (>1200) | ARMA | XGBoost | 1.0 h |

### Trajectory bias at short horizon (mean predicted minus mean actual, kW, averaged over 0-6h)

A negative value means the model decays away from its starting state faster than reality does; positive means it holds on too long.

| regime | Seasonal naive | Climatology | ARMA | XGBoost | GRU |
|---|---|---|---|---|---|
| parked (<30 kW) | -33.1 | -285.5 | -267.4 | -375.8 | -256.9 |
| low (30-300) | -72.0 | -425.3 | -396.6 | -479.5 | -411.4 |
| mid (300-800) | -157.3 | -567.8 | -294.0 | -178.4 | -243.3 |
| high (800-1200) | -901.8 | -1092.1 | -485.2 | -272.0 | -380.6 |
| near rated (>1200) | -1064.0 | -1297.7 | -488.4 | -367.9 | -512.0 |
