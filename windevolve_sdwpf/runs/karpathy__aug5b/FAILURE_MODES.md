# Failure modes

Things that failed in a way worth remembering. Newest at the bottom.

## Multi-timescale linear regression does not beat a two-parameter decay

Fit per-lead-time coefficients on trailing means of the history at seven
timescales (last step, 1h, 6h, 24h, 3d, 7d, 14d) plus a farm-wide level and an
intercept, trained on the training split and scored out-of-sample on later
training days. Plain least squares scored 40.06 against 38.78 for the hand-tuned
decay; after the shrink that the metric wants (x0.80) it reached 38.76, a tie.
Optimising the exact competition objective by gradient descent, seeded from the
least-squares solution, did not improve on that either.

Two lessons. First, beyond "recent level" and "14-day level" the extra history
timescales carry essentially no additional information about power 0-48h ahead,
so more features is not the direction. Second, least squares is the wrong loss
here: the objective is 0.5*(RMSE+MAE), and because per-turbine RMSE is around
400 kW the derivative of the RMSE term is roughly e/(2*400), so for typical
errors the MAE term dominates. The optimal forecast sits well below the
conditional mean, which is why a plain shrink factor buys more than any amount
of feature engineering fit by least squares.
