# Exact binary-logit orbits

Appendix A.9 studies unregularized Euclidean gradient flow directly in binary
logit space, with exact normalization z/(tau ||z||^gamma), fixed gamma in [0,1],
fixed positive tau, and trajectories away from the origin. The conserved quantity
is r^gamma |u_perp|^(1-gamma). The endpoints preserve the common-logit coordinate
(CE) or radius (LN). Temperature changes the time parametrization of each orbit.

`simulate.py` defines 432 cases and independently integrates Cartesian and polar
ODEs, with RK4 refinement, finite differences, and metric/discretization controls.
`summary.json` fixes the grid, float64 seed, source and trajectory hashes, and
errors. `trajectories.npz` contains all 151 saved states on [0,12], using final RK4
step 0.005. `check_second_moment.py` reuses these trajectories to verify the exact
symmetric-population transport identity in 48 groups.

```bash
python theory/verify_archives.py
python theory/binary_orbits/check_second_moment.py
python theory/binary_orbits/plot_orbits.py
```

To rerun every numerical check without overwriting the archive, copy
`simulate.py` and `check_second_moment.py` into a fresh directory, then run them
in that order. Dependencies are NumPy and, for plotting, Matplotlib. The plotter
writes orbit plots to `figures/`. These are CPU calculations and do not train a
network. General feature-space metrics, finite Euler steps, regularization and
a numerical norm offset are separate controls, with their own scope recorded in
the summary. They are not covered by the exact orbit-invariance statement.
