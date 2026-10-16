# Population-flow validation

This archive supports Section 3.9 and Appendix B of the manuscript. It evolves
logits directly under per-sample Euclidean gradient flow. The finite class clouds
have equal weights and exact stated symmetries; no neural network is trained.
Analytic curves use the initial moments, without fitting rates or rescaling time.

The 240 saved cases comprise 120 symmetric baselines, 36 simplex-deformation
controls, 48 RK4 refinements, and 36 explicit-Euler controls. Class counts are
2, 3 and 10. `manifest.json` fixes the full grid, initial clouds, metric definitions,
source hashes and 101 saved times on [0,40]. Each record in `results/` has the
SHA-256 of its matching NPZ archive. NPZ files contain time series, initial
populations and final transported populations, including the three variance
blocks (the contrast block is absent when K=2).

```bash
python theory/verify_archives.py --recompute-selected
python theory/population_flow/preflight.py
python theory/population_flow/autograd_check.py   # optional; requires PyTorch
python theory/population_flow/plot_figures.py
```

To reproduce the complete campaign in a fresh directory, copy `dynamics.py` and
`run_experiments.py` there and run `python run_experiments.py`. Running it in this
archived directory verifies the source hashes and resumes missing cases. The
`--only baseline|symmetry|refinement|euler` and `--limit N` options select subsets.
The source-hash check deliberately prevents reusing cached results after changes
to either numerical source file. Full simulation uses only NumPy; plotting also
uses Matplotlib. It is deterministic float64 CPU computation. The independent
finite-difference preflight uses seed 20261016.

`summary.json` and `case_summary.csv` summarize the existing records. Width,
geometry and time-window errors specify the scope of the local approximation;
the small-cloud formulas are conditional leading-order predictions. The plotter
writes five PDFs and PNGs under `figures/`, retaining the corrected Euler-step
ticks 0.05, 0.1, 0.2 and 0.4. No image datasets, model checkpoints or training runs
are downloaded or launched by these commands.
