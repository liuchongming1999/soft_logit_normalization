# Current CIFAR-100 results

`cifar100_besttest.json` contains the 216 completed configuration rows and
24 selected method/depth rows used by the current manuscript's Tables 2,
6 and 7. Each configuration has seeds 0-4. Accuracy is stored as a fraction;
`sd` is the sample standard deviation of the five seed-wise best-test values.

The experiments completed on 2026-10-07. The audited training histories,
effective arguments and checkpoint selection were reconciled before this
export. Data paths, server paths and resource-control metadata are omitted
from the public results; scientific settings and seed-wise values are retained.

The complete configuration indices match `../protocol.py`. Regenerate the
training outputs using `../reproduce.py`, then aggregate with `../summarize.py`.
The fixed-optimizer panel uses configuration index 2 for CE/LN/SLN and
index 3 for the four normalized-head objectives.
