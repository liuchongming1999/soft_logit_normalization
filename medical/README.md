# Biomedical classification experiments

This directory reproduces the medical experiments in **Optimization Dynamics of
Soft Logit Normalization with Applications to Biomedical Image Classification**.
The main comparison uses official MedMNIST v2 PneumoniaMNIST, BloodMNIST and
PathMNIST splits, native 28x28 images and the official ResNet-18. No original
images or trained weights are distributed here. Frozen prediction archives contain
class scores, labels and sample indices, so published tables can be rebuilt offline.
The fixed snapshot also checks every archived file against its SHA256 manifest.

## Rebuild the reported tables without training

Install the scientific dependencies in `requirements.txt`; PyTorch is needed only
for training. The archived environment is in `environment.json` (Python3.13.5,
PyTorch2.8.0+cu128, torchvision0.23.0+cu128, NumPy2.3.4, scikit-learn1.7.1).
For the CUDA build, install PyTorch/torchvision from the official CUDA12.8 wheel
index before installing the remaining dependencies.

```bash
python medical/verify_records.py
python medical/plot_trajectories.py
```

`verify_records.py` checks all150 entries of the original combined medical table,
125 unique frozen test-prediction files,24 paired comparisons and20 independent
PneumoniaMNIST diagnostic runs. It uses the pinned official MedMNIST evaluator
functions and independently computes task metrics. `medical/generated/` contains
Table4 values and LaTeX cells, selected parameters (Table14), paired intervals
(Table15), pneumonia sensitivity/specificity/confusion counts (Table16), macro-F1
and balanced accuracy (Table17), and checkpoint-matched geometry (Table18).
`plot_trajectories.py` recreates Figs12-13 from all101 saved epochs and all five
seeds. Each independent diagnostic checkpoint's train/validation raw logits are
included for direct CDNV recalculation.

## Data and training recipe

```bash
python medical/prepare_data.py --data-root medical/data
python medical/reproduce.py --setting default --dataset pneumoniamnist --method sln
CUDA_VISIBLE_DEVICES=0 python medical/reproduce.py --setting default \
  --dataset pneumoniamnist --method sln --execute
```

Commands are printed by default. `--execute` runs them sequentially on the visible
GPU. `--data-root`, `--output-root`, `--dataset`, `--method`, and `--seeds` control
scope. Adam uses lr0.001, weight decay0, batch128,100epochs and learning-rate
multipliers0.1 after epochs50 and75. Training uses float32, channels-last, cuDNN
TF32 enabled, matrix-multiplication TF32 disabled, and no augmentation. Official
uint8-to-tensor normalization is checked exactly. All methods in a seed share
initialization and the sample permutation generator (`seed+1000003`). The training
runner is a portable adaptation of the exact archived sources in
`historical_sources/`, with configurable paths and an optional trajectory hook.

Checkpoints maximize validation AUC, taking the earliest epoch on ties. Test scores
are evaluated after the checkpoint is locked. AUC uses softmax of the scores used
by the loss; accuracy uses argmax of the same scores. Class-balanced CE weights
are normalized to mean one across classes; reduction is the arithmetic mean of
weighted per-example CE, as in the recorded experiments.

## Default and validation-selected settings

Defaults: CE(0,1), LN(1,0.2), SLN(0.7,0.2), focal exponent2, CB-CE beta0.999.
The original validation search uses seed0, temperatures0.05/0.1/0.2/0.4/1 and
SLN exponents0.2/0.4/0.7/0.9. Focal searches1/2/3 and CB-CE searches
0.99/0.999/0.9999. Every screening run trains100epochs and evaluates validation
only. `records/selected_configs.json` contains the original locked choices.

| Dataset | CE tau | LN tau | SLN (gamma,tau) | Focal exponent | CB-CE beta |
|---|---:|---:|---|---:|---:|
| PneumoniaMNIST |1|0.4|(0.9,1)|1|0.999|
| BloodMNIST |1|0.05|(0.7,1)|3|0.99|
| PathMNIST |0.4|0.05|(0.4,0.4)|2|0.999|

```bash
# Reproduce the original locked choices directly:
python medical/reproduce.py --setting selected --execute
# Or rerun validation selection from scratch, then lock choices before testing:
python medical/reproduce.py --setting screen --execute
python medical/select_configs.py --run-root medical/runs --output medical/new_selected.json
python medical/reproduce.py --setting selected --config-file medical/new_selected.json --execute
```

Five-seed summaries include the screening seed0. Exact repeated configurations
reused prior runs in the reported archive; `table4_cases.json` identifies each
source and prediction file, including this reuse. The command generator starts
fresh runs by default. cuDNN benchmarking/TF32 and hardware can produce run-to-run
variation; archived predictions fix the exact outputs used for the reported tables.

## Additional trajectory and temperature records

```bash
python medical/reproduce.py --setting trajectory --dataset pneumoniamnist --execute
```

This additional batch compares CE(0,1), SLN(0.9,1), LN(1,0.4) and LN(1,0.3) with
five paired seeds. It evaluates raw logits before normalization on the full4708
training and524validation images at initialization and every epoch. The archive
keeps it separate from the original Table4 runs. Late-stage observations refer to
epochs50-100; validation margin variance is approximately stable, whereas both
training directional variances decrease in all five SLN runs.

The later CE/LN temperature extension is retained in `records/temperature_extension*`.
It contains12 additional screens and15 confirmation records and is separate from
the original manuscript table/search budget. All actual original grid records,
loss-specific choices, per-seed configurations, data checksums and geometry are
preserved in `records/`. `manifest.json` records SHA256 hashes.

## Sources

The official architecture comes from MedMNIST/experiments commit
`70b6b3a7ad7afddff1df2a3b735235830fbdb142`; official dataset information and
metric functions come from MedMNIST/MedMNIST commit
`805b74237fb5bd2db376fd94e50646b6a8905d9d`. Their Apache2.0 license texts are
included. The tiny vendored package initializer exposes only INFO and Evaluator.
The SLN loss uses the project implementation, with numerical norm epsilon1e-8.
Data are downloaded from the providers under their own terms; dataset provenance
and label mappings are documented in the manuscript and official MedMNIST metadata.
