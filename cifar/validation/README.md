# CIFAR-100 validation-selected comparison

This archive supports the validation-selected column of the current manuscript Table3. It contains all675 completed ResNet-20 training configurations and160-epoch validation histories:405 runs for nine standalone losses and270 runs for six LS/Mixup/SAM combinations. Each method uses nine configurations and five training seeds. Checkpoints and shared configurations are selected by validation accuracy on the fixed45000/5000 training/validation split; reported test performance uses the official10000-image split.

## Rebuild the comparison without training

From the repository root:

```bash
python cifar/validation/verify_records.py
```

This verifies all675 validation histories and shared choices, recomputes the fifteen five-seed test means/sample standard deviations, and rebuilds both columns of Table3 in `generated/table3.csv`. The best-test column uses the existing `cifar/results/cifar100_besttest.json` archive. For the six regularizer methods,30 compressed files additionally retain the original test labels and argmax predictions; hashes identify their original full score files. Standalone test values retain the original checkpoint-hashed evaluation receipts. Raw images and model weights are not bundled.

## Retrain the frozen configurations

Use the PyTorch2.8.0/torchvision0.23.0/CUDA12.8 environment described in the parent CIFAR guide. Download CIFAR100 from torchvision first. Exact original source files are preserved under `source/`, with provenance and source hashes. Original absolute paths in historical records document their origin. The wrapper rewrites data/output paths in fresh configuration copies:

```bash
python cifar/validation/reproduce.py --data-root ./data --output-root ./runs/validation
python cifar/validation/reproduce.py --data-root ./data --output-root ./runs/validation --methods sam_sln --gpu 0 --execute
```

The first command prints all675 commands without training. The second trains one method's45 runs sequentially on the selected GPU. Distinct methods or seeds can be dispatched to separate GPUs. These original trainers read only training/validation images. Their `best.pth` checkpoint is the earliest maximum-validation checkpoint. The archived `source/*/evaluate*.py` files document the original selection-lock and final-test procedures; their historical scheduler root constants must be adapted to a new output directory. The complete selected configurations, checkpoint hashes and per-seed receipts are retained under `records/`.

The original appendix multi-depth validation tables and common-optimizer validation panel were removed from the article after Table3 was completed. The full original research records remain archived; this public entry concerns the current ResNet-20 comparison.
