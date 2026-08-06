# GLUE fine-tuning (BERT / DeBERTa, CE / LN / SLN)

Fine-tune encoder Transformers on GLUE classification tasks with CE / LN / SLN,
following the protocol of Devlin et al. (2019): AdamW, batch 32, linear warmup
6% + linear decay, weight decay 0.01, gradient clipping 1.0.

Only the loss on the classification head's logits is changed; the encoder,
tokenizer and pooling are untouched.

## Run

```bash
# SLN on MNLI (gamma=0.7, tau=0.2)
python train_glue.py --task_name mnli --model_name bert-base-uncased \
    --loss sln --gamma 0.7 --tau 0.2 --lr 2e-5 --epochs 3 \
    --batch_size 32 --seed 0 --output_dir runs/mnli_sln

# CE / LN
python train_glue.py --task_name mnli --model_name bert-base-uncased \
    --loss ce --lr 2e-5 --epochs 3 --batch_size 32 --seed 0 --output_dir runs/mnli_ce
python train_glue.py --task_name mnli --model_name bert-base-uncased \
    --loss ln --tau 0.2 --lr 2e-5 --epochs 3 --batch_size 32 --seed 0 --output_dir runs/mnli_ln
```

Tasks: `cola, sst2, mrpc, qqp, mnli, qnli, rte, stsb, wnli`. Any
`AutoModelForSequenceClassification` checkpoint works (e.g.
`bert-base-uncased`, `bert-large-uncased`, `microsoft/deberta-v3-base`).

The paper reports the best over learning rates {2e-5, 3e-5, 4e-5, 5e-5}, 3
epochs, selected on the dev set.

## Expected results (evaluation sets)

| Model | Loss | Avg (8 tasks) |
|---|---|---|
| BERT-BASE | CE | 82.6 |
| | LN | 82.7 |
| | SLN | **83.8** |
| BERT-LARGE | CE | 84.0 |
| | LN | 84.2 |
| | SLN | **85.0** |

## Files

- `train_glue.py` — single-GPU trainer (CE/LN/SLN, seeded, dev-set selection,
  bf16 autocast). Unlike earlier internal scripts, the random seed and the
  `gamma`/`tau` exponent are proper CLI arguments.
