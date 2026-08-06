"""Single-GPU GLUE fine-tuning with CE / LN / SLN loss selection.

Fixes vs. the legacy train_glue_ddp.py:
  * set_seed is actually called (it was commented out);
  * gamma/tau are CLI arguments instead of a hard-coded `norm ** 1.0`;
  * bf16 autocast on RTX 5090, per-epoch dev evaluation, best-dev selection.

Protocol (rebuttal plan E7): AdamW, linear warmup 6% + linear decay,
effective batch 32, weight decay 0.01, grad clip 1.0, max seq len 128.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
import evaluate as hf_evaluate
from transformers import (AutoTokenizer, AutoConfig,
                          AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup,
                          DataCollatorWithPadding, set_seed)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sln import sln_logits  # noqa: E402

NUM_LABELS = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3,
              "qnli": 2, "rte": 2, "stsb": 1, "wnli": 2}
BEST_METRIC = {"cola": "matthews_correlation", "sst2": "accuracy",
               "mrpc": "f1", "qqp": "f1", "mnli": "accuracy",
               "qnli": "accuracy", "rte": "accuracy", "stsb": "pearson",
               "wnli": "accuracy"}


def get_sentence_keys(task):
    if task in ["cola", "sst2"]:
        return "sentence", None
    if task in ["mrpc", "stsb", "wnli", "rte"]:
        return "sentence1", "sentence2"
    if task == "qqp":
        return "question1", "question2"
    if task == "mnli":
        return "premise", "hypothesis"
    if task == "qnli":
        return "question", "sentence"
    raise ValueError(task)


def transform_logits(logits, gamma, tau):
    return sln_logits(logits, gamma, tau)


@torch.no_grad()
def evaluate_model(model, loader, task, metric, device, gamma, tau):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        y = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(**batch).logits
        if task == "stsb":
            p = logits.squeeze(-1).float()
        else:
            p = transform_logits(logits.float(), gamma, tau).argmax(-1)
        preds.append(p.cpu())
        labels.append(y.cpu())
    return metric.compute(predictions=torch.cat(preds).numpy(),
                          references=torch.cat(labels).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--loss", default="ce", choices=["ce", "ln", "sln"])
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accumulation_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gradient_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    device = torch.device("cuda:0")
    task = args.task_name.lower()
    gamma = {"ce": 0.0, "ln": 1.0, "sln": args.gamma}[args.loss]
    tau = 1.0 if args.loss == "ce" else args.tau
    is_regression = task == "stsb"
    metric_key = BEST_METRIC[task]

    raw = load_dataset("glue", task)
    s1, s2 = get_sentence_keys(task)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess(ex):
        if s2 is None:
            return tokenizer(ex[s1], truncation=True, max_length=args.max_length)
        return tokenizer(ex[s1], ex[s2], truncation=True, max_length=args.max_length)

    cols = raw["train"].column_names
    proc = raw.map(preprocess, batched=True,
                   remove_columns=[c for c in cols if c != "label"])
    proc = proc.rename_column("label", "labels")
    proc.set_format(type="torch")

    train_ds = proc["train"]
    if task == "mnli":
        eval_sets = {"matched": proc["validation_matched"],
                     "mismatched": proc["validation_mismatched"]}
    else:
        eval_sets = {"dev": proc["validation"]}

    collator = DataCollatorWithPadding(tokenizer)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
        collate_fn=collator, num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=args.num_workers > 0)
    eval_loaders = {k: torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=False, collate_fn=collator,
        num_workers=2, pin_memory=True) for k, ds in eval_sets.items()}

    steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup_ratio)

    config = AutoConfig.from_pretrained(args.model_name, num_labels=NUM_LABELS[task],
                                        finetuning_task=task)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config).to(device)

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    metric = hf_evaluate.load("glue", task)

    history = []
    best = -1e9
    best_epoch = -1
    best_results = None
    t0 = time.time()
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(**batch).logits
                logits = transform_logits(logits.float(), gamma, tau)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            if (step + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        results = {}
        for name, loader in eval_loaders.items():
            r = evaluate_model(model, loader, task, metric, device, gamma, tau)
            # the primary dev set keeps unprefixed metric keys
            if name in ("dev", "matched"):
                results.update(r)
            else:
                results.update({f"{name}_{k}": v for k, v in r.items()})
        results["epoch"] = epoch + 1
        results["elapsed"] = time.time() - t0
        history.append(results)
        print(f"[eval] epoch {epoch+1}: {results}", flush=True)
        if results[metric_key] > best:
            best = results[metric_key]
            best_epoch = epoch + 1
            best_results = results

    summary = {
        "task": task, "loss": args.loss, "gamma": gamma, "tau": tau,
        "lr": args.lr, "seed": args.seed,
        "best_epoch": best_epoch, "best_metric": metric_key,
        "best_value": best, "best_results": best_results,
        "final_results": history[-1], "history": history,
        "wall_time_sec": time.time() - t0,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] {args.output_dir} best_{metric_key}={best:.4f} @epoch{best_epoch} "
          f"time={summary['wall_time_sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
