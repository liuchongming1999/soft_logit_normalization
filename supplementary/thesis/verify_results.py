#!/usr/bin/env python3
"""Recompute the supplementary ImageNet and GLUE tables from saved metrics."""
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def percentage(value):
    return Decimal(str(value)) * 100


def display(value):
    return str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def imagenet_table():
    data = json.loads((ROOT / "imagenet.json").read_text())
    lines = [
        "| Model | Loss | Temperature | Selected epoch | Top-1 (%) | Top-5 (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run in data["runs"]:
        with (ROOT / run["history"]).open(newline="") as file:
            history = list(csv.DictReader(file))
        best = max(history, key=lambda row: Decimal(row["eval_top1"]))
        lines.append(
            f"| {run['model']} | {run['loss']} | {run['temperature']} "
            f"| {int(best['epoch']) + 1} | {display(best['eval_top1'])} "
            f"| {display(best['eval_top5'])} |"
        )
    return "\n".join(lines)


def glue_table():
    data = json.loads((ROOT / "glue.json").read_text())
    lines = [
        "| Model | Loss | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | MRPC | RTE | Mean |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in data["runs"]:
        tasks = run["tasks"]
        metrics = [
            percentage(tasks["MNLI"]["accuracy"]),
            percentage(tasks["MNLI"]["mismatched_accuracy"]),
            percentage(tasks["QQP"]["f1"]),
            percentage(tasks["QNLI"]["accuracy"]),
            percentage(tasks["SST-2"]["accuracy"]),
            percentage(tasks["CoLA"]["matthews_correlation"]),
            percentage(tasks["MRPC"]["f1"]),
            percentage(tasks["RTE"]["accuracy"]),
        ]
        scores = [display(metric) for metric in metrics]
        mean = display(sum(metrics) / Decimal(len(metrics)))
        lines.append(
            f"| {run['model']} | {run['loss']} | {scores[0]}/{scores[1]} "
            + "".join(f"| {score} " for score in scores[2:])
            + f"| {mean} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print("ImageNet-1K")
    print(imagenet_table())
    print()
    print("GLUE")
    print(glue_table())
