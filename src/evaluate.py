"""Evaluate a trained fusion Transformer on a given test split.

Inputs: test audio/visual .npy files and a saved model checkpoint.
Outputs: prints metrics and optionally appends to a CSV table.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_fusion import (
    AUDIO_DIM,
    LABELS,
    VISUAL_DIM,
    EmbeddingDataset,
    FusionTransformer,
    accuracy,
    build_samples,
    load_embeddings,
    macro_f1,
)


def load_label_mapping(meta_path: Path) -> Dict[str, int]:
    # Prefer label mapping stored alongside the checkpoint; fall back to canonical order
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
            return {k: int(v) for k, v in meta.get("label_to_idx", {}).items()}
    return {lbl: i for i, lbl in enumerate(LABELS)}


def build_loader(
    audio_path: Optional[Path], visual_path: Optional[Path], label_to_idx: Dict[str, int], batch_size: int
) -> DataLoader:
    # Create DataLoader over aligned audio/visual samples for evaluation
    samples = build_samples(load_embeddings(audio_path), load_embeddings(visual_path))
    ds = EmbeddingDataset(samples, label_to_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def evaluate(
    model: FusionTransformer,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> Dict[str, float]:
    # Compute loss, accuracy, and macro-F1 over a dataset
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    with torch.inference_mode():
        for audio, visual, label in loader:
            audio = audio.to(device)
            visual = visual.to(device)
            label = label.to(device)
            logits = model(audio, visual)
            loss = criterion(logits, label)
            total_loss += loss.item() * len(label)
            total_acc += accuracy(logits, label) * len(label)
            total_f1 += macro_f1(logits, label, num_classes=num_classes) * len(label)
    count = len(loader.dataset)
    return {
        "loss": total_loss / max(count, 1),
        "accuracy": total_acc / max(count, 1),
        "f1": total_f1 / max(count, 1),
    }


def maybe_append_table(row: Dict[str, object], table_path: Optional[Path]) -> None:
    # Optionally append a row to a CSV results table
    if table_path is None:
        return
    table_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not table_path.exists()
    with table_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fusion Transformer")
    parser.add_argument("--audio", type=Path, required=False, help="Test audio embeddings .npy")
    parser.add_argument("--visual", type=Path, required=False, help="Test visual embeddings .npy")
    parser.add_argument("--model", type=Path, required=True, help="Path to fusion_transformer.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--table_path", type=Path, required=False, help="Optional CSV output for table II")
    parser.add_argument("--dataset_name", type=str, default="dataset")
    args = parser.parse_args()

    # Load label mapping (supports custom label order per checkpoint)
    label_to_idx = load_label_mapping(args.model.with_suffix(".json"))
    loader = build_loader(args.audio, args.visual, label_to_idx, args.batch_size)

    # Restore fusion model and run evaluation
    model = FusionTransformer().to(args.device)
    state = torch.load(args.model, map_location=args.device)
    model.load_state_dict(state.get("state_dict", state))

    metrics = evaluate(model, loader, args.device, num_classes=len(label_to_idx))
    row = {
        "Dataset": args.dataset_name,
        "Accuracy": round(metrics["accuracy"] * 100, 2),
        "F1_Score": round(metrics["f1"], 4),
    }
    maybe_append_table(row, args.table_path)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
