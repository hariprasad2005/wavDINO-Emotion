"""Train an audio-only emotion classifier from precomputed Wav2Vec2 embeddings.

Inputs: train/val .npy files produced by extract_audio.py (list of dicts with keys path,label,embedding).
Outputs: best checkpoint (.pt) + meta (.json) + log (.txt). Metrics: accuracy and macro-F1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_fusion import AUDIO_DIM, LABELS, accuracy, load_embeddings, macro_f1


class AudioDataset(Dataset):
    def __init__(self, items: Sequence[Dict[str, object]], label_to_idx: Dict[str, int]):
        self.items = list(items)
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        emb = torch.tensor(item["embedding"], dtype=torch.float32)
        if emb.ndim > 1:
            # Average over time if the embedding contains a sequence
            emb = emb.mean(dim=0)
        label = torch.tensor(self.label_to_idx[item["label"]], dtype=torch.long)
        return emb, label


def build_items(path: Path) -> List[Dict[str, object]]:
    # Load list-of-dicts from .npy produced by extract_audio.py
    return list(load_embeddings(path) or [])


class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_classes: int):
        super().__init__()
        # Simple MLP head for classification over embeddings
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def eval_epoch(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> Dict[str, float]:
    # Evaluate one epoch on a loader, returning loss/acc/F1
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    with torch.inference_mode():
        for emb, label in loader:
            emb = emb.to(device)
            label = label.to(device)
            logits = model(emb)
            loss = criterion(logits, label)
            total_loss += loss.item() * len(label)
            total_acc += accuracy(logits, label) * len(label)
            total_f1 += macro_f1(logits, label, num_classes=num_classes) * len(label)
    count = len(loader.dataset)
    return {
        "loss": total_loss / max(count, 1),
        "acc": total_acc / max(count, 1),
        "f1": total_f1 / max(count, 1),
    }


def train(train_loader: DataLoader, val_loader: DataLoader, device: str, epochs: int, lr: float, model_args: Dict[str, object], log_path: Path, model_path: Path) -> Dict[str, float]:
    # Train loop with best-checkpoint tracking by validation accuracy
    model = EmbeddingClassifier(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1.0
    best_state: Dict[str, object] = {}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("epoch,train_loss,val_loss,val_acc,val_f1\n")
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for emb, label in train_loader:
                emb = emb.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                logits = model(emb)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(label)

            train_loss = running_loss / max(len(train_loader.dataset), 1)
            val_metrics = eval_epoch(model, val_loader, device, num_classes=model_args["num_classes"])
            log.write(f"{epoch},{train_loss:.4f},{val_metrics['loss']:.4f},{val_metrics['acc']:.4f},{val_metrics['f1']:.4f}\n")
            log.flush()

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_state = {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_metrics["acc"],
                    "val_f1": val_metrics["f1"],
                }
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, model_path)

    return {"best_val_acc": best_state.get("val_acc", 0.0), "best_val_f1": best_state.get("val_f1", 0.0)}


def build_loader(path: Path, label_to_idx: Dict[str, int], batch_size: int, shuffle: bool) -> DataLoader:
    # Utility to create a DataLoader from embeddings with consistent label mapping
    ds = AudioDataset(build_items(path), label_to_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def derive_labels(paths: Sequence[Path]) -> List[str]:
    # Infer label set from provided embedding files while preserving canonical order
    labels = set()
    for p in paths:
        for item in load_embeddings(p) or []:
            labels.add(item["label"])
    ordered = [lbl for lbl in LABELS if lbl in labels]
    return ordered or LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Train audio-only emotion classifier")
    parser.add_argument("--train", type=Path, required=True, help="Train embeddings .npy")
    parser.add_argument("--val", type=Path, required=True, help="Validation embeddings .npy")
    parser.add_argument("--test", type=Path, required=False, help="Optional test embeddings .npy")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_path", type=Path, default=Path("./logs/audio_training_log.txt"))
    parser.add_argument("--model_path", type=Path, default=Path("./models/audio_classifier.pt"))
    args = parser.parse_args()

    label_list = derive_labels([args.train, args.val] + ([args.test] if args.test else []))
    label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}

    model_args = {
        "input_dim": AUDIO_DIM,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "num_classes": len(label_to_idx),
    }

    train_loader = build_loader(args.train, label_to_idx, args.batch_size, shuffle=True)
    val_loader = build_loader(args.val, label_to_idx, args.batch_size, shuffle=False)

    best = train(train_loader, val_loader, args.device, args.epochs, args.lr, model_args, args.log_path, args.model_path)

    test_metrics: Dict[str, float] = {}
    if args.test is not None:
        test_loader = build_loader(args.test, label_to_idx, args.batch_size, shuffle=False)
        model = EmbeddingClassifier(**model_args).to(args.device)
        state = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(state.get("state_dict", state))
        test_metrics = eval_epoch(model, test_loader, args.device, num_classes=len(label_to_idx))
        print(json.dumps({"test": test_metrics}, indent=2))

    meta = {
        "label_to_idx": label_to_idx,
        "best_val_acc": best["best_val_acc"],
        "best_val_f1": best["best_val_f1"],
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "input_dim": AUDIO_DIM,
    }
    meta_path = args.model_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Training complete. Best val acc={best['best_val_acc']:.4f}, val f1={best['best_val_f1']:.4f}")
    if test_metrics:
        print(f"Test acc={test_metrics['acc']:.4f}, f1={test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
