"""Cross-dataset evaluation for TABLE I.

Takes a manifest JSON describing embedding files for each dataset and runs:
- Train on each source dataset's train/val
- Evaluate on every target dataset's test split
- Compute drop = target_acc - source_self_acc
Outputs CSV table_I_cross_dataset.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_fusion import FusionTransformer, accuracy, build_samples, load_embeddings, macro_f1


def build_loader(
    emb_audio: Optional[Path],
    emb_visual: Optional[Path],
    label_to_idx: Dict[str, int],
    batch_size: int,
    shuffle: bool,
    balance: bool = False,
) -> DataLoader:
    # Align audio and visual embeddings by path and create unified samples
    samples = build_samples(load_embeddings(emb_audio), load_embeddings(emb_visual))
    # Ensure all labels in samples exist in mapping (expand mapping on the fly)
    for s in samples:
        if s["label"] not in label_to_idx:
            label_to_idx[s["label"]] = len(label_to_idx)
    from train_fusion import EmbeddingDataset  # local import to avoid cycles at load time

    ds = EmbeddingDataset(samples, label_to_idx)
    if balance and shuffle:
        # Compute per-class frequencies and inverse-frequency weights for balancing
        counts = [0] * len(label_to_idx)
        for _, _, lbl in ds:
            counts[int(lbl.item())] += 1
        weights = [0.0] * len(ds)
        for i, (_, _, lbl) in enumerate(ds):
            freq = counts[int(lbl.item())]
            weights[i] = 1.0 / max(freq, 1)
        # Weighted sampler keeps epochs balanced even with class imbalance
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_on_source(
    source: Dict[str, str],
    device: str,
    batch_size: int,
    epochs: int,
    lr: float,
    balance: bool,
    weight_decay: float,
    use_cosine: bool,
) -> Dict[str, object]:
    from train_fusion import train

    # Collect all labels present in the source train/val splits to build a stable mapping
    label_set = set()
    for key in ["audio_train", "visual_train", "audio_val", "visual_val"]:
        path = source.get(key)
        if path:
            for item in load_embeddings(Path(path)) or []:
                label_set.add(item["label"])
    label_to_idx = {lbl: i for i, lbl in enumerate(sorted(label_set))}

    # Dataloaders for source training/validation (optionally class-balanced)
    train_loader = build_loader(
        Path(source.get("audio_train")) if source.get("audio_train") else None,
        Path(source.get("visual_train")) if source.get("visual_train") else None,
        label_to_idx,
        batch_size,
        shuffle=True,
        balance=balance,
    )
    val_loader = build_loader(
        Path(source.get("audio_val")) if source.get("audio_val") else None,
        Path(source.get("visual_val")) if source.get("visual_val") else None,
        label_to_idx,
        batch_size,
        shuffle=False,
        balance=False,
    )

    # Train a temporary fusion model on the source dataset
    tmp_model_path = Path("./models/tmp_fusion.pt")
    tmp_log_path = Path("./logs/tmp_training_log.txt")
    stats = train(
        train_loader,
        val_loader,
        device,
        epochs,
        lr,
        tmp_log_path,
        tmp_model_path,
        use_cosine=use_cosine,
        weight_decay=weight_decay,
    )
    return {
        "label_to_idx": label_to_idx,
        "model_path": tmp_model_path,
        "stats": stats,
    }


def evaluate_model(
    model_path: Path,
    label_to_idx: Dict[str, int],
    test_audio: Optional[Path],
    test_visual: Optional[Path],
    device: str,
) -> Dict[str, float]:
    # Build loader for the target test split (audio, visual, or both)
    loader = build_loader(test_audio, test_visual, label_to_idx, batch_size=32, shuffle=False, balance=False)
    # Load fusion Transformer and its state
    model = FusionTransformer().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state.get("state_dict", state))

    model.eval()
    total_acc = 0.0
    total_f1 = 0.0
    total = 0
    with torch.inference_mode():
        for audio, visual, label in loader:
            audio = audio.to(device)
            visual = visual.to(device)
            label = label.to(device)
            logits = model(audio, visual)
            # Accumulate accuracy and macro-F1 across the test set
            total_acc += accuracy(logits, label) * len(label)
            total_f1 += macro_f1(logits, label, num_classes=len(label_to_idx)) * len(label)
            total += len(label)
    return {"accuracy": total_acc / max(total, 1), "f1": total_f1 / max(total, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest of datasets")
    parser.add_argument("--output", type=Path, default=Path("./results/tables/table_I_cross_dataset.csv"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--balance", action="store_true", help="Use class-balanced sampling for source training")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--no_cosine", action="store_true", help="Disable cosine LR scheduler")
    args = parser.parse_args()

    # Load manifest describing train/val/test embedding paths for each dataset
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    datasets: List[Dict[str, str]] = manifest.get("datasets", [])

    rows: List[Dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Train", "Test", "Accuracy", "F1", "Drop"])
        writer.writeheader()

        for source in datasets:
            source_name = source["name"]
            # Train on the source dataset (uses temp checkpoint/log files)
            train_result = train_on_source(
                source,
                args.device,
                args.batch_size,
                args.epochs,
                args.lr,
                balance=args.balance,
                weight_decay=args.weight_decay,
                use_cosine=not args.no_cosine,
            )
            # Evaluate the trained model on its own test split (self accuracy)
            self_eval = evaluate_model(
                train_result["model_path"],
                train_result["label_to_idx"],
                Path(source.get("audio_test")) if source.get("audio_test") else None,
                Path(source.get("visual_test")) if source.get("visual_test") else None,
                args.device,
            )
            self_acc = self_eval["accuracy"]

            for target in datasets:
                target_name = target["name"]
                if target_name == source_name:
                    continue
                # Evaluate source-trained model on a different target dataset
                metrics = evaluate_model(
                    train_result["model_path"],
                    train_result["label_to_idx"],
                    Path(target.get("audio_test")) if target.get("audio_test") else None,
                    Path(target.get("visual_test")) if target.get("visual_test") else None,
                    args.device,
                )
                # Accuracy drop relative to self-evaluation
                drop = metrics["accuracy"] - self_acc
                row = {
                    "Train": source_name,
                    "Test": target_name,
                    "Accuracy": round(metrics["accuracy"] * 100, 2),
                    "F1": round(metrics["f1"], 4),
                    "Drop": round(drop * 100, 2),
                }
                writer.writerow(row)
                rows.append(row)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
