"""Late-fuse audio and visual emotion classifiers and report accuracy/F1.

Inputs:
- audio_model (.pt) + .json meta from train_audio_only.py
- visual_model (.pt) + .json meta from train_visual_only.py
- test audio/visual embeddings (.npy) from extract_audio.py / extract_visual.py

Outputs: printed metrics; optional CSV with per-sample predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from train_audio_only import EmbeddingClassifier as AudioClassifier
from train_visual_only import EmbeddingClassifier as VisualClassifier
from train_fusion import AUDIO_DIM, LABELS, VISUAL_DIM, accuracy, build_samples, load_embeddings, macro_f1


class PairedDataset(Dataset):
    def __init__(self, samples: Sequence[Dict[str, object]], label_to_idx: Dict[str, int]):
        self.samples = list(samples)
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        audio = torch.tensor(item["audio"], dtype=torch.float32)
        visual = torch.tensor(item["visual"], dtype=torch.float32)
        label = torch.tensor(self.label_to_idx[item["label"]], dtype=torch.long)
        return audio, visual, label, item["path"]


def load_meta(model_path: Path) -> Dict[str, object]:
    # Load the JSON metadata saved alongside a trained classifier
    meta_path = model_path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta JSON next to {model_path}")
    return json.loads(meta_path.read_text())


def build_loader(audio_path: Path, visual_path: Path, label_to_idx: Dict[str, int], batch_size: int) -> DataLoader:
    # Align audio/visual embeddings by path and create a paired dataset
    samples = build_samples(load_embeddings(audio_path), load_embeddings(visual_path))
    ds = PairedDataset(samples, label_to_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse audio + visual classifiers")
    parser.add_argument("--audio_model", type=Path, required=True)
    parser.add_argument("--visual_model", type=Path, required=True)
    parser.add_argument("--audio_embeddings", type=Path, required=True, help="Test audio embeddings .npy")
    parser.add_argument("--visual_embeddings", type=Path, required=True, help="Test visual embeddings .npy")
    parser.add_argument("--audio_weight", type=float, default=0.5)
    parser.add_argument("--visual_weight", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preds_csv", type=Path, required=False, help="Optional CSV for per-sample predictions")
    args = parser.parse_args()

    audio_meta = load_meta(args.audio_model)
    visual_meta = load_meta(args.visual_model)

    # Ensure both models use identical label indexing
    if audio_meta.get("label_to_idx") != visual_meta.get("label_to_idx"):
        raise ValueError("Audio and visual models use different label mappings; retrain to align")
    label_to_idx = {k: int(v) for k, v in audio_meta["label_to_idx"].items()}
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    aw = args.audio_weight
    vw = args.visual_weight
    if aw + vw <= 0:
        raise ValueError("audio_weight + visual_weight must be > 0")
    # Normalize weights to sum to 1.0
    aw = aw / (aw + vw)
    vw = vw / (aw + vw)

    # Rebuild both classifiers from metadata for consistent dimensions
    audio_model = AudioClassifier(
        input_dim=int(audio_meta.get("input_dim", AUDIO_DIM)),
        hidden_dim=int(audio_meta.get("hidden_dim", 512)),
        dropout=float(audio_meta.get("dropout", 0.2)),
        num_classes=len(label_to_idx),
    ).to(args.device)
    audio_state = torch.load(args.audio_model, map_location=args.device)
    audio_model.load_state_dict(audio_state.get("state_dict", audio_state))
    audio_model.eval()

    visual_model = VisualClassifier(
        input_dim=int(visual_meta.get("input_dim", VISUAL_DIM)),
        hidden_dim=int(visual_meta.get("hidden_dim", 512)),
        dropout=float(visual_meta.get("dropout", 0.2)),
        num_classes=len(label_to_idx),
    ).to(args.device)
    visual_state = torch.load(args.visual_model, map_location=args.device)
    visual_model.load_state_dict(visual_state.get("state_dict", visual_state))
    visual_model.eval()

    loader = build_loader(args.audio_embeddings, args.visual_embeddings, label_to_idx, args.batch_size)

    total_acc = 0.0
    total_f1 = 0.0
    total = 0
    rows: List[Dict[str, object]] = []

    with torch.inference_mode():
        for audio, visual, label, path in loader:
            audio = audio.to(args.device)
            visual = visual.to(args.device)
            label = label.to(args.device)
            # Get logits from each branch and combine using normalized weights
            logits_audio = audio_model(audio)
            logits_visual = visual_model(visual)
            logits = aw * logits_audio + vw * logits_visual
            total_acc += accuracy(logits, label) * len(label)
            total_f1 += macro_f1(logits, label, num_classes=len(label_to_idx)) * len(label)
            total += len(label)

            if args.preds_csv is not None:
                preds = logits.argmax(dim=1).cpu().tolist()
                paths = list(path)
                for pth, pred_idx, true_idx in zip(paths, preds, label.cpu().tolist()):
                    rows.append(
                        {
                            "path": pth,
                            "pred": idx_to_label[pred_idx],
                            "true": idx_to_label[true_idx],
                        }
                    )

    metrics = {
        "accuracy": total_acc / max(total, 1),
        "f1": total_f1 / max(total, 1),
        "audio_weight": aw,
        "visual_weight": vw,
    }
    print(json.dumps(metrics, indent=2))

    if args.preds_csv is not None:
        args.preds_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.preds_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "pred", "true"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
