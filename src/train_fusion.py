"""Train a fusion Transformer on precomputed audio/visual embeddings.

- Inputs: .npy embedding files produced by extract_audio.py / extract_visual.py
- Handles missing modality by zero-filling the absent branch.
- Saves best model (by val accuracy) and a training log.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

LABELS = ["angry", "happy", "sad", "neutral", "surprise"]
AUDIO_DIM = 1024
VISUAL_DIM = 1024


def load_embeddings(path: Optional[Path]) -> Optional[List[Dict[str, object]]]:
    # Load list-of-dicts saved by extract_audio/visual (path, label, embedding)
    if path is None:
        return None
    arr = np.load(path, allow_pickle=True)
    return list(arr.tolist())


def build_samples(
    audio_data: Optional[List[Dict[str, object]]], visual_data: Optional[List[Dict[str, object]]]
) -> List[Dict[str, object]]:
    # Merge audio and visual entries by path, zero-fill missing modality
    audio_map = {d["path"]: d for d in audio_data} if audio_data else {}
    visual_map = {d["path"]: d for d in visual_data} if visual_data else {}
    keys = set(audio_map) | set(visual_map)
    samples: List[Dict[str, object]] = []
    for key in keys:
        a = audio_map.get(key)
        v = visual_map.get(key)
        label = (a or v)["label"]
        audio_vec = a["embedding"] if a is not None else np.zeros(AUDIO_DIM, dtype=np.float32)
        visual_vec = v["embedding"] if v is not None else np.zeros(VISUAL_DIM, dtype=np.float32)
        samples.append({"path": key, "label": label, "audio": audio_vec, "visual": visual_vec})
    return samples


class EmbeddingDataset(Dataset):
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
        return audio, visual, label


class FusionTransformer(nn.Module):
    def __init__(self, hidden_dim: int = 768, nhead: int = 8, num_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        # Project each modality to a shared hidden dim
        self.audio_proj = nn.Linear(AUDIO_DIM, hidden_dim)
        self.visual_proj = nn.Linear(VISUAL_DIM, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # Transformer encoder over [CLS, audio, visual] tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, len(LABELS))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        # Build token sequence and classify using CLS output
        a = self.audio_proj(audio)
        v = self.visual_proj(visual)
        tokens = torch.stack([a, v], dim=1)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_labels = pred.argmax(dim=1)
    correct = (pred_labels == target).sum().item()
    return correct / len(target) if len(target) > 0 else 0.0


def macro_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    # Compute macro-F1 manually to avoid dependency on sklearn
    pred_labels = pred.argmax(dim=1)
    f1_per_class = []
    for cls in range(num_classes):
        tp = ((pred_labels == cls) & (target == cls)).sum().item()
        fp = ((pred_labels == cls) & (target != cls)).sum().item()
        fn = ((pred_labels != cls) & (target == cls)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom > 0 else 0.0
        f1_per_class.append(f1)
    return float(sum(f1_per_class) / len(f1_per_class))


def eval_epoch(model: FusionTransformer, loader: DataLoader, device: str) -> Tuple[float, float, float]:
    # Evaluate loss/accuracy/macro-F1 over a loader
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for audio, visual, label in loader:
            audio = audio.to(device)
            visual = visual.to(device)
            label = label.to(device)
            logits = model(audio, visual)
            loss = criterion(logits, label)
            total_loss += loss.item() * len(label)
            total_acc += accuracy(logits, label) * len(label)
            total_f1 += macro_f1(logits, label, num_classes=len(LABELS)) * len(label)
    count = len(loader.dataset)
    return total_loss / max(count, 1), total_acc / max(count, 1), total_f1 / max(count, 1)


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    log_path: Path,
    model_path: Path,
    use_cosine: bool = True,
    weight_decay: float = 1e-2,
) -> Dict[str, float]:
    # Main training loop for the fusion Transformer
    model = FusionTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if use_cosine:
        # Optional cosine annealing learning rate schedule
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = -math.inf
    best_state: Dict[str, object] = {}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("epoch,train_loss,val_loss,val_acc,val_f1\n")
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for audio, visual, label in train_loader:
                audio = audio.to(device)
                visual = visual.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                logits = model(audio, visual)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(label)

            train_loss = running_loss / max(len(train_loader.dataset), 1)
            val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, device)
            if scheduler is not None:
                scheduler.step()
            log.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_acc:.4f},{val_f1:.4f}\n")
            log.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    "model": model.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "epoch": epoch,
                }
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "epoch": epoch}, model_path)

    return {
        "best_val_acc": best_val_acc,
        "best_state_epoch": best_state.get("epoch", 0),
    }


def build_loader(
    audio_path: Optional[Path],
    visual_path: Optional[Path],
    label_to_idx: Dict[str, int],
    batch_size: int,
    shuffle: bool,
    balance: bool = False,
) -> DataLoader:
    # Construct dataset and optionally apply class-balanced sampling
    samples = build_samples(load_embeddings(audio_path), load_embeddings(visual_path))
    ds = EmbeddingDataset(samples, label_to_idx)
    if balance:
        # Inverse-frequency weights to balance classes
        counts = [0] * len(label_to_idx)
        for _, _, lbl in ds:
            counts[int(lbl.item())] += 1
        weights = [0.0] * len(ds)
        for i, (_, _, lbl) in enumerate(ds):
            freq = counts[int(lbl.item())]
            weights[i] = 1.0 / max(freq, 1)
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fusion Transformer")
    parser.add_argument("--train_audio", type=Path, required=False)
    parser.add_argument("--train_visual", type=Path, required=False)
    parser.add_argument("--val_audio", type=Path, required=False)
    parser.add_argument("--val_visual", type=Path, required=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_path", type=Path, default=Path("./logs/training_log.txt"))
    parser.add_argument("--model_path", type=Path, default=Path("./models/fusion_transformer.pt"))
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--cosine", action="store_true", help="Use cosine annealing LR")
    parser.add_argument("--balance", action="store_true", help="Use class-balanced sampling")
    args = parser.parse_args()

    # Derive label set from provided data
    # (ensures consistent label ordering across splits)
    label_set = set()
    for path in [args.train_audio, args.train_visual, args.val_audio, args.val_visual]:
        if path is None:
            continue
        for item in load_embeddings(path) or []:
            label_set.add(item["label"])
    label_list = [lbl for lbl in LABELS if lbl in label_set] or LABELS
    label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}

    train_loader = build_loader(args.train_audio, args.train_visual, label_to_idx, args.batch_size, shuffle=True, balance=args.balance)
    val_loader = build_loader(args.val_audio, args.val_visual, label_to_idx, args.batch_size, shuffle=False, balance=False)

    stats = train(
        train_loader,
        val_loader,
        args.device,
        args.epochs,
        args.lr,
        args.log_path,
        args.model_path,
        use_cosine=args.cosine,
        weight_decay=args.weight_decay,
    )

    meta = {
        "label_to_idx": label_to_idx,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_val_acc": stats["best_val_acc"],
    }
    meta_path = args.model_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Training complete. Best val acc={stats['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
