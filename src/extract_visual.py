"""Extract 1024-d visual embeddings using DINOv2 (vitl14) from CSVs.

Input CSV columns: path,label (images or video frames).
Output .npy contains list of dicts: path, label, embedding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

MODEL_NAME = "dinov2_vitl14"
IMG_SIZE = 518


def load_csv(csv_path: Path) -> List[Dict[str, str]]:
    # Parse headered CSV (path,label) into a list of dicts
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            vals = line.strip().split(",")
            if len(vals) != len(header):
                continue
            rows.append({header[i]: vals[i] for i in range(len(header))})
    return rows


def get_transform() -> T.Compose:
    # Resize, center-crop, normalize to ImageNet stats
    return T.Compose(
        [
            T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def extract_embedding(model: torch.nn.Module, image: Image.Image, device: str) -> torch.Tensor:
    # Run DINOv2 forward pass and grab a CLS-like token
    tensor = get_transform()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        feats = model.forward_features(tensor)
        cls = feats.get("x_norm_clstoken")
        if cls is None:
            cls = feats.get("cls_token")
        if cls is None:
            cls = feats.get("x_norm_clf")
        if cls is None:
            raise RuntimeError("DINOv2 forward_features did not return a CLS token")
        return cls.squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract visual embeddings with DINOv2")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load DINOv2 backbone
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME).to(args.device)
    model.eval()

    # Build rows and accumulate embeddings
    rows = load_csv(args.csv)
    data: List[Dict[str, object]] = []

    for row in rows:
        img_path = Path(row["path"])
        label = row["label"]
        # Open image as RGB and extract a 1024-d feature vector
        image = Image.open(img_path).convert("RGB")
        emb = extract_embedding(model, image, args.device)
        data.append({"path": img_path.as_posix(), "label": label, "embedding": emb.cpu().numpy()})

    # Save array of dicts atomically plus a small manifest JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_name(args.output.name + ".tmp")
    with tmp_output.open("wb") as f:
        np.save(f, np.array(data, dtype=object), allow_pickle=True)
    tmp_output.replace(args.output)

    manifest = {
        "source_csv": args.csv.as_posix(),
        "count": len(data),
        "model": MODEL_NAME,
    }
    meta_path = args.output.with_suffix(".json")
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    with tmp_meta.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp_meta.replace(meta_path)

    # Log completion info
    print(f"Saved {len(data)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
