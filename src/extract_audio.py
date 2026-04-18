"""Extract 1024-d audio embeddings using Wav2Vec2 XLSR-53 (transformers + librosa).

Input CSV columns: path,label
Output: .npy list of dicts with keys: path, label, embedding
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

TARGET_SR = 16000
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"


def load_csv(csv_path: Path) -> List[Dict[str, str]]:
    # Parse a simple CSV with header path,label into a list of dicts
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            vals = line.strip().split(",")
            if len(vals) != len(header):
                continue
            rows.append({header[i]: vals[i] for i in range(len(header))})
    return rows


def extract_embedding(
    model: Wav2Vec2Model,
    feat_extractor: Wav2Vec2FeatureExtractor,
    wav: np.ndarray,
    device: str,
) -> torch.Tensor:
    # Convert waveform to model inputs, run forward pass, and mean-pool over time
    inputs = feat_extractor(wav, sampling_rate=TARGET_SR, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)
    with torch.inference_mode():
        outputs = model(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, 1024)
        pooled = hidden.mean(dim=1)  # (B, 1024)
        return pooled.squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio embeddings with Wav2Vec2 XLSR-53 (librosa + transformers)")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with path,label columns")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    # Load feature extractor and backbone once, move to device
    feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Read CSV entries and accumulate dicts of path/label/embedding
    rows = load_csv(args.csv)
    data: List[Dict[str, object]] = []

    for row in rows:
        wav_path = Path(row["path"])
        label = row["label"]
        # Load audio (mono, resampled) and extract a fixed 1024-d vector
        wav, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        emb = extract_embedding(model, feat_extractor, wav, device)
        data.append({"path": wav_path.as_posix(), "label": label, "embedding": emb.cpu().numpy()})

    # Write data to .npy atomically and store a small JSON manifest
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

    # Log a short completion message
    print(f"Saved {len(data)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
