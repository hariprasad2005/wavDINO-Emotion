"""Generate reproducible train/val/test CSV splits for CREMA-D, RAVDESS, and AFEW.

- Uses 70/15/15 with random_state=42 by default.
- Expects folder-per-label layout with the five labels: angry, happy, sad, neutral, surprise.
- Writes CSVs with columns: path,label
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

LABELS = ["angry", "happy", "sad", "neutral", "surprise"]
DEFAULT_SEED = 42


def list_files(root: Path, exts: Sequence[str]) -> List[Tuple[str, str]]:
    # Collect (path, label) for every file under root/<label> with allowed extensions
    items: List[Tuple[str, str]] = []
    for label in LABELS:
        label_dir = root / label
        if not label_dir.exists():
            continue
        for ext in exts:
            for path in label_dir.rglob(f"*.{ext}"):
                items.append((path.as_posix(), label))
    return items


def split_train_val_test(
    items: Sequence[Tuple[str, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    # Shuffle once with a fixed seed and slice into train/val/test
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_split = shuffled[:n_train]
    val_split = shuffled[n_train : n_train + n_val]
    test_split = shuffled[n_train + n_val :]
    return train_split, val_split, test_split


def subsample(items: Sequence[Tuple[str, str]], fraction: float, seed: int) -> List[Tuple[str, str]]:
    # Randomly keep a fraction of the items (at least one)
    if fraction >= 1.0:
        return list(items)
    rng = random.Random(seed)
    k = max(1, int(len(items) * fraction))
    return rng.sample(list(items), k)


def cap_per_label(items: Sequence[Tuple[str, str]], cap: int, seed: int) -> List[Tuple[str, str]]:
    # Limit the number of samples per label to mitigate class imbalance
    if cap <= 0:
        return list(items)
    rng = random.Random(seed)
    grouped: dict[str, List[Tuple[str, str]]] = {lbl: [] for lbl in LABELS}
    for path, lbl in items:
        if lbl in grouped:
            grouped[lbl].append((path, lbl))
    capped: List[Tuple[str, str]] = []
    for lbl, rows in grouped.items():
        if len(rows) <= cap:
            capped.extend(rows)
        else:
            capped.extend(rng.sample(rows, cap))
    return capped


def write_csv(rows: Iterable[Tuple[str, str]], path: Path) -> None:
    # Write header and rows as path,label to a CSV file
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for p, lbl in rows:
            writer.writerow([p, lbl])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--crema_root", type=Path, default=Path("G:/CREMA-D-CLASSIFIED"))
    parser.add_argument("--ravdess_root", type=Path, default=Path("G:/ravdess1/RAVDESS_CLASSIFIED"))
    parser.add_argument("--afew_root", type=Path, default=Path("G:/afew1/AFEW_CLASSIFIED"))
    parser.add_argument("--output_dir", type=Path, default=Path("./splits"))
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--afew_fraction", type=float, default=1.0, help="Use this fraction of AFEW before splitting")
    parser.add_argument("--afew_cap_per_label", type=int, default=0, help="Optional cap of samples per AFEW label before splitting")
    parser.add_argument("--crema_fraction", type=float, default=1.0, help="Use this fraction of CREMA-D before splitting")
    parser.add_argument("--ravdess_fraction", type=float, default=1.0, help="Use this fraction of RAVDESS before splitting")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # CREMA-D (audio)
    # - Load all wavs, optionally subsample, then split
    crema_items_full = list_files(args.crema_root, ["wav"])
    crema_items = subsample(crema_items_full, args.crema_fraction, args.seed)
    crema_train, crema_val, crema_test = split_train_val_test(
        crema_items, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    write_csv(crema_train, args.output_dir / "crema_train.csv")
    write_csv(crema_val, args.output_dir / "crema_val.csv")
    write_csv(crema_test, args.output_dir / "crema_test.csv")

    # RAVDESS (audio)
    # - Same pipeline as CREMA-D
    ravdess_items_full = list_files(args.ravdess_root, ["wav"])
    ravdess_items = subsample(ravdess_items_full, args.ravdess_fraction, args.seed)
    ravdess_train, ravdess_val, ravdess_test = split_train_val_test(
        ravdess_items, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    write_csv(ravdess_train, args.output_dir / "ravdess_train.csv")
    write_csv(ravdess_val, args.output_dir / "ravdess_val.csv")
    write_csv(ravdess_test, args.output_dir / "ravdess_test.csv")

    # AFEW (visual); treat images as visual modality
    # - Optionally subsample and cap per label before splitting
    afew_items_full = list_files(args.afew_root, ["jpg", "png"])
    afew_items_frac = subsample(afew_items_full, args.afew_fraction, args.seed)
    afew_items = cap_per_label(afew_items_frac, args.afew_cap_per_label, args.seed)
    afew_train, afew_val, afew_test = split_train_val_test(
        afew_items, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    write_csv(afew_train, args.output_dir / "afew_train.csv")
    write_csv(afew_val, args.output_dir / "afew_val.csv")
    write_csv(afew_test, args.output_dir / "afew_test.csv")

    print("CREMA-D files (total):", len(crema_items_full))
    print("CREMA-D files (used):", len(crema_items))
    print("RAVDESS files (total):", len(ravdess_items_full))
    print("RAVDESS files (used):", len(ravdess_items))
    print("AFEW files (total):", len(afew_items_full))
    print("AFEW files (used):", len(afew_items))


if __name__ == "__main__":
    main()
