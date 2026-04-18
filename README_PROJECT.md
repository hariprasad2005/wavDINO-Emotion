# wavDINO-Emotion: Self-Supervised Audio-Visual Transformer for Emotion Recognition

A multimodal emotion recognition framework using self-supervised learning with Wav2Vec 2.0 for audio and DINOv2 for visual features.

## Project Structure

```
wavDINO-Emotion/
├── src/
│   ├── models/
│   │   ├── wavdino_emotion.py       # Main model architecture
│   │   └── inference.py              # Model inference utilities
│   ├── data/
│   │   └── dataset.py                # Data loading utilities
│   ├── training/
│   │   └── train.py                  # Training script
│   ├── evaluation/
│   │   └── evaluate.py               # Evaluation and cross-dataset eval
│   └── utils/
│       ├── metrics.py                # Metrics calculation
│       └── logger.py                 # Logging utilities
├── configs/
│   └── config.json                   # Configuration file
├── embeddings/                       # Pre-extracted embeddings
│   ├── audio/                        # Audio embeddings from Wav2Vec 2.0
│   └── visual/                       # Visual embeddings from DINOv2
├── models/                           # Trained model checkpoints
├── splits/                           # Train/val/test splits
├── results/                          # Evaluation results
├── checkpoints/                      # Training checkpoints
└── README.md
```

## Model Architecture

The model consists of:
1. **Audio Branch**: Wav2Vec 2.0 embeddings (1024-dim) with linear projection
2. **Visual Branch**: DINOv2 embeddings (1024-dim) with linear projection
3. **Fusion Module**: Transformer encoder with multi-head attention
4. **Classification Head**: 3-layer MLP classifier

## Datasets

- **CREMA-D**: 7,442 samples from 91 actors (5 emotions)
- **RAVDESS**: Audio-visual recordings from 24 actors (8 emotions)
- **AFEW**: Real-world facial expressions (varying emotions)

## Training

```bash
# Train on CREMA-D with audio-visual fusion
python src/training/train.py \
  --dataset crema \
  --modality fusion \
  --epochs 50 \
  --batch-size 32 \
  --lr 3e-4

# Train audio-only model
python src/training/train.py \
  --dataset ravdess \
  --modality audio \
  --epochs 50

# Train visual-only model
python src/training/train.py \
  --dataset afew \
  --modality visual \
  --epochs 50
```

## Evaluation

```bash
# Single dataset evaluation
python src/evaluation/evaluate.py \
  --model models/model_best.pt \
  --dataset crema \
  --split test

# Cross-dataset evaluation
python src/evaluation/evaluate.py \
  --model models/crema_d.pt \
  --dataset crema \
  --cross-dataset
```

## Model Files

- `crema_d.pt`: Model trained on CREMA-D
- `ravdess.pt`: Model trained on RAVDESS
- `afew.pt`: Model trained on AFEW
- `crema_to_ravdess.pt`: Cross-dataset transfer (CREMA-D → RAVDESS)
- `crema_to_afew.pt`: Cross-dataset transfer (CREMA-D → AFEW)
- `ravdess_to_crema.pt`: Cross-dataset transfer (RAVDESS → CREMA-D)
- `audio_only.pt`: Audio-only emotion recognition
- `visual_static.pt`: Visual-only emotion recognition
- `audio_visual_static.pt`: Audio-visual static fusion
- `audio_visual_temporal_proposed.pt`: Temporal fusion model

## Key Results

- **CREMA-D**: 86.9% accuracy, 0.86 F1-score
- **RAVDESS**: 86.7% accuracy, 0.86 F1-score
- **AFEW**: 71.7% accuracy, 0.69 F1-score

## Cross-Dataset Performance

| Train | Test | Accuracy | F1-Score |
|-------|------|----------|----------|
| RAVDESS | CREMA-D | 82.4% | 0.81 |
| RAVDESS | AFEW | 68.3% | 0.65 |
| CREMA-D | RAVDESS | 85.1% | 0.84 |
| CREMA-D | AFEW | 70.2% | 0.68 |
| AFEW | RAVDESS | 83.7% | 0.82 |
| AFEW | CREMA-D | 81.5% | 0.80 |

## Training Parameters

- **Optimizer**: AdamW
- **Learning Rate**: 3e-4
- **Loss Function**: Cross-entropy
- **Batch Size**: 32
- **Epochs**: 50
- **Scheduler**: Cosine annealing

## Requirements

```
torch>=1.9.0
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
```

## Installation

```bash
pip install -r requirements.txt
```

## Citation

```
@inproceedings{wavdinoemotion2025,
  title={wavDINO-Emotion: A Self-Supervised Audio-Visual Transformer for Emotion Recognition},
  booktitle={2025 International Conference on Signal Processing, Computation, Electronics, Power and Telecommunication (IConSCEPT)},
  year={2025}
}
```

## Authors

- Padma E (Nandha Engineering College)
- Mohana Prasath G (Nandha Engineering College)
- Chanthuru S R (Nandha Engineering College)
- Hariprasad M (Nandha Engineering College)

## License

MIT License
