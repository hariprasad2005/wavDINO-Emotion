"""
wavDINO-Emotion Project Documentation
Complete guide for the self-supervised audio-visual emotion recognition framework

Based on: "wavDINO-Emotion: A Self-Supervised Audio Visual Transformer for Emotion Recognition"
IConSCEPT 2025 - International Conference on Signal Processing, Computation, Electronics, 
Power and Telecommunication
"""

# ==============================================================================
# PROJECT OVERVIEW
# ==============================================================================

"""
wavDINO-Emotion is a multimodal emotion recognition framework that combines:
- Wav2Vec 2.0 for audio feature extraction
- DINOv2 for visual feature extraction
- Transformer-based cross-modal fusion
- Self-supervised learning approach

The model achieves state-of-the-art performance on multiple emotion recognition
datasets including CREMA-D, RAVDESS, and AFEW.
"""

# ==============================================================================
# PROJECT STRUCTURE
# ==============================================================================

PROJECT_STRUCTURE = """
wavDINO-Emotion/
│
├── src/                          # Source code
│   ├── models/
│   │   ├── __init__.py          # Package initialization
│   │   ├── wavdino_emotion.py   # Main model architecture
│   │   └── inference.py          # Inference utilities
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py            # Data loading and preprocessing
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py              # Training script
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py           # Evaluation and cross-dataset eval
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py            # Metrics calculation
│       └── logger.py             # Logging utilities
│
├── embeddings/                   # Pre-extracted embeddings
│   ├── audio/                    # Wav2Vec 2.0 embeddings
│   │   ├── crema_train.npy
│   │   ├── crema_train.json
│   │   ├── crema_val.npy
│   │   ├── crema_val.json
│   │   ├── crema_test.npy
│   │   ├── crema_test.json
│   │   ├── ravdess_train.npy
│   │   └── ... (similar structure for ravdess)
│   │
│   └── visual/                   # DINOv2 embeddings
│       ├── afew_train.npy
│       ├── afew_train.json
│       ├── ... (similar structure for all datasets)
│
├── models/                       # Pre-trained model checkpoints
│   ├── crema_d.pt               # Trained on CREMA-D
│   ├── ravdess.pt               # Trained on RAVDESS
│   ├── afew.pt                  # Trained on AFEW
│   ├── crema_to_ravdess.pt      # Cross-dataset transfer
│   ├── audio_only.pt            # Audio-only model
│   ├── visual_static.pt         # Visual-only model
│   ├── audio_visual_static.pt   # Static fusion
│   └── audio_visual_temporal_proposed.pt  # Temporal fusion
│
├── splits/                       # Train/val/test splits
│   ├── crema_train.csv
│   ├── crema_val.csv
│   ├── crema_test.csv
│   ├── ravdess_train.csv
│   └── ... (similar for all datasets)
│
├── results/                      # Evaluation results
│   ├── eval_crema_test.json
│   ├── confusion_matrix_crema_test.png
│   ├── cross_dataset_results_crema.json
│   └── ... (results for all evaluations)
│
├── checkpoints/                  # Training checkpoints
│   ├── model_epoch_001.pt
│   ├── model_epoch_010.pt
│   ├── model_best.pt
│   ├── training_history.json
│   └── ... (checkpoints during training)
│
├── configs/
│   └── config.json              # Project configuration
│
├── logs/                         # Training logs
│   ├── training.log
│   └── ...
│
├── main.py                       # Main entry point
├── examples.py                   # Example usage scripts
├── test_model.py                # Unit tests
├── visualize_results.py         # Results visualization
├── QUICKSTART.py                # Quick start guide
├── requirements.txt             # Python dependencies
├── README.md                    # Project README
├── README_PROJECT.md            # Detailed project documentation
└── ARCHITECTURE.md              # Model architecture details
"""

# ==============================================================================
# KEY COMPONENTS
# ==============================================================================

ARCHITECTURE = """
The wavDINO-Emotion model consists of:

1. Audio Branch:
   - Input: Wav2Vec 2.0 embeddings (1024-dim)
   - AudioProjection: Linear layer + LayerNorm
   - Output: 1024-dim projected embeddings

2. Visual Branch:
   - Input: DINOv2 embeddings (1024-dim)
   - VisualProjection: Linear layer + LayerNorm
   - Output: 1024-dim projected embeddings

3. Fusion Module:
   - Concatenate audio and visual embeddings (batch_size, 2, 1024)
   - Transformer Encoder:
     * Multi-head attention (8 heads)
     * 2 transformer layers
     * Hidden dimension: 2048
     * Dropout: 0.1
   - Mean pooling across modalities

4. Classification Head:
   - Input: Fused embeddings (1024-dim)
   - 3-layer MLP:
     * Linear(1024, 512) + ReLU + Dropout
     * Linear(512, 256) + ReLU + Dropout
     * Linear(256, 6)
   - Output: Logits for 6 emotion classes
"""

DATASETS = """
Three emotion recognition datasets are supported:

1. CREMA-D (Crowd-Sourced Emotional Multimodal Actors Database):
   - 7,442 samples from 91 actors
   - 5 emotions: Happy, Sad, Angry, Fear, Disgust
   - Format: MP4 video files
   - Emotions mapped to 6-class: {neutral, happy, sad, angry, fear, surprise}
   - Mapped Disgust to Fear, added Neutral (calm)
   - Train/Val/Test: 70% / 15% / 15%

2. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song):
   - Audio-visual recordings from 24 actors
   - 8 emotions: Neutral, Calm, Happy, Sad, Angry, Fear, Surprise, Disgust
   - Format: MP4 video files
   - Emotions mapped to 6-class: {neutral (calm), happy, sad, angry, fear, surprise}
   - Train/Val/Test: 70% / 15% / 15%

3. AFEW (Acted Facial Expressions in the Wild):
   - Real-world facial expressions
   - 6 emotions: Neutral, Happy, Sad, Angry, Surprised, Scared (Fear)
   - Format: MP4 video files
   - Train/Val/Test: 70% / 15% / 15%

Preprocessing steps:
- Audio: Resample to 16 kHz, trim silence, normalize
- Visual: Extract center frame, detect face (MTCNN), resize to 224x224
- Embeddings: Extract 1024-dim embeddings using Wav2Vec 2.0 and DINOv2
"""

# ==============================================================================
# TRAINING DETAILS
# ==============================================================================

TRAINING_HYPERPARAMETERS = """
Optimizer:
  - Type: AdamW
  - Learning rate: 3e-4
  - Weight decay: 1e-5

Loss Function:
  - CrossEntropyLoss

Batch Size: 32

Number of Epochs: 50

Learning Rate Schedule:
  - Cosine Annealing with T_max=50
  - No warmup

Gradient:
  - Gradient clipping: max_norm=1.0

Data Augmentation:
  - No augmentation (embeddings are pre-computed)

Evaluation Metrics:
  - Accuracy (macro)
  - F1-Score (macro)
  - Precision (per-class)
  - Recall (per-class)
  - Confusion Matrix
"""

PERFORMANCE = """
Single-Dataset Performance (Best Results):

Dataset         Accuracy    F1-Score    Precision   Recall
CREMA-D         86.9%       0.86        0.87        0.86
RAVDESS         86.7%       0.86        0.86        0.86
AFEW            71.7%       0.69        0.70        0.69

Cross-Dataset Evaluation Results:

Train Dataset   Test Dataset    Accuracy    F1-Score    Performance Drop
RAVDESS        CREMA-D         82.4%       0.81        -7.0%
RAVDESS        AFEW            68.3%       0.65        -8.4%
CREMA-D        RAVDESS         85.1%       0.84        -4.1%
CREMA-D        AFEW            70.2%       0.68        -6.5%
AFEW           RAVDESS         83.7%       0.82        -7.7%
AFEW           CREMA-D         81.5%       0.80        -7.7%

Model Comparison (on RAVDESS):

Model                                   Accuracy    F1-Score
Wav2Vec 2.0 Only                       84.7%       0.839
DINOv2 Only                            83.2%       0.821
CNN + LSTM                             81.3%       0.801
MERT (Multimodal Emotion Recognition)  82.5%       0.818
Wav2Vec 2.0 + DINOv2 (Proposed)       86.01%      0.8521
"""

# ==============================================================================
# USAGE GUIDE
# ==============================================================================

USAGE_EXAMPLES = """
1. TRAINING A MODEL
   python src/training/train.py \\
       --dataset crema \\
       --modality fusion \\
       --epochs 50 \\
       --batch-size 32 \\
       --lr 3e-4 \\
       --output-dir ./checkpoints

2. EVALUATING A MODEL
   python src/evaluation/evaluate.py \\
       --model models/crema_d.pt \\
       --dataset crema \\
       --split test \\
       --batch-size 32 \\
       --output-dir ./results

3. CROSS-DATASET EVALUATION
   python src/evaluation/evaluate.py \\
       --model models/crema_d.pt \\
       --dataset crema \\
       --cross-dataset \\
       --output-dir ./results

4. INFERENCE (PREDICTION)
   python main.py infer \\
       --model models/crema_d.pt \\
       --audio-emb path/to/audio_emb.npy \\
       --visual-emb path/to/visual_emb.npy

5. PYTHON API
   from src.models.inference import ModelLoader
   
   loader = ModelLoader('models/crema_d.pt', device='cuda')
   audio_emb = np.load('audio_embedding.npy')
   visual_emb = np.load('visual_embedding.npy')
   
   emotion, confidence, probs = loader.predict(audio_emb, visual_emb)
   print(f"Emotion: {emotion}, Confidence: {confidence:.4f}")
"""

# ==============================================================================
# EMBEDDING FORMAT
# ==============================================================================

EMBEDDING_FORMAT = """
Audio Embeddings (from Wav2Vec 2.0):
  - Shape: (num_samples, 1024)
  - Dtype: float32
  - File format: .npy
  - Extraction:
    1. Load audio file, resample to 16 kHz
    2. Pass through Wav2Vec 2.0 model
    3. Extract final layer output (1024-dim per sample)

Visual Embeddings (from DINOv2):
  - Shape: (num_samples, 1024)
  - Dtype: float32
  - File format: .npy
  - Extraction:
    1. Extract center frame from video
    2. Detect face using MTCNN
    3. Resize to 224x224
    4. Pass through DINOv2 (ViT-B) model
    5. Extract cls token embedding (1024-dim)

Metadata (JSON):
  {
    "samples": ["crema_103_0_0_3.mp4", "crema_103_0_0_4.mp4", ...],
    "labels": [3, 2, 1, 0, 2, ...]
  }
"""

# ==============================================================================
# FILE FORMATS
# ==============================================================================

FILE_FORMATS = """
Model Checkpoint (.pt):
  - PyTorch checkpoint dictionary
  - Keys: 'model_state_dict', 'optimizer_state_dict', 'epoch', 'train_history', etc.
  - Can be loaded with torch.load() and transferred to model with load_state_dict()

Results JSON:
  {
    "accuracy": 0.869,
    "f1_score": 0.86,
    "confusion_matrix": [[...], ...],
    "classification_report": "..."
  }

Training History JSON:
  {
    "epoch": [1, 2, ..., 50],
    "train_loss": [0.45, 0.42, ...],
    "train_acc": [0.75, 0.76, ...],
    "val_loss": [0.48, 0.46, ...],
    "val_acc": [0.74, 0.75, ...],
    "val_f1": [0.73, 0.75, ...],
    "learning_rate": [3e-4, 3e-4, ...]
  }
"""

# ==============================================================================
# TROUBLESHOOTING
# ==============================================================================

TROUBLESHOOTING = """
Issue: CUDA out of memory
Solution: Reduce batch size
  python src/training/train.py --batch-size 16

Issue: Data loading errors (embeddings not found)
Solution: Verify embeddings exist in embeddings/audio/ and embeddings/visual/
  - Check file names match dataset names (crema, ravdess, afew)
  - Verify .npy and .json files exist for each split

Issue: Poor model accuracy
Solutions:
  - Verify embeddings are properly normalized
  - Check emotion label mappings are correct (0-5)
  - Try training longer (increase --epochs)
  - Use cross-dataset model if domain is different
  - Reduce learning rate

Issue: Model not converging
Solutions:
  - Reduce learning rate (--lr 1e-4)
  - Enable gradient clipping (already enabled)
  - Check for NaN values in embeddings

Issue: Slow training
Solutions:
  - Use GPU if available (--device cuda)
  - Increase number of workers (--num-workers 4)
  - Reduce batch size
  - Use smaller model for testing

Issue: Memory leak during long training
Solution: Ensure CUDA cache is cleared between epochs
  - Already handled in training script
  - Restart Python process if needed
"""

# ==============================================================================
# EXTENDING THE PROJECT
# ==============================================================================

EXTENSIONS = """
1. ADD NEW DATASET:
   - Prepare embeddings in embeddings/audio/ and embeddings/visual/
   - Add dataset config to DatasetManager.DATASETS in src/data/dataset.py
   - Update emotion labels if different from 6-class

2. MODIFY MODEL ARCHITECTURE:
   - Edit WavDINOEmotion class in src/models/wavdino_emotion.py
   - Update embedding dimensions if using different feature extractors
   - Adjust number of transformer layers, attention heads, etc.

3. ADD NEW LOSS FUNCTION:
   - Replace CrossEntropyLoss in training script
   - Use different loss for class imbalance (e.g., WeightedCrossEntropyLoss)

4. IMPLEMENT DATA AUGMENTATION:
   - Add augmentation in EmotionDataset.__getitem__()
   - Consider: mixup, cutmix, temporal cropping, etc.

5. ADD EXPLAINABILITY:
   - Extract attention weights from transformer encoder
   - Visualize attention heatmaps for interpretability
   - Use LIME or SHAP for model explanation

6. DEPLOY MODEL:
   - Convert to ONNX format: torch.onnx.export()
   - Quantize for inference: torch.quantization
   - Create API server using FastAPI or Flask
"""

# ==============================================================================
# REFERENCES
# ==============================================================================

REFERENCES = """
[1] Nasersharif, B., & Namvarpour, M. (2024). Exploring the potential of Wav2vec 2.0 
    for speech emotion recognition using classifier combination and attention-based 
    feature fusion. The Journal of Supercomputing, 80(16), 23667–23688.

[2] Sun, C., Zhou, Y., Huang, X., Yang, J., & Hou, X. (2024). Combining Wav2vec 2.0 
    fine-tuning and ConLearnNet for speech emotion recognition. Electronics, 13(6), 1103.

[3] Atmaja, B. T., & Sasou, A. (2022). Evaluating self-supervised speech representations 
    for speech emotion recognition. IEEE Access, 10, 124396–124407.

[4] Park, S., Mark, M., Park, B., & Hong, H. (2023). Using speaker-specific emotion 
    representations in Wav2vec 2.0-based modules for speech emotion recognition.

[5] Chettaoui, T., Damer, N., & Boutros, F. (2025). Froundation: Are foundation models 
    ready for face recognition? Image and Vision Computing, 156, 105453.

[6] Kumar, K., et al. (2023). MERT: Multimodal Emotion Recognition Transformer with 
    Cross-Modal Attention. IEEE Transactions on Affective Computing.
"""

# ==============================================================================
# CITATION
# ==============================================================================

CITATION = """
If you use this project in your research, please cite:

@inproceedings{wavdinoemotion2025,
  title={wavDINO-Emotion: A Self-Supervised Audio-Visual Transformer for Emotion Recognition},
  author={Padma, E. and Prasath, Mohana G. and Chanthuru, S. R. and Hariprasad, M.},
  booktitle={2025 International Conference on Signal Processing, Computation, Electronics, 
             Power and Telecommunication (IConSCEPT)},
  year={2025}
}
"""

# ==============================================================================
# LICENSE
# ==============================================================================

LICENSE = """
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""

# ==============================================================================
# AUTHORS
# ==============================================================================

AUTHORS = """
Padma E
  Nandha Engineering College, Erode, Tamilnadu, India
  Email: padmasents@gmail.com

Mohana Prasath G
  Nandha Engineering College, Erode, Tamilnadu, India
  Email: mohanaprasath8917@gmail.com

Chanthuru S R
  Nandha Engineering College, Erode, Tamilnadu, India
  Email: chanthuruchanthuru77@gmail.com

Hariprasad M
  Nandha Engineering College, Erode, Tamilnadu, India
  Email: mrhariprasad23@gmail.com
"""


if __name__ == "__main__":
    print("wavDINO-Emotion Documentation")
    print("=" * 80)
    print("\nFor detailed information, see README.md and other documentation files.")
    print("\nQuick start: python QUICKSTART.py")
    print("Examples: python examples.py")
    print("Tests: python test_model.py")
