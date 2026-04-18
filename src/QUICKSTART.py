"""
Quick Start Guide for wavDINO-Emotion
"""

# INSTALLATION
# ============
# 1. Install dependencies:
#    pip install -r requirements.txt
#
# 2. Ensure you have the following directory structure:
#    - embeddings/audio/ (audio embeddings from Wav2Vec 2.0)
#    - embeddings/visual/ (visual embeddings from DINOv2)
#    - models/ (pre-trained model checkpoints)


# BASIC USAGE
# ===========

# Example 1: Create and test the model
# ------------------------------------
import torch
from src.models.wavdino_emotion import WavDINOEmotion

# Create model
model = WavDINOEmotion(num_emotions=6)

# Create dummy embeddings
audio_emb = torch.randn(4, 1024)      # (batch_size, embedding_dim)
visual_emb = torch.randn(4, 1024)     # (batch_size, embedding_dim)

# Forward pass
logits, probabilities = model(audio_emb, visual_emb)
print(f"Output shape: {logits.shape}")  # (4, 6)
print(f"Probabilities: {probabilities[0]}")


# Example 2: Load pre-trained model and make predictions
# ------------------------------------------------------
from src.models.inference import ModelLoader
import numpy as np

loader = ModelLoader('models/crema_d.pt', device='cuda')

# Create sample embeddings
audio_embedding = np.random.randn(1024).astype(np.float32)
visual_embedding = np.random.randn(1024).astype(np.float32)

# Make prediction
emotion, confidence, probabilities = loader.predict(audio_embedding, visual_embedding)
print(f"Predicted emotion: {emotion}")
print(f"Confidence: {confidence:.4f}")
print(f"All probabilities: {probabilities}")


# Example 3: Train on a dataset
# ----------------------------
import subprocess
import sys

# Train on CREMA-D with audio-visual fusion
cmd = [
    sys.executable, 'src/training/train.py',
    '--dataset', 'crema',
    '--modality', 'fusion',
    '--epochs', '50',
    '--batch-size', '32',
    '--lr', '3e-4'
]
subprocess.run(cmd)


# Example 4: Evaluate a trained model
# -----------------------------------
from src.evaluation.evaluate import Evaluator
from src.data.dataset import DatasetManager

# Load model
model = WavDINOEmotion(num_emotions=6)
checkpoint = torch.load('models/crema_d.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Load test data
dataloaders = DatasetManager.get_dataloaders('crema', batch_size=32, modality='fusion')

# Evaluate
evaluator = Evaluator(model, torch.device('cuda'))
results = evaluator.evaluate(dataloaders['test'])

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1-Score: {results['f1_score']:.4f}")


# Example 5: Cross-dataset evaluation
# -----------------------------------
from src.evaluation.evaluate import CrossDatasetEvaluator

evaluator = CrossDatasetEvaluator('models/crema_d.pt')
results = evaluator.evaluate_cross_dataset(
    train_dataset='crema',
    test_datasets=['ravdess', 'afew']
)

for dataset, metrics in results['test_results'].items():
    print(f"{dataset}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")


# COMMAND LINE INTERFACE
# ======================

# Train model:
# python src/training/train.py --dataset crema --modality fusion --epochs 50

# Evaluate model:
# python src/evaluation/evaluate.py --model models/crema_d.pt --dataset crema --split test

# Cross-dataset evaluation:
# python src/evaluation/evaluate.py --model models/crema_d.pt --dataset crema --cross-dataset

# Using main.py:
# python main.py train --dataset crema --modality fusion
# python main.py eval --model models/crema_d.pt --dataset crema
# python main.py cross-eval --model models/crema_d.pt --dataset crema
# python main.py infer --model models/crema_d.pt --audio-emb path/to/audio.npy --visual-emb path/to/visual.npy


# MODEL FILES DESCRIPTION
# =======================

# Single-dataset models:
# - crema_d.pt: Trained on CREMA-D, best on CREMA-D (86.9% acc)
# - ravdess.pt: Trained on RAVDESS, best on RAVDESS (86.7% acc)
# - afew.pt: Trained on AFEW, best on AFEW (71.7% acc)

# Cross-dataset transfer models:
# - crema_to_ravdess.pt: CREMA-D trained, tested on RAVDESS (85.1% acc)
# - crema_to_afew.pt: CREMA-D trained, tested on AFEW (70.2% acc)
# - ravdess_to_crema.pt: RAVDESS trained, tested on CREMA-D (82.4% acc)

# Modality-specific models:
# - audio_only.pt: Audio embeddings only
# - visual_static.pt: Visual embeddings only
# - audio_visual_static.pt: Concatenated fusion
# - audio_visual_temporal_proposed.pt: Proposed temporal fusion


# EMOTION CLASSES
# ===============
EMOTIONS = {
    0: 'neutral',
    1: 'happy',
    2: 'sad',
    3: 'angry',
    4: 'fear',
    5: 'surprise'
}


# EMBEDDING PREPARATION
# ====================

# Audio Embeddings:
# 1. Resample audio to 16 kHz
# 2. Trim silence
# 3. Normalize
# 4. Extract using Wav2Vec 2.0: (sample_length, 1024)

# Visual Embeddings:
# 1. Extract center frame from video
# 2. Detect face using MTCNN
# 3. Resize to 224x224
# 4. Normalize
# 5. Extract using DINOv2 (ViT-B): (1024,)


# DATASET STRUCTURE
# =================

# Expected embeddings directory structure:
# embeddings/
#   audio/
#     crema_train.npy     # shape: (num_samples, 1024)
#     crema_train.json    # metadata with labels
#     crema_val.npy
#     crema_val.json
#     crema_test.npy
#     crema_test.json
#     ravdess_train.npy
#     ... (similar for ravdess and audio)
#   visual/
#     afew_train.npy
#     afew_train.json
#     ... (similar structure)


# TRAINING TIPS
# =============

# 1. Use learning rate 3e-4 with AdamW optimizer
# 2. Batch size of 32 works well
# 3. Use cosine annealing learning rate scheduler
# 4. Train for 50 epochs
# 5. Monitor validation accuracy for early stopping
# 6. Use cross-entropy loss for classification
# 7. Save best model based on validation accuracy

# Hyperparameter search suggestions:
# - Learning rate: [1e-4, 3e-4, 1e-3]
# - Batch size: [16, 32, 64]
# - Dropout: [0.1, 0.2, 0.3]
# - Num transformer layers: [1, 2, 3]
# - Num attention heads: [4, 8, 16]


# TROUBLESHOOTING
# ===============

# Issue: CUDA out of memory
# Solution: Reduce batch_size (e.g., --batch-size 16)

# Issue: Data loading errors
# Solution: Check embeddings path and file formats

# Issue: Low accuracy
# Solution: 
# - Verify embeddings are correctly normalized
# - Check that emotion labels are mapped correctly (0-5)
# - Try training for more epochs
# - Use cross-dataset model for different domains

# Issue: Model not converging
# Solution:
# - Reduce learning rate
# - Use gradient clipping (already enabled in training script)
# - Check for NaN values in embeddings


if __name__ == "__main__":
    print("wavDINO-Emotion Quick Start Guide")
    print("=" * 50)
    print("\nFor examples, run:")
    print("  python examples.py")
    print("\nFor training, run:")
    print("  python src/training/train.py --help")
    print("\nFor evaluation, run:")
    print("  python src/evaluation/evaluate.py --help")
