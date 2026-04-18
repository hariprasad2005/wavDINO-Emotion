"""
Example scripts demonstrating model usage
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.models.wavdino_emotion import WavDINOEmotion
from src.models.inference import ModelLoader
from src.data.dataset import DatasetManager, EmotionDataset


def example_1_model_creation():
    """Example 1: Create and initialize model"""
    print("=" * 50)
    print("Example 1: Model Creation")
    print("=" * 50)
    
    # Create model
    model = WavDINOEmotion(num_emotions=6)
    
    # Print model information
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass with random inputs
    batch_size = 4
    audio_emb = torch.randn(batch_size, 1024)
    visual_emb = torch.randn(batch_size, 1024)
    
    logits, probs = model(audio_emb, visual_emb)
    
    print(f"\nForward pass successful!")
    print(f"  Input audio shape: {audio_emb.shape}")
    print(f"  Input visual shape: {visual_emb.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Output probs shape: {probs.shape}")
    print(f"  Sample probabilities: {probs[0].detach().numpy()}")
    print()


def example_2_model_loading():
    """Example 2: Load pretrained model and make predictions"""
    print("=" * 50)
    print("Example 2: Model Loading and Inference")
    print("=" * 50)
    
    # Create dummy model for demonstration
    model = WavDINOEmotion(num_emotions=6)
    checkpoint_path = "example_model.pt"
    
    # Save model
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    # Load using ModelLoader
    try:
        loader = ModelLoader(checkpoint_path, device='cpu')
        print(f"Model loaded successfully!")
        
        # Make prediction
        audio_emb = np.random.randn(1024).astype(np.float32)
        visual_emb = np.random.randn(1024).astype(np.float32)
        
        emotion, confidence, probs = loader.predict(audio_emb, visual_emb)
        
        print(f"\nPrediction Results:")
        print(f"  Predicted emotion: {emotion}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Probabilities:")
        for emotion_name, prob in probs.items():
            print(f"    {emotion_name}: {prob:.4f}")
    
    except Exception as e:
        print(f"Note: {e}")
    
    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print()


def example_3_data_loading():
    """Example 3: Load datasets"""
    print("=" * 50)
    print("Example 3: Data Loading")
    print("=" * 50)
    
    try:
        # Get dataloaders
        dataloaders = DatasetManager.get_dataloaders(
            'crema',
            batch_size=4,
            modality='fusion'
        )
        
        print("Dataloaders created successfully!")
        for split, loader in dataloaders.items():
            print(f"  {split.upper()}: {len(loader.dataset)} samples, {len(loader)} batches")
        
        # Show sample batch
        print("\nSample batch from training data:")
        for audio_emb, visual_emb, labels, names in dataloaders['train']:
            print(f"  Audio embeddings shape: {audio_emb.shape}")
            print(f"  Visual embeddings shape: {visual_emb.shape}")
            print(f"  Labels: {labels}")
            print(f"  Sample names: {names[:2]}...")
            break
    
    except Exception as e:
        print(f"Note: {e}")
        print("This is expected if embedding files don't exist yet.")
    print()


def example_4_audio_visual_fusion():
    """Example 4: Audio-visual fusion"""
    print("=" * 50)
    print("Example 4: Audio-Visual Fusion")
    print("=" * 50)
    
    model = WavDINOEmotion(num_emotions=6)
    
    # Create different modality inputs
    batch_size = 8
    
    # Audio-visual fusion
    audio_emb = torch.randn(batch_size, 1024)
    visual_emb = torch.randn(batch_size, 1024)
    
    logits_fusion, probs_fusion = model(audio_emb, visual_emb)
    
    print(f"Audio-Visual Fusion:")
    print(f"  Fused logits shape: {logits_fusion.shape}")
    print(f"  Mean probability per class: {probs_fusion.mean(dim=0).detach().numpy()}")
    
    # Audio-only (simplified)
    audio_proj = model.audio_projection(audio_emb)
    logits_audio = model.classifier(audio_proj)
    probs_audio = torch.softmax(logits_audio, dim=1)
    
    print(f"\nAudio-Only Pathway:")
    print(f"  Audio logits shape: {logits_audio.shape}")
    print(f"  Mean probability per class: {probs_audio.mean(dim=0).detach().numpy()}")
    
    # Visual-only (simplified)
    visual_proj = model.visual_projection(visual_emb)
    logits_visual = model.classifier(visual_proj)
    probs_visual = torch.softmax(logits_visual, dim=1)
    
    print(f"\nVisual-Only Pathway:")
    print(f"  Visual logits shape: {logits_visual.shape}")
    print(f"  Mean probability per class: {probs_visual.mean(dim=0).detach().numpy()}")
    print()


def example_5_training_setup():
    """Example 5: Training setup"""
    print("=" * 50)
    print("Example 5: Training Setup")
    print("=" * 50)
    
    model = WavDINOEmotion(num_emotions=6)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    print(f"Optimizer: AdamW")
    print(f"Learning rate: 3e-4")
    print(f"Weight decay: 1e-5")
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    print(f"\nLoss function: CrossEntropyLoss")
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    print(f"Scheduler: CosineAnnealingLR (T_max=50)")
    
    # Test forward-backward pass
    audio_emb = torch.randn(4, 1024)
    visual_emb = torch.randn(4, 1024)
    labels = torch.randint(0, 6, (4,))
    
    logits, _ = model(audio_emb, visual_emb)
    loss = criterion(logits, labels)
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    print(f"\nTraining step completed!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("wavDINO-Emotion Examples")
    print("=" * 50 + "\n")
    
    example_1_model_creation()
    example_2_model_loading()
    example_3_data_loading()
    example_4_audio_visual_fusion()
    example_5_training_setup()
    
    print("=" * 50)
    print("All examples completed!")
    print("=" * 50)
