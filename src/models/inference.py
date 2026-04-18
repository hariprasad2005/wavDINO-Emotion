"""
Inference and model loading utilities
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.wavdino_emotion import WavDINOEmotion


class ModelLoader:
    """Load and manage wavDINO-Emotion models"""
    
    EMOTION_LABELS = {
        0: 'neutral',
        1: 'happy',
        2: 'sad',
        3: 'angry',
        4: 'fear',
        5: 'surprise'
    }
    
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize model loader
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            device: Device to load model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path: str) -> WavDINOEmotion:
        """Load model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model
        model = WavDINOEmotion(num_emotions=6)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        
        model = model.to(self.device)
        return model
    
    def predict(self, 
                audio_embedding: np.ndarray or torch.Tensor,
                visual_embedding: np.ndarray or torch.Tensor) -> Tuple[str, float, dict]:
        """
        Make prediction on audio-visual embeddings
        
        Args:
            audio_embedding: Audio embedding (1024,) from Wav2Vec 2.0
            visual_embedding: Visual embedding (1024,) from DINOv2
        
        Returns:
            emotion: Predicted emotion label
            confidence: Confidence score for prediction
            probabilities: Dictionary with probabilities for all emotions
        """
        # Convert to tensors if needed
        if isinstance(audio_embedding, np.ndarray):
            audio_embedding = torch.from_numpy(audio_embedding).float()
        if isinstance(visual_embedding, np.ndarray):
            visual_embedding = torch.from_numpy(visual_embedding).float()
        
        # Add batch dimension
        if audio_embedding.dim() == 1:
            audio_embedding = audio_embedding.unsqueeze(0)
        if visual_embedding.dim() == 1:
            visual_embedding = visual_embedding.unsqueeze(0)
        
        # Move to device
        audio_embedding = audio_embedding.to(self.device)
        visual_embedding = visual_embedding.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits, probs = self.model(audio_embedding, visual_embedding)
        
        # Get prediction
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
        
        # Build probabilities dict
        prob_dict = {
            self.EMOTION_LABELS[i]: probs[0, i].item()
            for i in range(6)
        }
        
        return self.EMOTION_LABELS[pred_idx], confidence, prob_dict
    
    def predict_audio_only(self, 
                          audio_embedding: np.ndarray or torch.Tensor) -> Tuple[str, float, dict]:
        """Prediction using audio embedding only"""
        if isinstance(audio_embedding, np.ndarray):
            audio_embedding = torch.from_numpy(audio_embedding).float()
        
        if audio_embedding.dim() == 1:
            audio_embedding = audio_embedding.unsqueeze(0)
        
        audio_embedding = audio_embedding.to(self.device)
        
        # Simple projection to 6 classes (audio only)
        with torch.no_grad():
            # Use model's audio projection
            audio_proj = self.model.audio_projection(audio_embedding)
            logits = self.model.classifier(audio_proj)
            probs = torch.softmax(logits, dim=1)
        
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
        
        prob_dict = {
            self.EMOTION_LABELS[i]: probs[0, i].item()
            for i in range(6)
        }
        
        return self.EMOTION_LABELS[pred_idx], confidence, prob_dict
    
    def predict_visual_only(self,
                           visual_embedding: np.ndarray or torch.Tensor) -> Tuple[str, float, dict]:
        """Prediction using visual embedding only"""
        if isinstance(visual_embedding, np.ndarray):
            visual_embedding = torch.from_numpy(visual_embedding).float()
        
        if visual_embedding.dim() == 1:
            visual_embedding = visual_embedding.unsqueeze(0)
        
        visual_embedding = visual_embedding.to(self.device)
        
        # Simple projection to 6 classes (visual only)
        with torch.no_grad():
            # Use model's visual projection
            visual_proj = self.model.visual_projection(visual_embedding)
            logits = self.model.classifier(visual_proj)
            probs = torch.softmax(logits, dim=1)
        
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
        
        prob_dict = {
            self.EMOTION_LABELS[i]: probs[0, i].item()
            for i in range(6)
        }
        
        return self.EMOTION_LABELS[pred_idx], confidence, prob_dict


def load_model(model_path: str, device: str = None) -> WavDINOEmotion:
    """Convenience function to load model"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    loader = ModelLoader(model_path, device)
    return loader.model


if __name__ == "__main__":
    # Example usage
    print("Model Loader Example")
    
    # Create dummy embeddings
    audio_emb = np.random.randn(1024).astype(np.float32)
    visual_emb = np.random.randn(1024).astype(np.float32)
    
    print(f"Audio embedding shape: {audio_emb.shape}")
    print(f"Visual embedding shape: {visual_emb.shape}")
    
    # Note: This requires a model file to exist
    # loader = ModelLoader('path/to/model.pt')
    # emotion, confidence, probs = loader.predict(audio_emb, visual_emb)
    # print(f"Predicted emotion: {emotion} (confidence: {confidence:.4f})")
    # print(f"Probabilities: {probs}")
