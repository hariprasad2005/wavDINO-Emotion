"""
wavDINO-Emotion: Self-Supervised Audio-Visual Transformer for Emotion Recognition
Based on paper: wavDINO-Emotion (IConSCEPT 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class AudioProjection(nn.Module):
    """Linear projection layer for audio embeddings from Wav2Vec 2.0"""
    
    def __init__(self, input_dim: int = 1024, output_dim: int = 1024):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.layer_norm(x)
        return x


class VisualProjection(nn.Module):
    """Linear projection layer for visual embeddings from DINOv2"""
    
    def __init__(self, input_dim: int = 1024, output_dim: int = 1024):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.layer_norm(x)
        return x


class MultiHeadAttentionFusion(nn.Module):
    """Transformer encoder with multi-head attention for cross-modal fusion"""
    
    def __init__(self, embedding_dim: int = 1024, num_heads: int = 8, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Multi-head attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, concatenated_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concatenated_embeddings: (batch_size, 2, embedding_dim)
        
        Returns:
            fused: (batch_size, embedding_dim)
        """
        # Apply transformer encoder
        encoded = self.transformer_encoder(concatenated_embeddings)  # (batch, 2, embedding_dim)
        
        # Average pooling across the two modalities
        fused = encoded.mean(dim=1)  # (batch, embedding_dim)
        
        return fused


class WavDINOEmotion(nn.Module):
    """
    Complete wavDINO-Emotion model combining Wav2Vec 2.0 and DINOv2 embeddings
    with transformer-based cross-modal fusion for emotion recognition.
    """
    
    def __init__(self, 
                 audio_embedding_dim: int = 1024,
                 visual_embedding_dim: int = 1024,
                 fusion_dim: int = 1024,
                 num_emotions: int = 6,
                 num_heads: int = 8,
                 num_transformer_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            audio_embedding_dim: Dimension of Wav2Vec 2.0 embeddings
            visual_embedding_dim: Dimension of DINOv2 embeddings
            fusion_dim: Dimension after projection
            num_emotions: Number of emotion classes
            num_heads: Number of attention heads
            num_transformer_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.audio_embedding_dim = audio_embedding_dim
        self.visual_embedding_dim = visual_embedding_dim
        self.fusion_dim = fusion_dim
        self.num_emotions = num_emotions
        
        # Audio and visual projection layers
        self.audio_projection = AudioProjection(audio_embedding_dim, fusion_dim)
        self.visual_projection = VisualProjection(visual_embedding_dim, fusion_dim)
        
        # Multi-head attention fusion
        self.fusion_module = MultiHeadAttentionFusion(
            embedding_dim=fusion_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )
    
    def forward(self, audio_embedding: torch.Tensor, 
                visual_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            audio_embedding: (batch_size, audio_embedding_dim) - from Wav2Vec 2.0
            visual_embedding: (batch_size, visual_embedding_dim) - from DINOv2
        
        Returns:
            logits: (batch_size, num_emotions) - raw model outputs
            predictions: (batch_size, num_emotions) - softmax probabilities
        """
        # Project embeddings to common dimension
        audio_proj = self.audio_projection(audio_embedding)  # (batch, fusion_dim)
        visual_proj = self.visual_projection(visual_embedding)  # (batch, fusion_dim)
        
        # Stack embeddings for attention
        concatenated = torch.stack([audio_proj, visual_proj], dim=1)  # (batch, 2, fusion_dim)
        
        # Fuse modalities with transformer
        fused = self.fusion_module(concatenated)  # (batch, fusion_dim)
        
        # Classify emotions
        logits = self.classifier(fused)  # (batch, num_emotions)
        predictions = F.softmax(logits, dim=1)
        
        return logits, predictions
    
    def get_attention_weights(self, audio_embedding: torch.Tensor,
                             visual_embedding: torch.Tensor) -> torch.Tensor:
        """Get attention weights from the transformer for visualization"""
        audio_proj = self.audio_projection(audio_embedding)
        visual_proj = self.visual_projection(visual_embedding)
        concatenated = torch.stack([audio_proj, visual_proj], dim=1)
        
        # Get encoder attention weights
        encoder = self.fusion_module.transformer_encoder
        attention_weights = []
        
        for layer in encoder.layers:
            # This extracts attention from the self-attention layer
            q = layer.self_attn.in_proj_weight[:self.fusion_dim]
            attention_weights.append(q)
        
        return concatenated


def create_model(num_emotions: int = 6, **kwargs) -> WavDINOEmotion:
    """Factory function to create wavDINO-Emotion model"""
    return WavDINOEmotion(num_emotions=num_emotions, **kwargs)


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    audio_emb = torch.randn(batch_size, 1024)
    visual_emb = torch.randn(batch_size, 1024)
    
    model = WavDINOEmotion(num_emotions=6)
    logits, predictions = model(audio_emb, visual_emb)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
