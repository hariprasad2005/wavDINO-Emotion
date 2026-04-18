"""
Data loading utilities for emotion recognition datasets
Supports CREMA-D, RAVDESS, and AFEW datasets
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Tuple, List, Dict, Optional
from pathlib import Path


class EmotionDataset(Dataset):
    """
    Dataset class for emotion recognition with pre-extracted embeddings
    """
    
    EMOTION_LABELS = {
        'neutral': 0,
        'calm': 0,  # Map calm to neutral
        'happy': 1,
        'sad': 2,
        'angry': 3,
        'fear': 4,
        'surprise': 5,
        'disgust': 4  # Map disgust to fear for 6-class
    }
    
    def __init__(self, 
                 audio_embedding_file: str,
                 visual_embedding_file: str,
                 metadata_file: str,
                 dataset_type: str = 'train'):
        """
        Args:
            audio_embedding_file: Path to .npy file with audio embeddings
            visual_embedding_file: Path to .npy file with visual embeddings
            metadata_file: Path to .json file with metadata (labels, filenames)
            dataset_type: 'train', 'val', or 'test'
        """
        self.dataset_type = dataset_type
        
        # Load embeddings
        self.audio_embeddings = np.load(audio_embedding_file)
        self.visual_embeddings = np.load(visual_embedding_file)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.samples = self.metadata.get('samples', [])
        self.labels = self.metadata.get('labels', [])
        
        assert len(self.audio_embeddings) == len(self.labels), \
            f"Audio embeddings ({len(self.audio_embeddings)}) and labels ({len(self.labels)}) mismatch"
        assert len(self.visual_embeddings) == len(self.labels), \
            f"Visual embeddings ({len(self.visual_embeddings)}) and labels ({len(self.labels)}) mismatch"
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """
        Returns:
            audio_emb: Audio embedding (1024,)
            visual_emb: Visual embedding (1024,)
            label: Emotion label (0-5)
            sample_name: Sample identifier
        """
        audio_emb = torch.FloatTensor(self.audio_embeddings[idx])
        visual_emb = torch.FloatTensor(self.visual_embeddings[idx])
        label = int(self.labels[idx])
        sample_name = self.samples[idx] if idx < len(self.samples) else f"sample_{idx}"
        
        return audio_emb, visual_emb, label, sample_name


class AudioOnlyDataset(Dataset):
    """Dataset for audio-only emotion recognition"""
    
    EMOTION_LABELS = EmotionDataset.EMOTION_LABELS
    
    def __init__(self, 
                 audio_embedding_file: str,
                 metadata_file: str,
                 dataset_type: str = 'train'):
        """Args:
            audio_embedding_file: Path to .npy file with audio embeddings
            metadata_file: Path to .json file with metadata
            dataset_type: 'train', 'val', or 'test'
        """
        self.dataset_type = dataset_type
        self.audio_embeddings = np.load(audio_embedding_file)
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.samples = self.metadata.get('samples', [])
        self.labels = self.metadata.get('labels', [])
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        audio_emb = torch.FloatTensor(self.audio_embeddings[idx])
        label = int(self.labels[idx])
        sample_name = self.samples[idx] if idx < len(self.samples) else f"sample_{idx}"
        
        return audio_emb, label, sample_name


class VisualOnlyDataset(Dataset):
    """Dataset for visual-only emotion recognition"""
    
    EMOTION_LABELS = EmotionDataset.EMOTION_LABELS
    
    def __init__(self, 
                 visual_embedding_file: str,
                 metadata_file: str,
                 dataset_type: str = 'train'):
        """Args:
            visual_embedding_file: Path to .npy file with visual embeddings
            metadata_file: Path to .json file with metadata
            dataset_type: 'train', 'val', or 'test'
        """
        self.dataset_type = dataset_type
        self.visual_embeddings = np.load(visual_embedding_file)
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.samples = self.metadata.get('samples', [])
        self.labels = self.metadata.get('labels', [])
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        visual_emb = torch.FloatTensor(self.visual_embeddings[idx])
        label = int(self.labels[idx])
        sample_name = self.samples[idx] if idx < len(self.samples) else f"sample_{idx}"
        
        return visual_emb, label, sample_name


class DatasetManager:
    """Manager class to handle dataset creation and dataloaders"""
    
    DATASETS = {
        'crema': {
            'audio_train': 'embeddings/audio/crema_train.npy',
            'audio_val': 'embeddings/audio/crema_val.npy',
            'audio_test': 'embeddings/audio/crema_test.npy',
            'audio_meta_train': 'embeddings/audio/crema_train.json',
            'audio_meta_val': 'embeddings/audio/crema_val.json',
            'audio_meta_test': 'embeddings/audio/crema_test.json',
            'visual_train': 'embeddings/visual/crema_train.npy',
            'visual_val': 'embeddings/visual/crema_val.npy',
            'visual_test': 'embeddings/visual/crema_test.npy',
            'visual_meta_train': 'embeddings/visual/crema_train.json',
            'visual_meta_val': 'embeddings/visual/crema_val.json',
            'visual_meta_test': 'embeddings/visual/crema_test.json',
        },
        'ravdess': {
            'audio_train': 'embeddings/audio/ravdess_train.npy',
            'audio_val': 'embeddings/audio/ravdess_val.npy',
            'audio_test': 'embeddings/audio/ravdess_test.npy',
            'audio_meta_train': 'embeddings/audio/ravdess_train.json',
            'audio_meta_val': 'embeddings/audio/ravdess_val.json',
            'audio_meta_test': 'embeddings/audio/ravdess_test.json',
            'visual_train': 'embeddings/visual/ravdess_train.npy',
            'visual_val': 'embeddings/visual/ravdess_val.npy',
            'visual_test': 'embeddings/visual/ravdess_test.npy',
            'visual_meta_train': 'embeddings/visual/ravdess_train.json',
            'visual_meta_val': 'embeddings/visual/ravdess_val.json',
            'visual_meta_test': 'embeddings/visual/ravdess_test.json',
        },
        'afew': {
            'visual_train': 'embeddings/visual/afew_train.npy',
            'visual_val': 'embeddings/visual/afew_val.npy',
            'visual_test': 'embeddings/visual/afew_test.npy',
            'visual_meta_train': 'embeddings/visual/afew_train.json',
            'visual_meta_val': 'embeddings/visual/afew_val.json',
            'visual_meta_test': 'embeddings/visual/afew_test.json',
        }
    }
    
    @classmethod
    def get_dataloaders(cls, 
                       dataset_name: str,
                       batch_size: int = 32,
                       num_workers: int = 0,
                       modality: str = 'fusion',
                       base_path: str = '.') -> Dict[str, DataLoader]:
        """
        Get dataloaders for a specific dataset
        
        Args:
            dataset_name: 'crema', 'ravdess', or 'afew'
            batch_size: Batch size
            num_workers: Number of workers for data loading
            modality: 'fusion', 'audio', or 'visual'
            base_path: Base path to embeddings
        
        Returns:
            Dictionary with 'train', 'val', 'test' dataloaders
        """
        dataset_paths = cls.DATASETS.get(dataset_name)
        if not dataset_paths:
            raise ValueError(f"Dataset {dataset_name} not found. Available: {list(cls.DATASETS.keys())}")
        
        dataloaders = {}
        
        for split in ['train', 'val', 'test']:
            if modality in ['fusion', 'audio', 'visual']:
                if dataset_name == 'afew' and modality == 'audio':
                    raise ValueError("AFEW dataset does not have audio embeddings")
                
                if modality == 'fusion':
                    audio_key = f'audio_{split}'
                    visual_key = f'visual_{split}'
                    
                    if audio_key not in dataset_paths or visual_key not in dataset_paths:
                        raise ValueError(f"Dataset {dataset_name} doesn't support {modality} modality for split {split}")
                    
                    audio_path = os.path.join(base_path, dataset_paths[audio_key])
                    visual_path = os.path.join(base_path, dataset_paths[visual_key])
                    audio_meta_path = os.path.join(base_path, dataset_paths[f'audio_meta_{split}'])
                    visual_meta_path = os.path.join(base_path, dataset_paths[f'visual_meta_{split}'])
                    
                    dataset = EmotionDataset(audio_path, visual_path, audio_meta_path, split)
                
                elif modality == 'audio':
                    audio_key = f'audio_{split}'
                    audio_path = os.path.join(base_path, dataset_paths[audio_key])
                    audio_meta_path = os.path.join(base_path, dataset_paths[f'audio_meta_{split}'])
                    
                    dataset = AudioOnlyDataset(audio_path, audio_meta_path, split)
                
                else:  # visual
                    visual_key = f'visual_{split}'
                    visual_path = os.path.join(base_path, dataset_paths[visual_key])
                    visual_meta_path = os.path.join(base_path, dataset_paths[f'visual_meta_{split}'])
                    
                    dataset = VisualOnlyDataset(visual_path, visual_meta_path, split)
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True
            )
            dataloaders[split] = dataloader
        
        return dataloaders


if __name__ == "__main__":
    # Test data loading
    try:
        dataloaders = DatasetManager.get_dataloaders('crema', batch_size=4, modality='fusion')
        print(f"Created dataloaders: {list(dataloaders.keys())}")
        
        for batch_audio, batch_visual, batch_labels, batch_names in dataloaders['train']:
            print(f"Audio shape: {batch_audio.shape}")
            print(f"Visual shape: {batch_visual.shape}")
            print(f"Labels: {batch_labels}")
            break
    except Exception as e:
        print(f"Data loading test: {e}")
        print("Note: This is expected if embedding files don't exist yet")
