"""
Training script for wavDINO-Emotion model
Supports training on single dataset and cross-dataset evaluation
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import time
from datetime import datetime
import argparse
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.wavdino_emotion import WavDINOEmotion
from data.dataset import DatasetManager, EmotionDataset, AudioOnlyDataset, VisualOnlyDataset
from utils.metrics import MetricsCalculator
from utils.logger import TrainingLogger


class Trainer:
    """Main training class for wavDINO-Emotion"""
    
    def __init__(self, 
                 model: WavDINOEmotion,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 scheduler,
                 device: torch.device,
                 num_emotions: int = 6,
                 output_dir: str = './checkpoints'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_emotions = num_emotions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = MetricsCalculator(num_emotions)
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_labels = []
        all_predictions = []
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            if len(batch) == 4:  # Audio-visual fusion
                audio_emb, visual_emb, labels, _ = batch
                audio_emb = audio_emb.to(self.device)
                visual_emb = visual_emb.to(self.device)
                logits, _ = self.model(audio_emb, visual_emb)
            elif len(batch) == 3:  # Audio or visual only
                emb, labels, _ = batch
                emb = emb.to(self.device)
                logits = self.model(emb)
            else:
                continue
            
            labels = labels.to(self.device)
            
            # Forward pass
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.detach().cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float, float]:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validating')
            
            for batch in progress_bar:
                if len(batch) == 4:  # Audio-visual fusion
                    audio_emb, visual_emb, labels, _ = batch
                    audio_emb = audio_emb.to(self.device)
                    visual_emb = visual_emb.to(self.device)
                    logits, _ = self.model(audio_emb, visual_emb)
                elif len(batch) == 3:  # Audio or visual only
                    emb, labels, _ = batch
                    emb = emb.to(self.device)
                    logits = self.model(emb)
                else:
                    continue
                
                labels = labels.to(self.device)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.detach().cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
        f1_score = self.metrics.calculate_f1(all_labels, all_predictions)
        
        return avg_loss, accuracy, f1_score
    
    def train(self, num_epochs: int = 50):
        """Train the model for specified number of epochs"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['val_f1'].append(val_f1)
            self.train_history['learning_rate'].append(current_lr)
            
            elapsed = time.time() - start_time
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{num_epochs} [{elapsed:.1f}s]")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch + 1, is_best=True)
                print(f"  ✓ New best model saved (Acc: {val_acc:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
        
        print(f"\nTraining complete!")
        print(f"Best model: Epoch {self.best_epoch} with accuracy {self.best_val_accuracy:.4f}")
        
        return self.train_history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch
        }
        
        filename = f"model_epoch_{epoch:03d}.pt"
        if is_best:
            filename = "model_best.pt"
        
        filepath = self.output_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Train wavDINO-Emotion model')
    parser.add_argument('--dataset', type=str, default='crema', 
                        choices=['crema', 'ravdess', 'afew'],
                        help='Dataset to train on')
    parser.add_argument('--modality', type=str, default='fusion',
                        choices=['fusion', 'audio', 'visual'],
                        help='Modality to use')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading {args.dataset} dataset with {args.modality} modality...")
    try:
        dataloaders = DatasetManager.get_dataloaders(
            args.dataset,
            batch_size=args.batch_size,
            modality=args.modality
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Model setup
    if args.modality == 'fusion':
        model = WavDINOEmotion(num_emotions=6)
    elif args.modality == 'audio':
        # Audio-only model (simplified)
        model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 6)
        )
    else:  # visual
        # Visual-only model (simplified)
        model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 6)
        )
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir
    )
    
    # Train
    history = trainer.train(num_epochs=args.epochs)
    
    # Save training history
    history_file = Path(args.output_dir) / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_file}")


if __name__ == "__main__":
    main()
