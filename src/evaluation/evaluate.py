"""
Evaluation scripts for wavDINO-Emotion model
Includes single dataset and cross-dataset evaluation
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.wavdino_emotion import WavDINOEmotion
from data.dataset import DatasetManager
from utils.metrics import MetricsCalculator


class Evaluator:
    """Evaluation class for wavDINO-Emotion"""
    
    EMOTION_LABELS = {
        0: 'neutral',
        1: 'happy',
        2: 'sad',
        3: 'angry',
        4: 'fear',
        5: 'surprise'
    }
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 num_emotions: int = 6):
        
        self.model = model.to(device)
        self.device = device
        self.num_emotions = num_emotions
        self.metrics = MetricsCalculator(num_emotions)
    
    def evaluate(self, 
                 dataloader,
                 verbose: bool = True) -> Dict:
        """
        Evaluate model on a dataset
        
        Returns:
            Dictionary with accuracy, f1, precision, recall, and confusion matrix
        """
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probs = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc='Evaluating', disable=not verbose)
            
            for batch in progress_bar:
                if len(batch) == 4:  # Audio-visual fusion
                    audio_emb, visual_emb, labels, _ = batch
                    audio_emb = audio_emb.to(self.device)
                    visual_emb = visual_emb.to(self.device)
                    logits, probs = self.model(audio_emb, visual_emb)
                elif len(batch) == 3:  # Audio or visual only
                    emb, labels, _ = batch
                    emb = emb.to(self.device)
                    logits = self.model(emb)
                    probs = torch.softmax(logits, dim=1)
                else:
                    continue
                
                labels = labels.to(self.device)
                predictions = torch.argmax(logits, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = np.mean(all_labels == all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_predictions, labels=range(self.num_emotions))
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': all_predictions.tolist(),
            'probabilities': all_probs.tolist(),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': classification_report(
                all_labels, all_predictions,
                target_names=[self.EMOTION_LABELS[i] for i in range(self.num_emotions)],
                zero_division=0
            )
        }
        
        if verbose:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print("\nClassification Report:")
            print(results['classification_report'])
        
        return results
    
    def evaluate_and_save(self,
                         dataloader,
                         output_path: str,
                         verbose: bool = True) -> Dict:
        """Evaluate and save results to file"""
        results = self.evaluate(dataloader, verbose=verbose)
        
        # Save results
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score'],
                'confusion_matrix': results['confusion_matrix'],
                'classification_report': results['classification_report']
            }, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        
        return results


class CrossDatasetEvaluator:
    """Cross-dataset evaluation"""
    
    def __init__(self, 
                 model_path: str,
                 device: torch.device = None):
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.model_path = model_path
        self.evaluator = Evaluator(
            model=self._load_model(),
            device=device
        )
    
    def _load_model(self) -> WavDINOEmotion:
        """Load model from checkpoint"""
        model = WavDINOEmotion(num_emotions=6)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def evaluate_cross_dataset(self,
                              train_dataset: str,
                              test_datasets: List[str],
                              batch_size: int = 32) -> Dict:
        """
        Evaluate model trained on one dataset against others
        
        Args:
            train_dataset: Dataset the model was trained on
            test_datasets: List of datasets to test on
            batch_size: Batch size for evaluation
        
        Returns:
            Dictionary with cross-dataset evaluation results
        """
        results = {
            'train_dataset': train_dataset,
            'test_results': {}
        }
        
        for test_dataset in test_datasets:
            print(f"\nEvaluating {train_dataset} model on {test_dataset}...")
            
            try:
                dataloaders = DatasetManager.get_dataloaders(
                    test_dataset,
                    batch_size=batch_size,
                    modality='fusion'
                )
                
                eval_results = self.evaluator.evaluate(dataloaders['test'])
                results['test_results'][test_dataset] = {
                    'accuracy': eval_results['accuracy'],
                    'f1_score': eval_results['f1_score'],
                    'confusion_matrix': eval_results['confusion_matrix']
                }
                
                print(f"  Accuracy: {eval_results['accuracy']:.4f}")
                print(f"  F1-Score: {eval_results['f1_score']:.4f}")
            
            except Exception as e:
                print(f"  Error evaluating on {test_dataset}: {e}")
                results['test_results'][test_dataset] = {'error': str(e)}
        
        return results


def plot_confusion_matrix(conf_matrix: np.ndarray,
                         emotion_labels: List[str],
                         output_path: str):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_labels,
                yticklabels=emotion_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate wavDINO-Emotion model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='crema',
                        choices=['crema', 'ravdess', 'afew'],
                        help='Dataset to evaluate on')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Data split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--cross-dataset', action='store_true',
                        help='Perform cross-dataset evaluation')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    if args.cross_dataset:
        # Cross-dataset evaluation
        evaluator = CrossDatasetEvaluator(args.model, device)
        results = evaluator.evaluate_cross_dataset(
            train_dataset=args.dataset,
            test_datasets=[d for d in ['crema', 'ravdess', 'afew'] if d != args.dataset],
            batch_size=args.batch_size
        )
        
        output_path = Path(args.output_dir) / f'cross_dataset_results_{args.dataset}.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    else:
        # Single dataset evaluation
        print(f"Evaluating on {args.dataset} {args.split} split...")
        
        # Load model
        model = WavDINOEmotion(num_emotions=6)
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Load data
        dataloaders = DatasetManager.get_dataloaders(
            args.dataset,
            batch_size=args.batch_size,
            modality='fusion'
        )
        
        # Evaluate
        evaluator = Evaluator(model, device)
        results = evaluator.evaluate_and_save(
            dataloaders[args.split],
            Path(args.output_dir) / f'eval_{args.dataset}_{args.split}.json'
        )
        
        # Plot confusion matrix
        conf_matrix = np.array(results['confusion_matrix'])
        emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise']
        plot_confusion_matrix(
            conf_matrix,
            emotion_labels,
            Path(args.output_dir) / f'confusion_matrix_{args.dataset}_{args.split}.png'
        )


if __name__ == "__main__":
    main()
