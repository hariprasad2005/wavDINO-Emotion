"""
Visualization and results analysis for wavDINO-Emotion
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import argparse


class ResultsVisualizer:
    """Visualize training and evaluation results"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        sns.set_style(style)
        self.emotion_names = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise']
    
    def plot_training_history(self, 
                            history: Dict,
                            output_path: str = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = history.get('epoch', range(1, len(history['train_loss']) + 1))
        
        # Loss plot
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], 'o-', label='Train', linewidth=2)
        ax.plot(epochs, history['val_loss'], 's-', label='Validation', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax = axes[0, 1]
        ax.plot(epochs, history['train_acc'], 'o-', label='Train', linewidth=2)
        ax.plot(epochs, history['val_acc'], 's-', label='Validation', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # F1-Score plot
        ax = axes[1, 0]
        ax.plot(epochs, history['val_f1'], 'D-', color='green', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title('Validation F1-Score', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax = axes[1, 1]
        ax.plot(epochs, history['learning_rate'], '^-', color='red', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self,
                            conf_matrix: np.ndarray,
                            output_path: str = None,
                            normalize: bool = False):
        """Plot confusion matrix"""
        if normalize:
            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            cmap = 'Blues'
            fmt = '.2%'
        else:
            cmap = 'Blues'
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap=cmap,
                   xticklabels=self.emotion_names,
                   yticklabels=self.emotion_names,
                   cbar_kws={'label': 'Count'})
        
        plt.xlabel('Predicted Emotion', fontsize=12, fontweight='bold')
        plt.ylabel('True Emotion', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_cross_dataset_results(self,
                                 results: Dict,
                                 output_path: str = None):
        """Plot cross-dataset evaluation results"""
        datasets = list(results['test_results'].keys())
        accuracies = [results['test_results'][d]['accuracy'] for d in datasets]
        f1_scores = [results['test_results'][d]['f1_score'] for d in datasets]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy bar plot
        ax = axes[0]
        bars = ax.bar(datasets, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(datasets)],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'Cross-Dataset Accuracy\n(Trained on {results["train_dataset"].upper()})',
                    fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # F1-Score bar plot
        ax = axes[1]
        bars = ax.bar(datasets, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(datasets)],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Cross-Dataset F1-Score\n(Trained on {results["train_dataset"].upper()})',
                    fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{f1:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Cross-dataset results plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_emotion_accuracy_per_class(self,
                                       conf_matrix: np.ndarray,
                                       output_path: str = None):
        """Plot per-class accuracy"""
        per_class_acc = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.emotion_names, per_class_acc, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Per-Emotion Classification Accuracy', fontsize=14, fontweight='bold')
        plt.ylim([0, 1])
        
        for bar, acc in zip(bars, per_class_acc):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Per-class accuracy plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_model_comparison(self,
                            models: Dict[str, Dict],
                            output_path: str = None):
        """Plot comparison of multiple models"""
        model_names = list(models.keys())
        accuracies = [models[m]['accuracy'] for m in model_names]
        f1_scores = [models[m]['f1_score'] for m in model_names]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        ax = axes[0]
        x = np.arange(len(model_names))
        bars = ax.bar(x, accuracies, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # F1-Score comparison
        ax = axes[1]
        bars = ax.bar(x, f1_scores, color='seagreen', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{f1:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize wavDINO-Emotion results')
    parser.add_argument('--training-history', type=str,
                       help='Path to training history JSON file')
    parser.add_argument('--confusion-matrix', type=str,
                       help='Path to confusion matrix JSON file')
    parser.add_argument('--cross-dataset', type=str,
                       help='Path to cross-dataset results JSON file')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = ResultsVisualizer()
    
    # Plot training history
    if args.training_history:
        with open(args.training_history) as f:
            history = json.load(f)
        
        output_path = output_dir / 'training_history.png'
        visualizer.plot_training_history(history, str(output_path))
    
    # Plot confusion matrix
    if args.confusion_matrix:
        with open(args.confusion_matrix) as f:
            data = json.load(f)
            conf_matrix = np.array(data['confusion_matrix'])
        
        output_path = output_dir / 'confusion_matrix.png'
        visualizer.plot_confusion_matrix(conf_matrix, str(output_path))
        
        output_path = output_dir / 'confusion_matrix_normalized.png'
        visualizer.plot_confusion_matrix(conf_matrix, str(output_path), normalize=True)
        
        output_path = output_dir / 'per_class_accuracy.png'
        visualizer.plot_emotion_accuracy_per_class(conf_matrix, str(output_path))
    
    # Plot cross-dataset results
    if args.cross_dataset:
        with open(args.cross_dataset) as f:
            results = json.load(f)
        
        output_path = output_dir / 'cross_dataset_results.png'
        visualizer.plot_cross_dataset_results(results, str(output_path))


if __name__ == '__main__':
    main()
