"""
Metrics calculation utilities
"""

import numpy as np
from typing import List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    def __init__(self, num_classes: int = 6):
        self.num_classes = num_classes
    
    def calculate_f1(self, 
                     y_true: List[int] or np.ndarray,
                     y_pred: List[int] or np.ndarray,
                     average: str = 'macro') -> float:
        """Calculate F1 score"""
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    
    def calculate_precision(self,
                           y_true: List[int] or np.ndarray,
                           y_pred: List[int] or np.ndarray,
                           average: str = 'macro') -> float:
        """Calculate precision"""
        return precision_score(y_true, y_pred, average=average, zero_division=0)
    
    def calculate_recall(self,
                        y_true: List[int] or np.ndarray,
                        y_pred: List[int] or np.ndarray,
                        average: str = 'macro') -> float:
        """Calculate recall"""
        return recall_score(y_true, y_pred, average=average, zero_division=0)
    
    def calculate_accuracy(self,
                          y_true: List[int] or np.ndarray,
                          y_pred: List[int] or np.ndarray) -> float:
        """Calculate accuracy"""
        return accuracy_score(y_true, y_pred)
    
    def calculate_all(self,
                     y_true: List[int] or np.ndarray,
                     y_pred: List[int] or np.ndarray) -> dict:
        """Calculate all metrics"""
        return {
            'accuracy': self.calculate_accuracy(y_true, y_pred),
            'f1_macro': self.calculate_f1(y_true, y_pred, average='macro'),
            'f1_weighted': self.calculate_f1(y_true, y_pred, average='weighted'),
            'precision_macro': self.calculate_precision(y_true, y_pred, average='macro'),
            'precision_weighted': self.calculate_precision(y_true, y_pred, average='weighted'),
            'recall_macro': self.calculate_recall(y_true, y_pred, average='macro'),
            'recall_weighted': self.calculate_recall(y_true, y_pred, average='weighted'),
        }


class ConfusionMetrics:
    """Calculate metrics from confusion matrix"""
    
    def __init__(self, confusion_matrix: np.ndarray):
        self.cm = confusion_matrix
        self.num_classes = len(confusion_matrix)
    
    def get_per_class_accuracy(self) -> np.ndarray:
        """Get accuracy per class"""
        return np.diag(self.cm) / np.sum(self.cm, axis=1)
    
    def get_precision_per_class(self) -> np.ndarray:
        """Get precision per class"""
        return np.diag(self.cm) / np.sum(self.cm, axis=0)
    
    def get_recall_per_class(self) -> np.ndarray:
        """Get recall per class"""
        return np.diag(self.cm) / np.sum(self.cm, axis=1)


if __name__ == "__main__":
    # Test metrics
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 1, 0, 1, 2]
    
    calc = MetricsCalculator()
    metrics = calc.calculate_all(y_true, y_pred)
    print(metrics)
