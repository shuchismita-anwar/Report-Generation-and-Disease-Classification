"""
Threshold Tuning Utility for Hybrid Model
==========================================

This module provides functionality to find optimal classification thresholds
per disease based on validation set performance.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def find_optimal_thresholds(
    predictions: np.ndarray, 
    labels: np.ndarray, 
    metric: str = 'f1',
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    num_thresholds: int = 40
) -> List[float]:
    """
    Find optimal thresholds per disease using validation set.
    
    Args:
        predictions: Model predictions [num_samples, num_diseases]
        labels: Ground truth labels [num_samples, num_diseases]
        metric: Metric to optimize ('f1', 'precision', 'recall')
        threshold_range: Range of thresholds to search
        num_thresholds: Number of threshold values to test
        
    Returns:
        List of optimal thresholds for each disease
    """
    logger.info(f"Finding optimal thresholds using {metric} metric")
    logger.info(f"Threshold range: {threshold_range}, steps: {num_thresholds}")
    
    thresholds_to_test = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    num_diseases = predictions.shape[1]
    optimal_thresholds = []
    
    for disease_idx in range(num_diseases):
        best_score = 0.0
        best_threshold = 0.5  # Default fallback
        
        disease_predictions = predictions[:, disease_idx]
        disease_labels = labels[:, disease_idx]
        
        # Skip if no positive examples for this disease
        if disease_labels.sum() == 0:
            logger.warning(f"Disease {disease_idx}: No positive examples, using default threshold 0.5")
            optimal_thresholds.append(0.5)
            continue
            
        for threshold in thresholds_to_test:
            binary_preds = (disease_predictions > threshold).astype(int)
            
            # Calculate metric
            if metric == 'f1':
                score = f1_score(disease_labels, binary_preds, zero_division=0)
            elif metric == 'precision':
                score = precision_score(disease_labels, binary_preds, zero_division=0)
            elif metric == 'recall':
                score = recall_score(disease_labels, binary_preds, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
        logger.debug(f"Disease {disease_idx}: threshold={best_threshold:.3f}, {metric}={best_score:.3f}")
    
    logger.info(f"Optimal thresholds found: {[f'{t:.3f}' for t in optimal_thresholds]}")
    return optimal_thresholds

def evaluate_with_thresholds(
    predictions: np.ndarray,
    labels: np.ndarray, 
    thresholds: List[float]
) -> Dict[str, float]:
    """
    Evaluate predictions using specified thresholds.
    
    Args:
        predictions: Model predictions [num_samples, num_diseases]
        labels: Ground truth labels [num_samples, num_diseases]
        thresholds: Threshold for each disease
        
    Returns:
        Dictionary with evaluation metrics
    """
    num_diseases = predictions.shape[1]
    assert len(thresholds) == num_diseases, f"Expected {num_diseases} thresholds, got {len(thresholds)}"
    
    # Apply thresholds
    binary_predictions = np.zeros_like(predictions)
    for disease_idx, threshold in enumerate(thresholds):
        binary_predictions[:, disease_idx] = (predictions[:, disease_idx] > threshold).astype(int)
    
    # Calculate metrics
    # Micro-averaged metrics (treats each prediction independently)
    micro_f1 = f1_score(labels.flatten(), binary_predictions.flatten(), zero_division=0)
    micro_precision = precision_score(labels.flatten(), binary_predictions.flatten(), zero_division=0)
    micro_recall = recall_score(labels.flatten(), binary_predictions.flatten(), zero_division=0)
    
    # Macro-averaged metrics (average of per-disease metrics)
    macro_f1 = f1_score(labels, binary_predictions, average='macro', zero_division=0)
    macro_precision = precision_score(labels, binary_predictions, average='macro', zero_division=0)
    macro_recall = recall_score(labels, binary_predictions, average='macro', zero_division=0)
    
    # Per-disease metrics
    per_disease_f1 = f1_score(labels, binary_predictions, average=None, zero_division=0)
    
    # Exact match accuracy (all diseases must be correct)
    exact_match = (binary_predictions == labels).all(axis=1).mean()
    
    # Hamming accuracy (fraction of correct labels)
    hamming_accuracy = (binary_predictions == labels).mean()
    
    results = {
        'micro_f1': micro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'exact_match_accuracy': exact_match,
        'hamming_accuracy': hamming_accuracy,
        'per_disease_f1': per_disease_f1.tolist(),
        'thresholds_used': thresholds
    }
    
    return results

def find_and_evaluate_thresholds(
    val_predictions: np.ndarray,
    val_labels: np.ndarray,
    test_predictions: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    metric: str = 'f1'
) -> Dict[str, any]:
    """
    Complete pipeline: find optimal thresholds on validation set and evaluate on test set.
    
    Args:
        val_predictions: Validation predictions [num_val_samples, num_diseases]
        val_labels: Validation labels [num_val_samples, num_diseases]
        test_predictions: Test predictions [num_test_samples, num_diseases] (optional)
        test_labels: Test labels [num_test_samples, num_diseases] (optional)
        metric: Metric to optimize thresholds for
        
    Returns:
        Dictionary with optimal thresholds and evaluation results
    """
    logger.info("Starting threshold optimization pipeline")
    
    # Find optimal thresholds on validation set
    optimal_thresholds = find_optimal_thresholds(val_predictions, val_labels, metric=metric)
    
    # Evaluate on validation set with optimal thresholds
    val_results = evaluate_with_thresholds(val_predictions, val_labels, optimal_thresholds)
    val_results = {f'val_{k}': v for k, v in val_results.items()}
    
    results = {
        'optimal_thresholds': optimal_thresholds,
        'optimization_metric': metric,
        **val_results
    }
    
    # Evaluate on test set if provided
    if test_predictions is not None and test_labels is not None:
        test_results = evaluate_with_thresholds(test_predictions, test_labels, optimal_thresholds)
        test_results = {f'test_{k}': v for k, v in test_results.items()}
        results.update(test_results)
        
        logger.info(f"Test results with optimal thresholds:")
        logger.info(f"  - F1: {test_results['test_micro_f1']:.4f}")
        logger.info(f"  - Precision: {test_results['test_micro_precision']:.4f}")
        logger.info(f"  - Recall: {test_results['test_micro_recall']:.4f}")
    
    return results

class ThresholdOptimizer:
    """
    Stateful threshold optimizer that can be used during training.
    """
    
    def __init__(self, metric: str = 'f1', num_diseases: int = 8):
        self.metric = metric
        self.num_diseases = num_diseases
        self.optimal_thresholds = [0.5] * num_diseases  # Initialize with default
        self.best_score = 0.0
        
    def update(self, predictions: np.ndarray, labels: np.ndarray) -> bool:
        """
        Update optimal thresholds if validation performance improves.
        
        Returns:
            True if thresholds were updated, False otherwise
        """
        new_thresholds = find_optimal_thresholds(predictions, labels, metric=self.metric)
        
        # Evaluate with new thresholds
        results = evaluate_with_thresholds(predictions, labels, new_thresholds)
        new_score = results['micro_f1']  # Use micro F1 as overall score
        
        if new_score > self.best_score:
            self.optimal_thresholds = new_thresholds
            self.best_score = new_score
            logger.info(f"Updated optimal thresholds: {self.metric} improved to {new_score:.4f}")
            return True
        
        return False
    
    def get_thresholds(self) -> List[float]:
        """Get current optimal thresholds."""
        return self.optimal_thresholds.copy()
    
    def apply_thresholds(self, predictions: np.ndarray) -> np.ndarray:
        """Apply optimal thresholds to predictions."""
        binary_predictions = np.zeros_like(predictions)
        for disease_idx, threshold in enumerate(self.optimal_thresholds):
            binary_predictions[:, disease_idx] = (predictions[:, disease_idx] > threshold).astype(int)
        return binary_predictions


def create_threshold_optimizer(metric: str = 'f1', num_diseases: int = 8) -> ThresholdOptimizer:
    """Factory function to create threshold optimizer."""
    return ThresholdOptimizer(metric=metric, num_diseases=num_diseases) 