"""
Training Improvements Utility
=============================

This module implements various training improvements to address overfitting and over-prediction:
- Temperature scaling for calibration
- Class weight computation for imbalanced data
- Mixup data augmentation
- Dynamic loss weighting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def temperature_scale(logits: torch.Tensor, temperature: float = 1.5) -> torch.Tensor:
    """
    Apply temperature scaling to make predictions less confident.
    
    Args:
        logits: Raw model logits [batch_size, num_classes]
        temperature: Temperature parameter (>1 makes predictions less confident)
        
    Returns:
        Temperature-scaled logits
    """
    return logits / temperature


def compute_class_weights(labels: np.ndarray, method: str = 'balanced') -> torch.Tensor:
    """
    Compute class weights for imbalanced multi-label data.
    
    Args:
        labels: Ground truth labels [num_samples, num_classes]
        method: Method for weight computation ('balanced', 'inverse_freq')
        
    Returns:
        Class weights tensor [num_classes]
    """
    num_classes = labels.shape[1]
    class_weights = []
    
    for class_idx in range(num_classes):
        class_labels = labels[:, class_idx]
        
        if method == 'balanced':
            # Use sklearn's balanced class weight
            unique_labels = np.unique(class_labels)
            if len(unique_labels) > 1:
                weights = compute_class_weight('balanced', classes=unique_labels, y=class_labels)
                # Get weight for positive class (class 1)
                pos_weight = weights[1] if len(weights) > 1 else 1.0
            else:
                pos_weight = 1.0
        elif method == 'inverse_freq':
            # Use inverse frequency
            pos_count = class_labels.sum()
            neg_count = len(class_labels) - pos_count
            pos_weight = neg_count / max(pos_count, 1)  # Avoid division by zero
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        class_weights.append(pos_weight)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    logger.info(f"Computed class weights using {method}: {class_weights.tolist()}")
    
    return class_weights


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply mixup data augmentation to prevent overfitting.
    
    Args:
        x: Input data [batch_size, ...]
        y: Target labels [batch_size, num_classes]
        alpha: Mixup parameter (0.2 recommended)
        
    Returns:
        Tuple of (mixed_x, y_a, y_b, lambda) where:
        - mixed_x: Mixed input data
        - y_a, y_b: Original target pairs
        - lambda: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Compute mixup loss.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a, y_b: Original target pairs
        lam: Mixing coefficient
        
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class DynamicLossWeighting:
    """
    Dynamic loss weighting strategy to prevent one task from dominating.
    """
    
    def __init__(self, initial_weights: Dict[str, float], adaptation_rate: float = 0.1):
        """
        Initialize dynamic loss weighting.
        
        Args:
            initial_weights: Initial loss weights
            adaptation_rate: Rate of weight adaptation
        """
        self.weights = initial_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.loss_history = {task: [] for task in initial_weights.keys()}
        
    def update_weights(self, current_losses: Dict[str, float]) -> Dict[str, float]:
        """
        Update loss weights based on current loss values.
        
        Args:
            current_losses: Current loss values for each task
            
        Returns:
            Updated loss weights
        """
        # Store current losses
        for task, loss in current_losses.items():
            if task in self.loss_history:
                self.loss_history[task].append(loss)
                # Keep only recent history
                if len(self.loss_history[task]) > 50:
                    self.loss_history[task] = self.loss_history[task][-50:]
        
        # Calculate relative loss ratios
        if all(len(history) > 10 for history in self.loss_history.values()):
            avg_losses = {task: np.mean(history[-10:]) for task, history in self.loss_history.items()}
            total_avg = sum(avg_losses.values())
            
            # Adjust weights inversely to average losses
            for task in self.weights.keys():
                if task in avg_losses and total_avg > 0:
                    target_weight = (total_avg - avg_losses[task]) / total_avg
                    # Smooth adjustment
                    self.weights[task] = (1 - self.adaptation_rate) * self.weights[task] + \
                                       self.adaptation_rate * target_weight
                    
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {task: weight / total_weight for task, weight in self.weights.items()}
        
        return self.weights.copy()
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.weights.copy()


class CalibrationTemperature(nn.Module):
    """
    Learnable temperature parameter for model calibration.
    """
    
    def __init__(self, initial_temperature: float = 1.0):
        super(CalibrationTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply learnable temperature scaling."""
        return logits / self.temperature
    
    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()


class GradualUnfreezing:
    """
    Gradual unfreezing strategy for transfer learning.
    """
    
    def __init__(self, model: nn.Module, unfreeze_schedule: Dict[int, list]):
        """
        Initialize gradual unfreezing.
        
        Args:
            model: Model to apply unfreezing to
            unfreeze_schedule: Dict mapping epoch -> list of layer names to unfreeze
        """
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule
        
    def unfreeze_layers(self, epoch: int):
        """Unfreeze layers according to schedule."""
        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = self.unfreeze_schedule[epoch]
            logger.info(f"Epoch {epoch}: Unfreezing layers {layers_to_unfreeze}")
            
            for name, param in self.model.named_parameters():
                if any(layer_name in name for layer_name in layers_to_unfreeze):
                    param.requires_grad = True
                    logger.debug(f"Unfrozen parameter: {name}")


def create_training_improvements(
    labels: np.ndarray,
    config: Dict
) -> Dict:
    """
    Create training improvements based on configuration.
    
    Args:
        labels: Training labels for class weight computation
        config: Configuration dictionary
        
    Returns:
        Dictionary with improvement utilities
    """
    improvements = {}
    
    # Class weights
    if config.get('use_class_weights', False):
        improvements['class_weights'] = compute_class_weights(labels, method='balanced')
    
    # Temperature scaling
    if config.get('temperature_scaling', 0) > 0:
        improvements['temperature'] = config['temperature_scaling']
    
    # Dynamic loss weighting
    if config.get('dynamic_loss_weighting', False):
        initial_weights = {
            'disease': config.get('disease_loss_weight', 1.0),
            'keyword': config.get('global_keyword_loss_weight', 0.1),
            'region': config.get('region_keyword_loss_weight', 0.05)
        }
        improvements['dynamic_weighting'] = DynamicLossWeighting(initial_weights)
    
    # Mixup parameters
    if config.get('mixup_alpha', 0) > 0:
        improvements['mixup_alpha'] = config['mixup_alpha']
    
    logger.info(f"Created training improvements: {list(improvements.keys())}")
    return improvements 