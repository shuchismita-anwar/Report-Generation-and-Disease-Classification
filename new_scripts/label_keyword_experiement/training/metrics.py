"""
Hybrid Metrics for Multi-Task Training
=====================================

This module provides comprehensive evaluation metrics for the hybrid model training,
supporting both disease classification and keyword prediction tasks.

The HybridMetrics class computes:
- Disease classification metrics (F1, precision, recall, AUC)
- Keyword prediction metrics (precision@k, F1, semantic similarity)
- Region-specific keyword metrics
- Combined system performance metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    accuracy_score, hamming_loss, jaccard_score
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# PyTorch availability check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Metrics computation may be limited.")

@dataclass
class MetricsConfig:
    """Configuration for metrics computation."""
    
    # Disease classification thresholds
    disease_threshold_range: Tuple[float, float] = (0.38, 0.78)
    disease_threshold_step: float = 0.02
    disease_precision_bias: float = 0.15
    disease_min_precision: float = 0.3
    
    # Keyword prediction settings
    keyword_threshold: float = 0.5
    precision_at_k_values: List[int] = None
    keyword_min_frequency: int = 1
    
    # Region-specific settings
    num_regions: int = 17
    region_weight_strategy: str = "uniform"  # "uniform", "frequency", "importance"
    
    # Semantic similarity settings
    use_semantic_similarity: bool = False
    semantic_similarity_threshold: float = 0.7
    
    def __post_init__(self):
        if self.precision_at_k_values is None:
            self.precision_at_k_values = [5, 10, 20]


class HybridMetrics:
    """
    Comprehensive metrics computation for hybrid disease classification + keyword prediction.
    
    Computes metrics for:
    1. Disease classification (existing approach with optimal thresholds)
    2. Global keyword prediction (precision@k, F1, recall)
    3. Region-specific keyword prediction (per-region and aggregated)
    4. Combined system performance (consistency, alignment)
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self.disease_conditions = None  # Will be set when first used
        self.keyword_vocabulary = None  # Will be set when first used
        
        logger.info(f"HybridMetrics initialized:")
        logger.info(f"  - Disease threshold range: {self.config.disease_threshold_range}")
        logger.info(f"  - Keyword threshold: {self.config.keyword_threshold}")
        logger.info(f"  - Precision@k values: {self.config.precision_at_k_values}")
        logger.info(f"  - Number of regions: {self.config.num_regions}")
    
    def compute(
        self,
        predictions: Dict[str, Union[torch.Tensor, np.ndarray]],
        targets: Dict[str, Union[torch.Tensor, np.ndarray]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics for multi-task predictions.
        
        Args:
            predictions: Dictionary containing model predictions
                - 'disease_logits': [batch, num_diseases] - Disease classification logits
                - 'global_keywords': [batch, vocab_size] - Global keyword logits
                - 'region_keywords': [batch, num_regions, vocab_size] - Region keyword logits
            targets: Dictionary containing ground truth labels
                - 'disease_labels': [batch, num_diseases] - Disease binary labels
                - 'global_keywords': [batch, vocab_size] - Global keyword binary labels
                - 'region_keywords': [batch, num_regions, vocab_size] - Region keyword labels
                - 'region_mask': [batch, num_regions] - Region availability mask
                
        Returns:
            Dictionary of computed metrics
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - returning dummy metrics")
            return self._get_dummy_metrics()
        
        # Convert tensors to numpy arrays
        pred_np = self._convert_to_numpy(predictions)
        target_np = self._convert_to_numpy(targets)
        
        metrics = {}
        
        # 1. Disease classification metrics
        if 'disease_logits' in pred_np and 'disease_labels' in target_np:
            disease_metrics = self._compute_disease_metrics(
                pred_np['disease_logits'], 
                target_np['disease_labels']
            )
            metrics.update(disease_metrics)
        
        # 2. Global keyword metrics
        if 'global_keywords' in pred_np and 'global_keywords' in target_np:
            global_kw_metrics = self._compute_global_keyword_metrics(
                pred_np['global_keywords'],
                target_np['global_keywords']
            )
            metrics.update(global_kw_metrics)
        
        # 3. Region-specific keyword metrics
        if ('region_keywords' in pred_np and 'region_keywords' in target_np):
            region_mask = target_np.get('region_mask', None)
            region_kw_metrics = self._compute_region_keyword_metrics(
                pred_np['region_keywords'],
                target_np['region_keywords'],
                region_mask
            )
            metrics.update(region_kw_metrics)
        
        # 4. Combined system metrics
        combined_metrics = self._compute_combined_metrics(pred_np, target_np)
        metrics.update(combined_metrics)
        
        # 5. Log key metrics
        self._log_key_metrics(metrics)
        
        return metrics
    
    def compute_metrics(
        self,
        predictions: Dict[str, Union[torch.Tensor, np.ndarray]],
        targets: Dict[str, Union[torch.Tensor, np.ndarray]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Alias for compute method for compatibility.
        """
        return self.compute(predictions, targets, **kwargs)
    
    def _convert_to_numpy(self, data: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Convert tensors to numpy arrays."""
        result = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                result[key] = value
            else:
                result[key] = np.array(value)
        return result
    
    def _compute_disease_metrics(
        self, 
        logits: np.ndarray, 
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute disease classification metrics with optimal thresholds."""
        metrics = {}
        
        # Apply sigmoid to get probabilities
        probs = 1 / (1 + np.exp(-logits))
        
        # Find optimal thresholds for each disease
        optimal_thresholds = []
        num_diseases = labels.shape[1]
        
        for i in range(num_diseases):
            best_threshold = 0.5
            best_score = 0.0
            
            threshold_start, threshold_end = self.config.disease_threshold_range
            for threshold in np.arange(threshold_start, threshold_end, self.config.disease_threshold_step):
                temp_preds = (probs[:, i] > threshold).astype(int)
                temp_f1 = f1_score(labels[:, i], temp_preds, zero_division=0)
                temp_precision = precision_score(labels[:, i], temp_preds, zero_division=0)
                
                # Apply precision bias
                if temp_precision > self.config.disease_min_precision:
                    adjusted_score = (1 - self.config.disease_precision_bias) * temp_f1 + \
                                   self.config.disease_precision_bias * temp_precision
                else:
                    adjusted_score = temp_f1
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_threshold = threshold
            
            optimal_thresholds.append(best_threshold)
        
        # Generate predictions with optimal thresholds
        preds = np.zeros_like(probs)
        for i in range(num_diseases):
            preds[:, i] = (probs[:, i] > optimal_thresholds[i]).astype(int)
        
        # Also compute with standard 0.5 threshold for comparison
        preds_05 = (probs > 0.5).astype(int)
        
        # Compute metrics
        try:
            # Micro-averaged metrics (primary)
            metrics['disease_micro_f1'] = f1_score(labels.flatten(), preds.flatten(), zero_division=0)
            metrics['disease_micro_precision'] = precision_score(labels.flatten(), preds.flatten(), zero_division=0)
            metrics['disease_micro_recall'] = recall_score(labels.flatten(), preds.flatten(), zero_division=0)
            
            # Macro-averaged metrics
            metrics['disease_macro_f1'] = f1_score(labels, preds, average="macro", zero_division=0)
            metrics['disease_macro_precision'] = precision_score(labels, preds, average="macro", zero_division=0)
            metrics['disease_macro_recall'] = recall_score(labels, preds, average="macro", zero_division=0)
            
            # Accuracy metrics
            metrics['disease_exact_match_accuracy'] = np.mean(np.all(labels == preds, axis=1))
            metrics['disease_hamming_accuracy'] = np.mean(labels == preds)
            
            # AUC scores
            auc_scores = []
            for i in range(num_diseases):
                if len(np.unique(labels[:, i])) > 1:
                    auc_scores.append(roc_auc_score(labels[:, i], probs[:, i]))
            metrics['disease_mean_auc'] = np.mean(auc_scores) if auc_scores else 0.0
            
            # Comparison with 0.5 threshold
            metrics['disease_micro_f1_05'] = f1_score(labels.flatten(), preds_05.flatten(), zero_division=0)
            
            # Store optimal thresholds for logging
            metrics['_optimal_thresholds'] = optimal_thresholds
            
        except Exception as e:
            logger.error(f"Error computing disease metrics: {e}")
            metrics.update({
                'disease_micro_f1': 0.0, 'disease_micro_precision': 0.0, 'disease_micro_recall': 0.0,
                'disease_macro_f1': 0.0, 'disease_macro_precision': 0.0, 'disease_macro_recall': 0.0,
                'disease_exact_match_accuracy': 0.0, 'disease_hamming_accuracy': 0.0,
                'disease_mean_auc': 0.0, 'disease_micro_f1_05': 0.0
            })
        
        return metrics
    
    def _compute_global_keyword_metrics(
        self, 
        logits: np.ndarray, 
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute global keyword prediction metrics."""
        metrics = {}
        
        # Apply sigmoid to get probabilities
        probs = 1 / (1 + np.exp(-logits))
        
        # Binary predictions with threshold
        preds = (probs > self.config.keyword_threshold).astype(int)
        
        try:
            # Basic classification metrics
            metrics['keyword_global_f1'] = f1_score(labels.flatten(), preds.flatten(), zero_division=0)
            metrics['keyword_global_precision'] = precision_score(labels.flatten(), preds.flatten(), zero_division=0)
            metrics['keyword_global_recall'] = recall_score(labels.flatten(), preds.flatten(), zero_division=0)
            
            # Hamming metrics (per-sample accuracy)
            metrics['keyword_global_hamming_accuracy'] = np.mean(labels == preds)
            metrics['keyword_global_hamming_loss'] = hamming_loss(labels, preds)
            
            # Jaccard similarity (intersection over union)
            metrics['keyword_global_jaccard'] = jaccard_score(labels, preds, average='samples', zero_division=0)
            
            # Precision@k metrics
            for k in self.config.precision_at_k_values:
                if k <= logits.shape[1]:
                    precision_at_k = self._compute_precision_at_k(probs, labels, k)
                    metrics[f'keyword_global_precision_at_{k}'] = precision_at_k
            
            # Coverage metrics (how many keywords predicted per sample)
            avg_keywords_predicted = np.mean(np.sum(preds, axis=1))
            avg_keywords_true = np.mean(np.sum(labels, axis=1))
            metrics['keyword_global_avg_predicted'] = avg_keywords_predicted
            metrics['keyword_global_avg_true'] = avg_keywords_true
            metrics['keyword_global_coverage_ratio'] = avg_keywords_predicted / (avg_keywords_true + 1e-8)
            
        except Exception as e:
            logger.error(f"Error computing global keyword metrics: {e}")
            metrics.update({
                'keyword_global_f1': 0.0, 'keyword_global_precision': 0.0, 'keyword_global_recall': 0.0,
                'keyword_global_hamming_accuracy': 0.0, 'keyword_global_hamming_loss': 1.0,
                'keyword_global_jaccard': 0.0, 'keyword_global_avg_predicted': 0.0,
                'keyword_global_avg_true': 0.0, 'keyword_global_coverage_ratio': 0.0
            })
            
            for k in self.config.precision_at_k_values:
                metrics[f'keyword_global_precision_at_{k}'] = 0.0
        
        return metrics
    
    def _compute_region_keyword_metrics(
        self, 
        logits: np.ndarray, 
        labels: np.ndarray,
        region_mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute region-specific keyword prediction metrics."""
        metrics = {}
        
        # Apply sigmoid to get probabilities
        probs = 1 / (1 + np.exp(-logits))
        
        # Binary predictions with threshold
        preds = (probs > self.config.keyword_threshold).astype(int)
        
        # If no region mask provided, assume all regions are available
        if region_mask is None:
            region_mask = np.ones((logits.shape[0], logits.shape[1]), dtype=bool)
        
        try:
            # Aggregate metrics across all regions (masked)
            masked_preds = preds * region_mask[:, :, np.newaxis]
            masked_labels = labels * region_mask[:, :, np.newaxis]
            
            # Flatten for overall region metrics
            flat_preds = masked_preds.flatten()
            flat_labels = masked_labels.flatten()
            
            # Remove zero entries (from masking)
            valid_mask = (flat_labels != 0) | (flat_preds != 0)
            if np.any(valid_mask):
                valid_preds = flat_preds[valid_mask]
                valid_labels = flat_labels[valid_mask]
                
                metrics['keyword_region_f1'] = f1_score(valid_labels, valid_preds, zero_division=0)
                metrics['keyword_region_precision'] = precision_score(valid_labels, valid_preds, zero_division=0)
                metrics['keyword_region_recall'] = recall_score(valid_labels, valid_preds, zero_division=0)
            else:
                metrics['keyword_region_f1'] = 0.0
                metrics['keyword_region_precision'] = 0.0
                metrics['keyword_region_recall'] = 0.0
            
            # Per-region metrics (average across available regions)
            region_f1_scores = []
            region_precision_scores = []
            region_recall_scores = []
            
            for region_idx in range(logits.shape[1]):
                # Get samples where this region is available
                region_available = region_mask[:, region_idx]
                
                if np.any(region_available):
                    region_preds = preds[region_available, region_idx, :].flatten()
                    region_labels = labels[region_available, region_idx, :].flatten()
                    
                    if len(region_preds) > 0:
                        region_f1 = f1_score(region_labels, region_preds, zero_division=0)
                        region_precision = precision_score(region_labels, region_preds, zero_division=0)
                        region_recall = recall_score(region_labels, region_preds, zero_division=0)
                        
                        region_f1_scores.append(region_f1)
                        region_precision_scores.append(region_precision)
                        region_recall_scores.append(region_recall)
            
            # Average per-region metrics
            metrics['keyword_region_avg_f1'] = np.mean(region_f1_scores) if region_f1_scores else 0.0
            metrics['keyword_region_avg_precision'] = np.mean(region_precision_scores) if region_precision_scores else 0.0
            metrics['keyword_region_avg_recall'] = np.mean(region_recall_scores) if region_recall_scores else 0.0
            
            # Region availability statistics
            metrics['keyword_region_avg_available'] = np.mean(np.sum(region_mask, axis=1))
            metrics['keyword_region_availability_ratio'] = np.mean(region_mask)
            
            # Precision@k for regions (using aggregated probabilities)
            for k in self.config.precision_at_k_values:
                if k <= logits.shape[2]:  # vocab_size dimension
                    # Average precision@k across available regions
                    region_precision_at_k = []
                    for region_idx in range(logits.shape[1]):
                        region_available = region_mask[:, region_idx]
                        if np.any(region_available):
                            region_probs = probs[region_available, region_idx, :]
                            region_labels_k = labels[region_available, region_idx, :]
                            if len(region_probs) > 0:
                                precision_k = self._compute_precision_at_k(region_probs, region_labels_k, k)
                                region_precision_at_k.append(precision_k)
                    
                    metrics[f'keyword_region_precision_at_{k}'] = np.mean(region_precision_at_k) if region_precision_at_k else 0.0
            
        except Exception as e:
            logger.error(f"Error computing region keyword metrics: {e}")
            metrics.update({
                'keyword_region_f1': 0.0, 'keyword_region_precision': 0.0, 'keyword_region_recall': 0.0,
                'keyword_region_avg_f1': 0.0, 'keyword_region_avg_precision': 0.0, 'keyword_region_avg_recall': 0.0,
                'keyword_region_avg_available': 0.0, 'keyword_region_availability_ratio': 0.0
            })
            
            for k in self.config.precision_at_k_values:
                metrics[f'keyword_region_precision_at_{k}'] = 0.0
        
        return metrics
    
    def _compute_combined_metrics(
        self, 
        predictions: Dict[str, np.ndarray], 
        targets: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Compute combined system performance metrics."""
        metrics = {}
        
        try:
            # Overall system performance (if both disease and keyword predictions available)
            if ('disease_logits' in predictions and 'global_keywords' in predictions and
                'disease_labels' in targets and 'global_keywords' in targets):
                
                # Disease predictions
                disease_probs = 1 / (1 + np.exp(-predictions['disease_logits']))
                disease_preds = (disease_probs > 0.5).astype(int)
                
                # Keyword predictions
                keyword_probs = 1 / (1 + np.exp(-predictions['global_keywords']))
                keyword_preds = (keyword_probs > self.config.keyword_threshold).astype(int)
                
                # Combined accuracy (both disease and keywords correct)
                disease_correct = np.all(targets['disease_labels'] == disease_preds, axis=1)
                keyword_correct = np.all(targets['global_keywords'] == keyword_preds, axis=1)
                combined_correct = disease_correct & keyword_correct
                
                metrics['combined_exact_match_accuracy'] = np.mean(combined_correct)
                metrics['combined_disease_accuracy'] = np.mean(disease_correct)
                metrics['combined_keyword_accuracy'] = np.mean(keyword_correct)
                
                # Partial accuracy (at least one task correct)
                partial_correct = disease_correct | keyword_correct
                metrics['combined_partial_accuracy'] = np.mean(partial_correct)
                
                # Task consistency (correlation between disease and keyword performance)
                disease_sample_f1 = []
                keyword_sample_f1 = []
                
                for i in range(len(disease_preds)):
                    # Per-sample F1 scores
                    d_f1 = f1_score(targets['disease_labels'][i], disease_preds[i], zero_division=0)
                    k_f1 = f1_score(targets['global_keywords'][i], keyword_preds[i], zero_division=0)
                    disease_sample_f1.append(d_f1)
                    keyword_sample_f1.append(k_f1)
                
                # Correlation between disease and keyword performance
                if len(disease_sample_f1) > 1:
                    correlation = np.corrcoef(disease_sample_f1, keyword_sample_f1)[0, 1]
                    metrics['combined_task_correlation'] = correlation if not np.isnan(correlation) else 0.0
                else:
                    metrics['combined_task_correlation'] = 0.0
            
            # Multi-task loss components (if available)
            if hasattr(self, '_last_loss_components'):
                for component, value in self._last_loss_components.items():
                    metrics[f'loss_{component}'] = value
            
        except Exception as e:
            logger.error(f"Error computing combined metrics: {e}")
            metrics.update({
                'combined_exact_match_accuracy': 0.0, 'combined_disease_accuracy': 0.0,
                'combined_keyword_accuracy': 0.0, 'combined_partial_accuracy': 0.0,
                'combined_task_correlation': 0.0
            })
        
        return metrics
    
    def _compute_precision_at_k(self, probs: np.ndarray, labels: np.ndarray, k: int) -> float:
        """Compute precision@k for keyword predictions."""
        if k <= 0 or k > probs.shape[1]:
            return 0.0
        
        precision_scores = []
        
        for i in range(len(probs)):
            # Get top-k predictions
            top_k_indices = np.argsort(probs[i])[-k:]
            
            # Check how many of top-k are correct
            correct_in_top_k = np.sum(labels[i][top_k_indices])
            precision_at_k = correct_in_top_k / k
            precision_scores.append(precision_at_k)
        
        return np.mean(precision_scores)
    
    def _log_key_metrics(self, metrics: Dict[str, float]):
        """Log key metrics for monitoring."""
        key_metrics = [
            'disease_micro_f1', 'disease_micro_precision', 'disease_micro_recall',
            'keyword_global_f1', 'keyword_global_precision_at_5', 'keyword_global_precision_at_10',
            'keyword_region_f1', 'combined_exact_match_accuracy'
        ]
        
        logger.info("Key Metrics Summary:")
        for metric in key_metrics:
            if metric in metrics:
                logger.info(f"  {metric}: {metrics[metric]:.4f}")
        
        # Special logging for disease metrics comparison
        if 'disease_micro_f1' in metrics and 'disease_micro_f1_05' in metrics:
            f1_optimal = metrics['disease_micro_f1']
            f1_05 = metrics['disease_micro_f1_05']
            logger.info(f"Disease F1 - Optimal: {f1_optimal:.4f} vs 0.5 threshold: {f1_05:.4f}")
            logger.info(f"Target F1: 0.6242, Current: {f1_optimal:.4f} (Gap: {0.6242-f1_optimal:+.4f})")
    
    def _get_dummy_metrics(self) -> Dict[str, float]:
        """Return dummy metrics when PyTorch is not available."""
        dummy_metrics = {
            'disease_micro_f1': 0.0, 'disease_micro_precision': 0.0, 'disease_micro_recall': 0.0,
            'keyword_global_f1': 0.0, 'keyword_global_precision': 0.0, 'keyword_global_recall': 0.0,
            'keyword_region_f1': 0.0, 'combined_exact_match_accuracy': 0.0
        }
        
        for k in self.config.precision_at_k_values:
            dummy_metrics[f'keyword_global_precision_at_{k}'] = 0.0
            dummy_metrics[f'keyword_region_precision_at_{k}'] = 0.0
        
        return dummy_metrics
    
    def set_loss_components(self, loss_components: Dict[str, float]):
        """Set loss components for inclusion in metrics."""
        self._last_loss_components = loss_components


def create_hybrid_metrics(
    disease_conditions: Optional[List[str]] = None,
    keyword_vocabulary: Optional[List[str]] = None,
    config: Optional[MetricsConfig] = None
) -> HybridMetrics:
    """
    Factory function to create HybridMetrics with proper configuration.
    
    Args:
        disease_conditions: List of disease condition names
        keyword_vocabulary: List of keyword vocabulary terms
        config: Custom metrics configuration
        
    Returns:
        Configured HybridMetrics instance
    """
    if config is None:
        config = MetricsConfig()
    
    metrics = HybridMetrics(config)
    metrics.disease_conditions = disease_conditions
    metrics.keyword_vocabulary = keyword_vocabulary
    
    return metrics


def create_dummy_predictions_and_targets(
    batch_size: int = 4,
    num_diseases: int = 8,
    vocab_size: int = 35,
    num_regions: int = 17
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Create dummy predictions and targets for testing metrics."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for creating dummy data")
    
    # Create dummy predictions (logits)
    predictions = {
        'disease_logits': np.random.randn(batch_size, num_diseases),
        'global_keywords': np.random.randn(batch_size, vocab_size),
        'region_keywords': np.random.randn(batch_size, num_regions, vocab_size)
    }
    
    # Create dummy targets (binary labels)
    targets = {
        'disease_labels': np.random.randint(0, 2, (batch_size, num_diseases)).astype(float),
        'global_keywords': np.random.randint(0, 2, (batch_size, vocab_size)).astype(float),
        'region_keywords': np.random.randint(0, 2, (batch_size, num_regions, vocab_size)).astype(float),
        'region_mask': np.random.randint(0, 2, (batch_size, num_regions)).astype(bool)
    }
    
    # Ensure at least some positive labels
    targets['disease_labels'][0, :3] = 1.0
    targets['global_keywords'][0, :5] = 1.0
    targets['region_keywords'][0, :3, :5] = 1.0
    
    return predictions, targets


def test_hybrid_metrics():
    """Test function for HybridMetrics."""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available - skipping metrics test")
        return
    
    print("=== Testing HybridMetrics ===")
    
    # Create test data
    predictions, targets = create_dummy_predictions_and_targets()
    
    print(f"Created dummy data:")
    print(f"  - Batch size: {predictions['disease_logits'].shape[0]}")
    print(f"  - Diseases: {predictions['disease_logits'].shape[1]}")
    print(f"  - Vocabulary size: {predictions['global_keywords'].shape[1]}")
    print(f"  - Regions: {predictions['region_keywords'].shape[1]}")
    
    # Create metrics computer
    config = MetricsConfig(precision_at_k_values=[3, 5, 10])
    metrics_computer = HybridMetrics(config)
    
    # Compute metrics
    metrics = metrics_computer.compute(predictions, targets)
    
    print(f"\nComputed {len(metrics)} metrics:")
    
    # Group metrics by category
    disease_metrics = {k: v for k, v in metrics.items() if k.startswith('disease_')}
    keyword_global_metrics = {k: v for k, v in metrics.items() if k.startswith('keyword_global_')}
    keyword_region_metrics = {k: v for k, v in metrics.items() if k.startswith('keyword_region_')}
    combined_metrics = {k: v for k, v in metrics.items() if k.startswith('combined_')}
    
    print(f"\nDisease Metrics ({len(disease_metrics)}):")
    for k, v in disease_metrics.items():
        if not k.startswith('_'):  # Skip internal metrics
            print(f"  {k}: {v:.4f}")
    
    print(f"\nGlobal Keyword Metrics ({len(keyword_global_metrics)}):")
    for k, v in keyword_global_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nRegion Keyword Metrics ({len(keyword_region_metrics)}):")
    for k, v in keyword_region_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nCombined Metrics ({len(combined_metrics)}):")
    for k, v in combined_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\n✅ HybridMetrics test completed successfully!")
    return metrics


if __name__ == "__main__":
    test_hybrid_metrics() 