#!/usr/bin/env python3
"""
Metric Aggregator
================
Utility for aggregating and combining multiple evaluation metric scores
with configurable weights and statistical analysis.

This module provides:
- Weighted score combination
- Statistical aggregation methods
- Confidence interval calculation
- Score normalization and validation
- Batch aggregation support

Author: Medical Report Evaluation System
Version: 1.0
"""

import logging
import statistics
from typing import Dict, List, Optional, Union, Tuple, Any
import math


class MetricAggregator:
    """
    Aggregates multiple evaluation metric scores using various combination methods.
    
    Supports weighted averaging, statistical analysis, and confidence intervals
    for comprehensive evaluation result aggregation.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the metric aggregator.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.default_weights = {
            'bleu': 0.2,
            'rouge_1': 0.15,
            'rouge_2': 0.10,
            'rouge_l': 0.15,
            'meteor': 0.1,
            'bert_score': 0.15,
            'cider': 0.1,
            'medical': 0.05
        }
        self.aggregation_methods = [
            'weighted_average',
            'simple_average',
            'harmonic_mean',
            'geometric_mean',
            'median',
            'max',
            'min'
        ]
        
        self.logger.debug("Initialized MetricAggregator")
    
    def aggregate_scores(self, metric_scores: Dict[str, float], 
                        weights: Optional[Dict[str, float]] = None,
                        method: str = 'weighted_average',
                        normalize: bool = True) -> Dict[str, Any]:
        """
        Aggregate multiple metric scores into a single overall score.
        
        Args:
            metric_scores (Dict[str, float]): Individual metric scores
            weights (Optional[Dict[str, float]]): Weights for each metric
            method (str): Aggregation method to use
            normalize (bool): Whether to normalize scores to [0, 1] range
            
        Returns:
            Dict[str, Any]: Aggregation results including overall score and details
        """
        if not metric_scores:
            return {
                'overall_score': 0.0,
                'method': method,
                'num_metrics': 0,
                'valid_scores': {},
                'invalid_scores': {},
                'aggregation_details': {}
            }
        
        # Filter valid scores
        valid_scores = {}
        invalid_scores = {}
        
        for metric_name, score in metric_scores.items():
            if isinstance(score, (int, float)) and not math.isnan(score) and math.isfinite(score):
                if normalize:
                    # Clamp scores to [0, 1] range
                    normalized_score = max(0.0, min(1.0, float(score)))
                    valid_scores[metric_name] = normalized_score
                else:
                    valid_scores[metric_name] = float(score)
            else:
                invalid_scores[metric_name] = score
        
        if not valid_scores:
            self.logger.warning("No valid scores found for aggregation")
            return {
                'overall_score': 0.0,
                'method': method,
                'num_metrics': len(metric_scores),
                'valid_scores': {},
                'invalid_scores': invalid_scores,
                'aggregation_details': {'error': 'No valid scores'}
            }
        
        # Use provided weights or defaults
        effective_weights = weights or self.default_weights
        
        # Calculate overall score based on method
        if method == 'weighted_average':
            overall_score = self._weighted_average(valid_scores, effective_weights)
        elif method == 'simple_average':
            overall_score = self._simple_average(valid_scores)
        elif method == 'harmonic_mean':
            overall_score = self._harmonic_mean(valid_scores)
        elif method == 'geometric_mean':
            overall_score = self._geometric_mean(valid_scores)
        elif method == 'median':
            overall_score = self._median(valid_scores)
        elif method == 'max':
            overall_score = max(valid_scores.values())
        elif method == 'min':
            overall_score = min(valid_scores.values())
        else:
            self.logger.warning(f"Unknown aggregation method: {method}, using weighted_average")
            overall_score = self._weighted_average(valid_scores, effective_weights)
        
        # Calculate additional statistics
        score_values = list(valid_scores.values())
        aggregation_details = {
            'mean': statistics.mean(score_values),
            'median': statistics.median(score_values),
            'std_dev': statistics.stdev(score_values) if len(score_values) > 1 else 0.0,
            'min_score': min(score_values),
            'max_score': max(score_values),
            'range': max(score_values) - min(score_values),
            'weights_used': {k: effective_weights.get(k, 0.0) for k in valid_scores.keys()}
        }
        
        # Add confidence interval if enough data points
        if len(score_values) > 2:
            try:
                confidence_interval = self._calculate_confidence_interval(score_values)
                aggregation_details['confidence_interval_95'] = confidence_interval
            except Exception as e:
                self.logger.debug(f"Could not calculate confidence interval: {e}")
        
        result = {
            'overall_score': overall_score,
            'method': method,
            'num_metrics': len(metric_scores),
            'num_valid_metrics': len(valid_scores),
            'valid_scores': valid_scores,
            'invalid_scores': invalid_scores,
            'aggregation_details': aggregation_details
        }
        
        self.logger.debug(f"Aggregated {len(valid_scores)} metrics using {method}: {overall_score:.4f}")
        
        return result
    
    def aggregate_batch_results(self, batch_results: List[Dict[str, Any]], 
                               method: str = 'weighted_average') -> Dict[str, Any]:
        """
        Aggregate results from a batch of evaluations.
        
        Args:
            batch_results (List[Dict[str, Any]]): List of individual evaluation results
            method (str): Aggregation method to use
            
        Returns:
            Dict[str, Any]: Batch aggregation results
        """
        if not batch_results:
            return {
                'batch_overall_score': 0.0,
                'batch_size': 0,
                'individual_scores': [],
                'batch_statistics': {},
                'method': method
            }
        
        # Extract overall scores from individual results
        individual_scores = []
        metric_score_lists = {}
        
        for result in batch_results:
            if 'overall_score' in result:
                individual_scores.append(result['overall_score'])
            
            # Collect individual metric scores for analysis
            if 'metrics' in result:
                for metric_name, metric_data in result['metrics'].items():
                    if metric_name not in metric_score_lists:
                        metric_score_lists[metric_name] = []
                    
                    # Extract primary score from metric data
                    if isinstance(metric_data, dict):
                        primary_score = self._extract_primary_score_from_result(metric_name, metric_data)
                        if primary_score is not None:
                            metric_score_lists[metric_name].append(primary_score)
        
        if not individual_scores:
            return {
                'batch_overall_score': 0.0,
                'batch_size': len(batch_results),
                'individual_scores': [],
                'batch_statistics': {'error': 'No valid overall scores found'},
                'method': method
            }
        
        # Calculate batch statistics
        batch_statistics = {
            'mean': statistics.mean(individual_scores),
            'median': statistics.median(individual_scores),
            'std_dev': statistics.stdev(individual_scores) if len(individual_scores) > 1 else 0.0,
            'min_score': min(individual_scores),
            'max_score': max(individual_scores),
            'range': max(individual_scores) - min(individual_scores),
            'count': len(individual_scores)
        }
        
        # Add confidence interval
        if len(individual_scores) > 2:
            try:
                confidence_interval = self._calculate_confidence_interval(individual_scores)
                batch_statistics['confidence_interval_95'] = confidence_interval
            except Exception as e:
                self.logger.debug(f"Could not calculate batch confidence interval: {e}")
        
        # Calculate per-metric batch statistics
        metric_batch_stats = {}
        for metric_name, scores in metric_score_lists.items():
            if scores:
                metric_batch_stats[metric_name] = {
                    'mean': statistics.mean(scores),
                    'median': statistics.median(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
        
        batch_statistics['per_metric'] = metric_batch_stats
        
        # Determine batch overall score based on method
        if method == 'simple_average':
            batch_overall_score = statistics.mean(individual_scores)
        elif method == 'median':
            batch_overall_score = statistics.median(individual_scores)
        elif method == 'harmonic_mean':
            batch_overall_score = statistics.harmonic_mean([max(0.001, s) for s in individual_scores])
        elif method == 'geometric_mean':
            batch_overall_score = statistics.geometric_mean([max(0.001, s) for s in individual_scores])
        else:
            batch_overall_score = statistics.mean(individual_scores)
        
        return {
            'batch_overall_score': batch_overall_score,
            'batch_size': len(batch_results),
            'individual_scores': individual_scores,
            'batch_statistics': batch_statistics,
            'method': method
        }
    
    def _weighted_average(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted average of scores."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_name, score in scores.items():
            weight = weights.get(metric_name, 0.1)  # Default weight
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _simple_average(self, scores: Dict[str, float]) -> float:
        """Calculate simple average of scores."""
        return statistics.mean(scores.values()) if scores else 0.0
    
    def _harmonic_mean(self, scores: Dict[str, float]) -> float:
        """Calculate harmonic mean of scores."""
        # Add small epsilon to avoid division by zero
        safe_scores = [max(0.001, score) for score in scores.values()]
        return statistics.harmonic_mean(safe_scores) if safe_scores else 0.0
    
    def _geometric_mean(self, scores: Dict[str, float]) -> float:
        """Calculate geometric mean of scores."""
        # Add small epsilon to avoid issues with zero values
        safe_scores = [max(0.001, score) for score in scores.values()]
        return statistics.geometric_mean(safe_scores) if safe_scores else 0.0
    
    def _median(self, scores: Dict[str, float]) -> float:
        """Calculate median of scores."""
        return statistics.median(scores.values()) if scores else 0.0
    
    def _calculate_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for scores.
        
        Args:
            scores (List[float]): List of scores
            confidence (float): Confidence level (default: 0.95)
            
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound)
        """
        if len(scores) < 2:
            return (0.0, 0.0)
        
        mean = statistics.mean(scores)
        std_dev = statistics.stdev(scores)
        n = len(scores)
        
        # Use t-distribution for small samples, normal for large samples
        if n < 30:
            # Approximate t-value for 95% confidence
            t_value = 2.0 + (2.0 / n)  # Rough approximation
        else:
            t_value = 1.96  # Normal distribution 95% confidence
        
        margin_of_error = t_value * (std_dev / math.sqrt(n))
        
        return (mean - margin_of_error, mean + margin_of_error)
    
    def _extract_primary_score_from_result(self, metric_name: str, metric_data: Dict) -> Optional[float]:
        """Extract primary score from metric result data."""
        if "error" in metric_data:
            return None
        
        # Define primary score extraction logic for each metric
        if metric_name == "bleu":
            return metric_data.get("bleu_4", metric_data.get("bleu", None))
        elif metric_name == "rouge":
            return metric_data.get("rouge_l", metric_data.get("rouge_1", None))
        elif metric_name == "meteor":
            return metric_data.get("meteor", metric_data.get("score", None))
        elif metric_name == "bert_score":
            return metric_data.get("f1", metric_data.get("bert_score", None))
        elif metric_name == "cider":
            return metric_data.get("cider", metric_data.get("score", None))
        elif metric_name == "medical":
            return metric_data.get("overall_score", metric_data.get("medical_score", None))
        else:
            # Generic fallback
            for field in ["score", "f1", "overall", "primary"]:
                if field in metric_data:
                    return metric_data[field]
        
        return None
    
    def get_supported_methods(self) -> List[str]:
        """Get list of supported aggregation methods."""
        return self.aggregation_methods.copy()
    
    def validate_weights(self, weights: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate metric weights.
        
        Args:
            weights (Dict[str, float]): Weights to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Check for negative weights
        negative_weights = [k for k, v in weights.items() if v < 0]
        if negative_weights:
            errors.append(f"Negative weights found: {negative_weights}")
        
        # Check if weights sum to reasonable value
        total_weight = sum(weights.values())
        if total_weight == 0:
            errors.append("Total weight is zero")
        elif total_weight < 0.1:
            errors.append(f"Total weight is very small: {total_weight}")
        
        # Check for extremely large weights
        large_weights = [k for k, v in weights.items() if v > 10.0]
        if large_weights:
            errors.append(f"Very large weights found: {large_weights}")
        
        return len(errors) == 0, errors


def test_metric_aggregator():
    """Test the MetricAggregator functionality."""
    print("Testing MetricAggregator...")
    
    # Create aggregator
    aggregator = MetricAggregator()
    
    # Test single aggregation
    test_scores = {
        'bleu': 0.45,
        'rouge': 0.62,
        'meteor': 0.51,
        'bert_score': 0.75,
        'medical': 0.68
    }
    
    result = aggregator.aggregate_scores(test_scores)
    print(f"Single aggregation result: {result['overall_score']:.4f}")
    print(f"Method: {result['method']}")
    print(f"Valid metrics: {result['num_valid_metrics']}")
    
    # Test batch aggregation
    batch_results = [
        {'overall_score': 0.65, 'metrics': {'bleu': {'bleu': 0.4}, 'rouge': {'rouge_1': 0.6}}},
        {'overall_score': 0.72, 'metrics': {'bleu': {'bleu': 0.5}, 'rouge': {'rouge_1': 0.7}}},
        {'overall_score': 0.58, 'metrics': {'bleu': {'bleu': 0.3}, 'rouge': {'rouge_1': 0.5}}}
    ]
    
    batch_result = aggregator.aggregate_batch_results(batch_results)
    print(f"Batch aggregation result: {batch_result['batch_overall_score']:.4f}")
    print(f"Batch size: {batch_result['batch_size']}")
    
    # Test different methods
    for method in ['simple_average', 'harmonic_mean', 'geometric_mean', 'median']:
        result = aggregator.aggregate_scores(test_scores, method=method)
        print(f"{method}: {result['overall_score']:.4f}")
    
    print("MetricAggregator test completed!")
    return True


if __name__ == "__main__":
    test_metric_aggregator() 