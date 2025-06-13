#!/usr/bin/env python3
"""
Base Metric Class
================
Abstract base class for all evaluation metrics in the medical report evaluation system.

This module provides:
- Standardized interface for all metrics
- Common functionality and utilities
- Result formatting and validation
- Error handling patterns

Author: Medical Report Evaluation System
Version: 1.0
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import time
from datetime import datetime


class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics.
    
    Provides standardized interface and common functionality for
    medical report evaluation metrics including BLEU, ROUGE, METEOR,
    BERTScore, and medical-specific metrics.
    """
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the base metric.
        
        Args:
            name (str): Name of the metric
            logger (Optional[logging.Logger]): Logger instance
        """
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.version = "1.0"
        self.description = ""
        self.metric_type = "similarity"  # similarity, accuracy, clinical, etc.
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0
        self.last_calculation_time = 0.0
        
        # Configuration
        self.config = {}
        self.is_initialized = False
        
        self.logger.debug(f"Initialized {self.name} metric")
    
    @abstractmethod
    def calculate(self, reference: str, candidate: str, **kwargs) -> Dict[str, float]:
        """
        Calculate the metric score between reference and candidate texts.
        
        Args:
            reference (str): Ground truth reference text
            candidate (str): Generated candidate text to evaluate
            **kwargs: Additional metric-specific parameters
            
        Returns:
            Dict[str, float]: Dictionary containing metric scores
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the metric name.
        
        Returns:
            str: Metric name
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get the metric description.
        
        Returns:
            str: Detailed description of what the metric measures
        """
        pass
    
    def get_version(self) -> str:
        """
        Get the metric version.
        
        Returns:
            str: Metric version
        """
        return self.version
    
    def get_metric_type(self) -> str:
        """
        Get the metric type category.
        
        Returns:
            str: Metric type (similarity, accuracy, clinical, etc.)
        """
        return self.metric_type
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the metric with parameters.
        
        Args:
            config (Dict[str, Any]): Configuration parameters
        """
        self.config.update(config)
        self.logger.debug(f"Configured {self.name} with parameters: {list(config.keys())}")
    
    def initialize(self) -> None:
        """
        Initialize the metric (load models, prepare resources, etc.).
        Override in subclasses if needed.
        """
        self.is_initialized = True
        self.logger.debug(f"{self.name} metric initialized")
    
    def calculate_with_timing(self, reference: str, candidate: str, **kwargs) -> Tuple[Dict[str, float], float]:
        """
        Calculate metric with performance timing.
        
        Args:
            reference (str): Ground truth reference text
            candidate (str): Generated candidate text to evaluate
            **kwargs: Additional metric-specific parameters
            
        Returns:
            Tuple[Dict[str, float], float]: (metric_scores, calculation_time)
        """
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            scores = self.calculate(reference, candidate, **kwargs)
            calculation_time = time.time() - start_time
            
            # Update performance tracking
            self.calculation_count += 1
            self.total_calculation_time += calculation_time
            self.last_calculation_time = calculation_time
            
            self.logger.debug(f"{self.name} calculation completed in {calculation_time:.4f}s")
            
            return scores, calculation_time
            
        except Exception as e:
            calculation_time = time.time() - start_time
            self.logger.error(f"Error calculating {self.name}: {e}")
            raise
    
    def calculate_batch(self, reference_list: List[str], candidate_list: List[str], 
                       **kwargs) -> List[Dict[str, float]]:
        """
        Calculate metric for a batch of text pairs.
        
        Args:
            reference_list (List[str]): List of reference texts
            candidate_list (List[str]): List of candidate texts
            **kwargs: Additional metric-specific parameters
            
        Returns:
            List[Dict[str, float]]: List of metric scores for each pair
            
        Raises:
            ValueError: If lists have different lengths
        """
        if len(reference_list) != len(candidate_list):
            raise ValueError(f"Reference and candidate lists must have same length: "
                           f"{len(reference_list)} vs {len(candidate_list)}")
        
        self.logger.info(f"Calculating {self.name} for batch of {len(reference_list)} pairs")
        
        batch_scores = []
        batch_start_time = time.time()
        
        for i, (ref, cand) in enumerate(zip(reference_list, candidate_list)):
            try:
                scores = self.calculate(ref, cand, **kwargs)
                batch_scores.append(scores)
                
                # Progress logging for large batches
                if (i + 1) % 100 == 0:
                    self.logger.debug(f"Processed {i + 1}/{len(reference_list)} pairs")
                    
            except Exception as e:
                self.logger.error(f"Error calculating {self.name} for pair {i}: {e}")
                # Add empty scores to maintain list alignment
                batch_scores.append({})
        
        batch_time = time.time() - batch_start_time
        self.logger.info(f"Batch calculation completed in {batch_time:.2f}s "
                        f"({batch_time/len(reference_list):.4f}s per pair)")
        
        return batch_scores
    
    def format_results(self, scores: Dict[str, float], precision: int = 4) -> Dict[str, Any]:
        """
        Format metric results for output.
        
        Args:
            scores (Dict[str, float]): Raw metric scores
            precision (int): Number of decimal places for formatting
            
        Returns:
            Dict[str, Any]: Formatted results with metadata
        """
        formatted_scores = {}
        
        # Round scores to specified precision
        for key, value in scores.items():
            if isinstance(value, float):
                formatted_scores[key] = round(value, precision)
            else:
                formatted_scores[key] = value
        
        # Add metadata
        result = {
            "metric_name": self.name,
            "metric_type": self.metric_type,
            "scores": formatted_scores,
            "metadata": {
                "version": self.version,
                "calculation_time": self.last_calculation_time,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return result
    
    def validate_inputs(self, reference: str, candidate: str) -> Tuple[bool, List[str]]:
        """
        Validate input texts for metric calculation.
        
        Args:
            reference (str): Reference text
            candidate (str): Candidate text
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for None or empty strings
        if reference is None:
            issues.append("Reference text is None")
        elif not isinstance(reference, str):
            issues.append(f"Reference text must be string, got {type(reference)}")
        elif len(reference.strip()) == 0:
            issues.append("Reference text is empty")
        
        if candidate is None:
            issues.append("Candidate text is None")
        elif not isinstance(candidate, str):
            issues.append(f"Candidate text must be string, got {type(candidate)}")
        elif len(candidate.strip()) == 0:
            issues.append("Candidate text is empty")
        
        # Check for reasonable length limits
        if isinstance(reference, str) and len(reference) > 50000:
            issues.append("Reference text is extremely long (>50k chars)")
        
        if isinstance(candidate, str) and len(candidate) > 50000:
            issues.append("Candidate text is extremely long (>50k chars)")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for this metric.
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        avg_time = (self.total_calculation_time / self.calculation_count 
                   if self.calculation_count > 0 else 0.0)
        
        return {
            "metric_name": self.name,
            "calculation_count": self.calculation_count,
            "total_time": self.total_calculation_time,
            "average_time": avg_time,
            "last_calculation_time": self.last_calculation_time,
            "is_initialized": self.is_initialized
        }
    
    def reset_performance_stats(self) -> None:
        """Reset performance tracking statistics."""
        self.calculation_count = 0
        self.total_calculation_time = 0.0
        self.last_calculation_time = 0.0
        self.logger.debug(f"Reset performance stats for {self.name}")
    
    def __str__(self) -> str:
        """String representation of the metric."""
        return f"{self.name} (v{self.version}) - {self.metric_type}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the metric."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"type='{self.metric_type}', version='{self.version}', "
                f"initialized={self.is_initialized})")


class DummyMetric(BaseMetric):
    """
    Dummy metric implementation for testing the base class.
    
    Returns random scores for testing purposes.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__("dummy", logger)
        self.description = "Dummy metric for testing base class functionality"
        self.metric_type = "test"
    
    def calculate(self, reference: str, candidate: str, **kwargs) -> Dict[str, float]:
        """
        Calculate dummy scores.
        
        Args:
            reference (str): Reference text
            candidate (str): Candidate text
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, float]: Dummy scores
        """
        # Validate inputs
        is_valid, issues = self.validate_inputs(reference, candidate)
        if not is_valid:
            raise ValueError(f"Invalid inputs: {issues}")
        
        # Calculate dummy scores based on text lengths
        ref_len = len(reference.split())
        cand_len = len(candidate.split())
        
        # Simple similarity based on length ratio
        length_ratio = min(ref_len, cand_len) / max(ref_len, cand_len) if max(ref_len, cand_len) > 0 else 0.0
        
        return {
            "dummy_score": length_ratio,
            "reference_length": float(ref_len),
            "candidate_length": float(cand_len),
            "length_difference": abs(ref_len - cand_len)
        }
    
    def get_name(self) -> str:
        return "Dummy Metric"
    
    def get_description(self) -> str:
        return self.description


def test_base_metric():
    """Test function for the Base Metric class."""
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    try:
        print("🧪 Testing Base Metric Class...")
        
        # Test dummy metric
        dummy = DummyMetric(logger)
        
        # Test basic properties
        print(f"✅ Metric name: {dummy.get_name()}")
        print(f"✅ Metric description: {dummy.get_description()}")
        print(f"✅ Metric type: {dummy.get_metric_type()}")
        print(f"✅ Metric version: {dummy.get_version()}")
        
        # Test configuration
        dummy.configure({"test_param": 1.0, "another_param": "test"})
        print(f"✅ Configuration: {dummy.config}")
        
        # Test single calculation
        ref_text = "Normal chest radiograph with clear lung fields."
        cand_text = "Chest X-ray shows normal lungs and heart."
        
        scores, calc_time = dummy.calculate_with_timing(ref_text, cand_text)
        print(f"✅ Single calculation: {scores}")
        print(f"✅ Calculation time: {calc_time:.4f}s")
        
        # Test result formatting
        formatted = dummy.format_results(scores)
        print(f"✅ Formatted results: {formatted['scores']}")
        
        # Test input validation
        is_valid, issues = dummy.validate_inputs(ref_text, cand_text)
        print(f"✅ Input validation: valid={is_valid}, issues={issues}")
        
        # Test invalid inputs
        is_valid, issues = dummy.validate_inputs("", None)
        print(f"✅ Invalid input test: valid={is_valid}, issues={len(issues)} found")
        
        # Test batch calculation
        ref_list = [
            "Normal chest radiograph.",
            "Clear lung fields bilaterally.",
            "Heart size within normal limits."
        ]
        cand_list = [
            "Normal chest X-ray.",
            "Lungs are clear on both sides.",
            "Normal heart size."
        ]
        
        batch_scores = dummy.calculate_batch(ref_list, cand_list)
        print(f"✅ Batch calculation: {len(batch_scores)} results")
        
        # Test performance stats
        perf_stats = dummy.get_performance_stats()
        print(f"✅ Performance stats: {perf_stats['calculation_count']} calculations")
        
        # Test string representations
        print(f"✅ String repr: {str(dummy)}")
        print(f"✅ Detailed repr: {repr(dummy)}")
        
        print("\n✅ All base metric tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    success = test_base_metric()
    
    if success:
        print("\n🎉 Base Metric tests passed!")
    else:
        print("\n❌ Some tests failed!") 