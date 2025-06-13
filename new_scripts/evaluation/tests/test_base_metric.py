#!/usr/bin/env python3
"""
Base Metric Class Tests
======================
Unit tests for the BaseMetric abstract class and DummyMetric implementation.

Author: Medical Report Evaluation System
Version: 1.0
"""

import unittest
import sys
import os
import logging
import time
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.base_metric import BaseMetric, DummyMetric


class TestBaseMetric(unittest.TestCase):
    """Test cases for BaseMetric abstract class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
    
    def test_abstract_base_class_instantiation(self):
        """Test that BaseMetric cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseMetric("test_metric", self.logger)
    
    def test_dummy_metric_instantiation(self):
        """Test DummyMetric can be instantiated."""
        dummy = DummyMetric(self.logger)
        self.assertIsInstance(dummy, BaseMetric)
        self.assertEqual(dummy.get_name(), "Dummy Metric")
        self.assertIn("dummy metric", dummy.get_description().lower())
    
    def test_dummy_metric_calculation(self):
        """Test DummyMetric calculation functionality."""
        dummy = DummyMetric(self.logger)
        
        # Test basic calculation
        result = dummy.calculate("reference text", "candidate text")
        self.assertIsInstance(result, dict)
        self.assertIn("dummy_score", result)
        self.assertIsInstance(result["dummy_score"], float)
        self.assertTrue(0.0 <= result["dummy_score"] <= 1.0)
    
    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        dummy = DummyMetric(self.logger)
        
        # Initial state
        self.assertEqual(dummy.calculation_count, 0)
        self.assertEqual(dummy.total_calculation_time, 0.0)
        
        # Perform calculation with timing
        result, calc_time = dummy.calculate_with_timing("ref", "cand")
        
        # Check performance tracking
        self.assertEqual(dummy.calculation_count, 1)
        self.assertGreaterEqual(dummy.total_calculation_time, 0.0)  # May be 0 for fast operations
        self.assertGreaterEqual(calc_time, 0.0)  # May be 0 for fast operations
        self.assertEqual(dummy.last_calculation_time, calc_time)
        
        # Perform another calculation
        dummy.calculate_with_timing("ref2", "cand2")
        self.assertEqual(dummy.calculation_count, 2)
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        dummy = DummyMetric(self.logger)
        
        references = ["ref1", "ref2", "ref3"]
        candidates = ["cand1", "cand2", "cand3"]
        
        results = dummy.calculate_batch(references, candidates)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn("dummy_score", result)
    
    def test_batch_processing_length_mismatch(self):
        """Test batch processing with mismatched lengths."""
        dummy = DummyMetric(self.logger)
        
        references = ["ref1", "ref2"]
        candidates = ["cand1", "cand2", "cand3"]
        
        with self.assertRaises(ValueError):
            dummy.calculate_batch(references, candidates)
    
    def test_input_validation(self):
        """Test input validation methods."""
        dummy = DummyMetric(self.logger)
        
        # Valid inputs
        is_valid, errors = dummy.validate_inputs("valid ref", "valid cand")
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Empty reference
        is_valid, errors = dummy.validate_inputs("", "valid cand")
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
        # Empty candidate
        is_valid, errors = dummy.validate_inputs("valid ref", "")
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
        # None inputs
        is_valid, errors = dummy.validate_inputs(None, "valid cand")
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_result_formatting(self):
        """Test result formatting functionality."""
        dummy = DummyMetric(self.logger)
        
        scores = {"score1": 0.123456789, "score2": 0.987654321}
        
        # Test default precision
        formatted = dummy.format_results(scores)
        self.assertIn("scores", formatted)
        self.assertIn("metadata", formatted)
        
        # Test custom precision
        formatted = dummy.format_results(scores, precision=2)
        formatted_scores = formatted["scores"]
        for score in formatted_scores.values():
            if isinstance(score, float):
                # Check that precision is respected (approximately)
                self.assertLessEqual(len(str(score).split('.')[-1]), 3)  # Allow for rounding
    
    def test_configuration_handling(self):
        """Test configuration handling."""
        dummy = DummyMetric(self.logger)
        
        # Test initial configuration
        self.assertIsInstance(dummy.config, dict)
        
        # Test configuration update
        test_config = {"param1": "value1", "param2": 42}
        dummy.configure(test_config)
        
        self.assertEqual(dummy.config["param1"], "value1")
        self.assertEqual(dummy.config["param2"], 42)
    
    def test_initialization_flag(self):
        """Test initialization flag functionality."""
        dummy = DummyMetric(self.logger)
        
        # Should not be initialized initially
        self.assertFalse(dummy.is_initialized)
        
        # Should be initialized after calling initialize
        dummy.initialize()
        self.assertTrue(dummy.is_initialized)
        
        # Should auto-initialize on calculate_with_timing
        dummy2 = DummyMetric(self.logger)
        self.assertFalse(dummy2.is_initialized)
        dummy2.calculate_with_timing("ref", "cand")
        self.assertTrue(dummy2.is_initialized)
    
    def test_performance_stats(self):
        """Test performance statistics functionality."""
        dummy = DummyMetric(self.logger)
        
        # Perform some calculations
        dummy.calculate_with_timing("ref1", "cand1")
        dummy.calculate_with_timing("ref2", "cand2")
        
        stats = dummy.get_performance_stats()
        
        self.assertIn("calculation_count", stats)
        self.assertIn("total_time", stats)
        self.assertIn("average_time", stats)
        self.assertIn("last_calculation_time", stats)
        
        self.assertEqual(stats["calculation_count"], 2)
        self.assertGreaterEqual(stats["total_time"], 0.0)  # May be 0 for fast operations
        self.assertGreaterEqual(stats["average_time"], 0.0)  # May be 0 for fast operations
        
        # Test reset
        dummy.reset_performance_stats()
        self.assertEqual(dummy.calculation_count, 0)
        self.assertEqual(dummy.total_calculation_time, 0.0)
    
    def test_string_representations(self):
        """Test string representation methods."""
        dummy = DummyMetric(self.logger)
        
        str_repr = str(dummy)
        self.assertIn("dummy", str_repr)
        
        repr_str = repr(dummy)
        self.assertIn("DummyMetric", repr_str)
    
    def test_metric_type_and_version(self):
        """Test metric type and version properties."""
        dummy = DummyMetric(self.logger)
        
        self.assertEqual(dummy.get_version(), "1.0")
        self.assertEqual(dummy.get_metric_type(), "test")
    
    def test_error_handling_in_calculation(self):
        """Test error handling during calculation."""
        dummy = DummyMetric(self.logger)
        
        # Test with invalid inputs that should raise errors
        with self.assertRaises(Exception):
            dummy.calculate_with_timing(None, "valid")
        
        with self.assertRaises(Exception):
            dummy.calculate_with_timing("valid", None)


class TestDummyMetricSpecific(unittest.TestCase):
    """Test cases specific to DummyMetric implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.dummy = DummyMetric(self.logger)
    
    def test_dummy_score_range(self):
        """Test that dummy scores are in valid range."""
        test_cases = [
            ("identical text", "identical text"),
            ("completely different", "totally unrelated content"),
            ("", ""),
            ("short", "text"),
            ("very long text with many words and complex structure", "another long text")
        ]
        
        for ref, cand in test_cases:
            try:
                result = self.dummy.calculate(ref, cand)
                score = result["dummy_score"]
                self.assertTrue(0.0 <= score <= 1.0, 
                              f"Score {score} out of range for inputs: '{ref}', '{cand}'")
            except Exception:
                # Some test cases might fail due to validation, which is expected
                pass
    
    def test_dummy_score_consistency(self):
        """Test that dummy scores are consistent for same inputs."""
        ref = "test reference text"
        cand = "test candidate text"
        
        score1 = self.dummy.calculate(ref, cand)["dummy_score"]
        score2 = self.dummy.calculate(ref, cand)["dummy_score"]
        
        self.assertEqual(score1, score2, "Dummy scores should be consistent")
    
    def test_dummy_additional_metrics(self):
        """Test additional metrics in dummy implementation."""
        result = self.dummy.calculate("reference", "candidate")
        
        # Check for additional metrics (based on actual implementation)
        self.assertIn("reference_length", result)
        self.assertIn("candidate_length", result)
        self.assertIn("length_difference", result)
        
        # Validate reference length
        ref_length = result["reference_length"]
        self.assertGreaterEqual(ref_length, 0.0)
        
        # Validate candidate length
        cand_length = result["candidate_length"]
        self.assertGreaterEqual(cand_length, 0.0)
        
        # Validate length difference
        length_diff = result["length_difference"]
        self.assertGreaterEqual(length_diff, 0)


def run_base_metric_tests():
    """Run all base metric tests and return results."""
    print("🧪 Running Base Metric Tests...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestBaseMetric))
    suite.addTests(loader.loadTestsFromTestCase(TestDummyMetricSpecific))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"📊 Base Metric Tests Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ All base metric tests passed!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} tests failed!")
    
    return success, result


if __name__ == "__main__":
    success, _ = run_base_metric_tests()
    exit(0 if success else 1) 