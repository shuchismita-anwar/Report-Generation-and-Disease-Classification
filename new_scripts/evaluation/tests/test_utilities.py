#!/usr/bin/env python3
"""
Utilities Tests
===============
Unit tests for utility classes (MetricAggregator, ResultsExporter).

Author: Medical Report Evaluation System
Version: 1.0
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metric_aggregator import MetricAggregator
from utils.results_exporter import ResultsExporter


class TestMetricAggregator(unittest.TestCase):
    """Test cases for MetricAggregator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.aggregator = MetricAggregator(self.logger)
    
    def test_aggregator_instantiation(self):
        """Test MetricAggregator instantiation."""
        self.assertIsInstance(self.aggregator, MetricAggregator)
        self.assertEqual(self.aggregator.logger, self.logger)
    
    def test_weighted_average(self):
        """Test weighted average calculation."""
        scores = {"bleu": 0.8, "rouge": 0.7, "meteor": 0.9}
        weights = {"bleu": 0.3, "rouge": 0.3, "meteor": 0.4}
        
        result = self.aggregator.weighted_average(scores, weights)
        expected = 0.8 * 0.3 + 0.7 * 0.3 + 0.9 * 0.4
        
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_weighted_average_equal_weights(self):
        """Test weighted average with equal weights."""
        scores = {"metric1": 0.6, "metric2": 0.8, "metric3": 0.7}
        
        result = self.aggregator.weighted_average(scores)
        expected = (0.6 + 0.8 + 0.7) / 3
        
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_simple_average(self):
        """Test simple average calculation."""
        scores = {"metric1": 0.5, "metric2": 0.7, "metric3": 0.9}
        
        result = self.aggregator.simple_average(scores)
        expected = (0.5 + 0.7 + 0.9) / 3
        
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_harmonic_mean(self):
        """Test harmonic mean calculation."""
        scores = {"metric1": 0.4, "metric2": 0.6, "metric3": 0.8}
        
        result = self.aggregator.harmonic_mean(scores)
        expected = 3 / (1/0.4 + 1/0.6 + 1/0.8)
        
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_harmonic_mean_with_zero(self):
        """Test harmonic mean with zero values."""
        scores = {"metric1": 0.0, "metric2": 0.6, "metric3": 0.8}
        
        result = self.aggregator.harmonic_mean(scores)
        self.assertEqual(result, 0.0)
    
    def test_geometric_mean(self):
        """Test geometric mean calculation."""
        scores = {"metric1": 0.4, "metric2": 0.6, "metric3": 0.8}
        
        result = self.aggregator.geometric_mean(scores)
        expected = (0.4 * 0.6 * 0.8) ** (1/3)
        
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_geometric_mean_with_zero(self):
        """Test geometric mean with zero values."""
        scores = {"metric1": 0.0, "metric2": 0.6, "metric3": 0.8}
        
        result = self.aggregator.geometric_mean(scores)
        self.assertEqual(result, 0.0)
    
    def test_median_score(self):
        """Test median calculation."""
        scores = {"metric1": 0.3, "metric2": 0.7, "metric3": 0.5, "metric4": 0.9}
        
        result = self.aggregator.median_score(scores)
        expected = (0.5 + 0.7) / 2  # Median of [0.3, 0.5, 0.7, 0.9]
        
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_aggregate_scores(self):
        """Test comprehensive score aggregation."""
        scores = {"bleu": 0.8, "rouge": 0.7, "meteor": 0.9}
        weights = {"bleu": 0.3, "rouge": 0.3, "meteor": 0.4}
        
        result = self.aggregator.aggregate_scores(scores, weights, method="weighted_average")
        
        self.assertIn("overall_score", result)
        self.assertIn("method", result)
        self.assertIn("individual_scores", result)
        self.assertIn("statistics", result)
        
        self.assertEqual(result["method"], "weighted_average")
        self.assertEqual(result["individual_scores"], scores)
    
    def test_normalize_scores(self):
        """Test score normalization."""
        scores = {"metric1": 0.3, "metric2": 0.7, "metric3": 0.5}
        
        normalized = self.aggregator.normalize_scores(scores)
        
        # Check that all values are between 0 and 1
        for score in normalized.values():
            self.assertTrue(0.0 <= score <= 1.0)
        
        # Check that the relative ordering is preserved
        original_order = sorted(scores.items(), key=lambda x: x[1])
        normalized_order = sorted(normalized.items(), key=lambda x: x[1])
        
        self.assertEqual([x[0] for x in original_order], [x[0] for x in normalized_order])
    
    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        scores = [0.6, 0.7, 0.8, 0.9, 0.5, 0.75, 0.85, 0.65, 0.95, 0.55]
        
        ci = self.aggregator.calculate_confidence_interval(scores)
        
        self.assertIn("mean", ci)
        self.assertIn("std", ci)
        self.assertIn("lower_bound", ci)
        self.assertIn("upper_bound", ci)
        self.assertIn("confidence_level", ci)
        
        # Check that bounds make sense
        self.assertLess(ci["lower_bound"], ci["mean"])
        self.assertGreater(ci["upper_bound"], ci["mean"])
    
    def test_batch_aggregate(self):
        """Test batch aggregation."""
        batch_results = [
            {"bleu": 0.8, "rouge": 0.7},
            {"bleu": 0.6, "rouge": 0.8},
            {"bleu": 0.9, "rouge": 0.6}
        ]
        
        result = self.aggregator.batch_aggregate(batch_results)
        
        self.assertIn("aggregated_scores", result)
        self.assertIn("statistics", result)
        self.assertIn("confidence_intervals", result)
        
        # Check that we have stats for each metric
        self.assertIn("bleu", result["statistics"])
        self.assertIn("rouge", result["statistics"])


class TestResultsExporter(unittest.TestCase):
    """Test cases for ResultsExporter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.exporter = ResultsExporter(self.logger)
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample results data
        self.sample_results = {
            "evaluation_id": "test_eval_001",
            "timestamp": "2024-01-15T10:30:00",
            "overall_score": 0.75,
            "individual_scores": {
                "bleu": 0.8,
                "rouge": 0.7,
                "meteor": 0.9
            },
            "metadata": {
                "reference_length": 150,
                "candidate_length": 140,
                "evaluation_time": 2.5
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_exporter_instantiation(self):
        """Test ResultsExporter instantiation."""
        self.assertIsInstance(self.exporter, ResultsExporter)
        self.assertEqual(self.exporter.logger, self.logger)
    
    def test_export_json(self):
        """Test JSON export functionality."""
        output_path = os.path.join(self.temp_dir, "results.json")
        
        success = self.exporter.export_json(self.sample_results, output_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data["evaluation_id"], "test_eval_001")
        self.assertEqual(loaded_data["overall_score"], 0.75)
    
    def test_export_csv(self):
        """Test CSV export functionality."""
        output_path = os.path.join(self.temp_dir, "results.csv")
        
        success = self.exporter.export_csv(self.sample_results, output_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, 'r') as f:
            content = f.read()
        
        self.assertIn("evaluation_id", content)
        self.assertIn("test_eval_001", content)
        self.assertIn("0.75", content)
    
    def test_export_txt(self):
        """Test TXT export functionality."""
        output_path = os.path.join(self.temp_dir, "results.txt")
        
        success = self.exporter.export_txt(self.sample_results, output_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, 'r') as f:
            content = f.read()
        
        self.assertIn("Evaluation Results", content)
        self.assertIn("test_eval_001", content)
        self.assertIn("0.75", content)
    
    def test_export_html(self):
        """Test HTML export functionality."""
        output_path = os.path.join(self.temp_dir, "results.html")
        
        success = self.exporter.export_html(self.sample_results, output_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, 'r') as f:
            content = f.read()
        
        self.assertIn("<html>", content)
        self.assertIn("test_eval_001", content)
        self.assertIn("0.75", content)
    
    def test_export_all_formats(self):
        """Test exporting to all formats."""
        base_path = os.path.join(self.temp_dir, "results")
        
        success = self.exporter.export_all_formats(self.sample_results, base_path)
        
        self.assertTrue(success)
        
        # Check that all format files were created
        expected_files = [
            f"{base_path}.json",
            f"{base_path}.csv",
            f"{base_path}.txt",
            f"{base_path}.html"
        ]
        
        for file_path in expected_files:
            self.assertTrue(os.path.exists(file_path), f"File not found: {file_path}")
    
    def test_batch_export(self):
        """Test batch export functionality."""
        batch_results = [
            {**self.sample_results, "evaluation_id": "test_001"},
            {**self.sample_results, "evaluation_id": "test_002", "overall_score": 0.85},
            {**self.sample_results, "evaluation_id": "test_003", "overall_score": 0.65}
        ]
        
        output_path = os.path.join(self.temp_dir, "batch_results.json")
        
        success = self.exporter.export_batch_json(batch_results, output_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(len(loaded_data["results"]), 3)
        self.assertIn("summary", loaded_data)
    
    def test_format_numeric_precision(self):
        """Test numeric precision formatting."""
        data = {"score": 0.123456789, "other": "text"}
        
        formatted = self.exporter._format_numeric_precision(data, precision=3)
        
        self.assertEqual(formatted["score"], 0.123)
        self.assertEqual(formatted["other"], "text")
    
    def test_invalid_export_path(self):
        """Test export with invalid path."""
        invalid_path = "/invalid/path/results.json"
        
        success = self.exporter.export_json(self.sample_results, invalid_path)
        
        self.assertFalse(success)


def run_utilities_tests():
    """Run all utilities tests and return results."""
    print("🧪 Running Utilities Tests...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMetricAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestResultsExporter))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"📊 Utilities Tests Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ All utilities tests passed!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} tests failed!")
    
    return success, result


if __name__ == "__main__":
    success, _ = run_utilities_tests()
    exit(0 if success else 1) 