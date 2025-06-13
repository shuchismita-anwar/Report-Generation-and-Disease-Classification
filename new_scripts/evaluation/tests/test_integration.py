#!/usr/bin/env python3
"""
Integration Tests
================
Comprehensive integration tests for the Medical Report Evaluation System.
Tests end-to-end pipeline functionality, single report scenarios,
batch processing, and error recovery scenarios.

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
import warnings
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import main components
from medical_report_evaluator import MedicalReportEvaluator
from utils.metric_aggregator import MetricAggregator
from utils.results_exporter import ResultsExporter


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete end-to-end pipeline functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.test_dir, "config")
        self.output_dir = os.path.join(self.test_dir, "output")
        
        # Create directories
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test configuration
        self.test_config = {
            "metrics": {
                "weights": {
                    "bleu": 0.2,
                    "rouge": 0.2,
                    "meteor": 0.15,
                    "bertscore": 0.2,
                    "cider": 0.1,
                    "medical": 0.15
                }
            },
            "evaluation": {
                "batch_size": 10,
                "output_dir": self.output_dir,
                "save_individual_scores": True,
                "save_aggregated_scores": True
            },
            "logging": {
                "level": "INFO",
                "log_file": os.path.join(self.test_dir, "evaluation.log")
            }
        }
        
        # Save test configuration
        config_path = os.path.join(self.config_dir, "evaluation_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)
        
        # Initialize evaluator
        self.evaluator = MedicalReportEvaluator(config_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        # Should initialize successfully
        self.assertIsNotNone(self.evaluator)
        self.assertIsNotNone(self.evaluator.config)
        self.assertIsNotNone(self.evaluator.logger)
        self.assertIsNotNone(self.evaluator.metrics)
        
        # Should have all expected metrics
        expected_metrics = ["bleu", "rouge", "meteor", "bertscore", "cider", "medical"]
        for metric_name in expected_metrics:
            self.assertIn(metric_name, self.evaluator.metrics)
    
    def test_single_report_evaluation_pipeline(self):
        """Test complete single report evaluation pipeline."""
        # Test data
        reference = "Patient presents with acute chest pain and shortness of breath. Chest X-ray shows bilateral lung infiltrates consistent with pneumonia."
        candidate = "Patient has chest pain and dyspnea. X-ray reveals bilateral pneumonia infiltrates."
        
        # Evaluate single report
        result = self.evaluator.evaluate_single(reference, candidate)
        
        # Should return comprehensive results
        self.assertIsInstance(result, dict)
        
        # Should have individual metric scores
        expected_metrics = ["bleu", "rouge", "meteor", "bertscore", "cider", "medical"]
        for metric_name in expected_metrics:
            self.assertIn(metric_name, result)
            self.assertIsInstance(result[metric_name], dict)
        
        # Should have aggregated score
        self.assertIn("overall_score", result)
        self.assertIsInstance(result["overall_score"], (int, float))
        self.assertTrue(0.0 <= result["overall_score"] <= 1.0)
        
        # Should have metadata
        self.assertIn("evaluation_metadata", result)
        metadata = result["evaluation_metadata"]
        self.assertIn("timestamp", metadata)
        self.assertIn("evaluation_time", metadata)
        self.assertIn("reference_length", metadata)
        self.assertIn("candidate_length", metadata)
    
    def test_batch_evaluation_pipeline(self):
        """Test complete batch evaluation pipeline."""
        # Test data
        test_cases = [
            {
                "id": "case_001",
                "reference": "Patient has pneumonia with bilateral infiltrates.",
                "candidate": "Patient shows pneumonia and bilateral lung infiltrates."
            },
            {
                "id": "case_002",
                "reference": "No evidence of acute cardiopulmonary disease.",
                "candidate": "No acute cardiac or pulmonary abnormalities detected."
            },
            {
                "id": "case_003",
                "reference": "Left lower lobe consolidation consistent with infection.",
                "candidate": "Left lower lung consolidation suggests pneumonia."
            }
        ]
        
        # Prepare input data
        references = [case["reference"] for case in test_cases]
        candidates = [case["candidate"] for case in test_cases]
        case_ids = [case["id"] for case in test_cases]
        
        # Evaluate batch
        results = self.evaluator.evaluate_batch(
            references=references,
            candidates=candidates,
            case_ids=case_ids,
            save_results=True
        )
        
        # Should return results for all cases
        self.assertEqual(len(results), len(test_cases))
        
        # Each result should be complete
        for i, result in enumerate(results):
            self.assertIsInstance(result, dict)
            self.assertEqual(result["case_id"], case_ids[i])
            self.assertIn("overall_score", result)
            self.assertIn("evaluation_metadata", result)
            
            # Should have all metric scores
            expected_metrics = ["bleu", "rouge", "meteor", "bertscore", "cider", "medical"]
            for metric_name in expected_metrics:
                self.assertIn(metric_name, result)
        
        # Should have saved results to files
        output_files = os.listdir(self.output_dir)
        self.assertGreater(len(output_files), 0)
        
        # Should have JSON results file
        json_files = [f for f in output_files if f.endswith('.json')]
        self.assertGreater(len(json_files), 0)


class TestSingleReportScenarios(unittest.TestCase):
    """Test various single report evaluation scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create minimal configuration
        config_path = os.path.join(self.test_dir, "config.json")
        config = {
            "metrics": {"weights": {"bleu": 1.0}},
            "evaluation": {"output_dir": self.test_dir}
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        self.evaluator = MedicalReportEvaluator(config_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_identical_reports(self):
        """Test evaluation of identical reports."""
        text = "Patient has clear lung fields with no acute abnormalities."
        
        result = self.evaluator.evaluate_single(text, text)
        
        # Should have very high scores
        self.assertGreater(result["overall_score"], 0.9)
        
        # BLEU should be perfect
        bleu_scores = result["bleu"]
        self.assertAlmostEqual(bleu_scores["bleu_1"], 1.0, places=2)
    
    def test_completely_different_reports(self):
        """Test evaluation of completely different reports."""
        reference = "Patient has pneumonia with bilateral infiltrates."
        candidate = "The weather is sunny today with clear skies."
        
        result = self.evaluator.evaluate_single(reference, candidate)
        
        # Should have very low scores
        self.assertLess(result["overall_score"], 0.3)
    
    def test_medical_vs_nonmedical_content(self):
        """Test medical vs non-medical content."""
        medical_ref = "Patient presents with acute myocardial infarction."
        medical_cand = "Patient has heart attack symptoms."
        
        nonmedical_ref = "The cat sat on the mat."
        nonmedical_cand = "A cat is sitting on a mat."
        
        medical_result = self.evaluator.evaluate_single(medical_ref, medical_cand)
        nonmedical_result = self.evaluator.evaluate_single(nonmedical_ref, nonmedical_cand)
        
        # Medical content should score higher on medical metric
        medical_score = medical_result["medical"]["overall_medical"]
        nonmedical_score = nonmedical_result["medical"]["overall_medical"]
        
        self.assertGreater(medical_score, nonmedical_score)


class TestErrorRecoveryScenarios(unittest.TestCase):
    """Test error recovery scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create configuration
        config_path = os.path.join(self.test_dir, "config.json")
        config = {
            "metrics": {"weights": {"bleu": 1.0}},
            "evaluation": {"output_dir": self.test_dir}
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        self.evaluator = MedicalReportEvaluator(config_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_empty_text_handling(self):
        """Test empty text handling."""
        # Empty reference
        with self.assertRaises(ValueError):
            self.evaluator.evaluate_single("", "some text")
        
        # Empty candidate
        with self.assertRaises(ValueError):
            self.evaluator.evaluate_single("some text", "")
        
        # Both empty
        with self.assertRaises(ValueError):
            self.evaluator.evaluate_single("", "")
    
    def test_whitespace_only_handling(self):
        """Test whitespace-only text handling."""
        # Whitespace reference
        with self.assertRaises(ValueError):
            self.evaluator.evaluate_single("   \t\n", "some text")
        
        # Whitespace candidate
        with self.assertRaises(ValueError):
            self.evaluator.evaluate_single("some text", "   \t\n")
    
    def test_invalid_configuration_handling(self):
        """Test invalid configuration handling."""
        # Invalid config file
        invalid_config_path = os.path.join(self.test_dir, "invalid.json")
        with open(invalid_config_path, 'w') as f:
            f.write("{invalid json")
        
        with self.assertRaises((json.JSONDecodeError, ValueError)):
            MedicalReportEvaluator(invalid_config_path)
        
        # Missing config file
        missing_config_path = os.path.join(self.test_dir, "missing.json")
        with self.assertRaises(FileNotFoundError):
            MedicalReportEvaluator(missing_config_path)


class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create comprehensive configuration
        config_path = os.path.join(self.test_dir, "system_config.json")
        config = {
            "metrics": {
                "weights": {
                    "bleu": 0.15,
                    "rouge": 0.15,
                    "meteor": 0.15,
                    "bertscore": 0.2,
                    "cider": 0.15,
                    "medical": 0.2
                },
                "bleu": {"smoothing_method": "add_one"},
                "bertscore": {"model_type": "distilbert-base-uncased"}
            },
            "evaluation": {
                "batch_size": 20,
                "output_dir": self.test_dir,
                "save_individual_scores": True,
                "save_aggregated_scores": True,
                "export_formats": ["json", "csv", "txt"]
            },
            "logging": {
                "level": "INFO",
                "log_file": os.path.join(self.test_dir, "system.log")
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        self.evaluator = MedicalReportEvaluator(config_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_complete_system_workflow(self):
        """Test complete system workflow from input to output."""
        # Prepare comprehensive test data
        test_data = [
            {
                "case_id": "sys_001",
                "reference": "Patient presents with acute chest pain radiating to left arm. ECG shows ST elevation in leads II, III, aVF consistent with inferior STEMI.",
                "candidate": "Patient has chest pain extending to left arm. ECG demonstrates ST elevation in inferior leads suggesting inferior wall myocardial infarction."
            },
            {
                "case_id": "sys_002",
                "reference": "Chest X-ray demonstrates bilateral lower lobe pneumonia with pleural effusions.",
                "candidate": "X-ray shows pneumonia in both lower lung zones with fluid collection in pleural spaces."
            },
            {
                "case_id": "sys_003",
                "reference": "No evidence of acute intracranial abnormality. Normal brain parenchyma.",
                "candidate": "No acute brain abnormalities detected. Brain tissue appears normal."
            }
        ]
        
        references = [case["reference"] for case in test_data]
        candidates = [case["candidate"] for case in test_data]
        case_ids = [case["case_id"] for case in test_data]
        
        # Execute complete workflow
        results = self.evaluator.evaluate_batch(
            references=references,
            candidates=candidates,
            case_ids=case_ids,
            save_results=True
        )
        
        # Verify results
        self.assertEqual(len(results), 3)
        
        # Verify each result has complete structure
        for result in results:
            # Basic structure
            self.assertIn("case_id", result)
            self.assertIn("overall_score", result)
            self.assertIn("evaluation_metadata", result)
            
            # All metrics present
            expected_metrics = ["bleu", "rouge", "meteor", "bertscore", "cider", "medical"]
            for metric in expected_metrics:
                self.assertIn(metric, result)
                self.assertIsInstance(result[metric], dict)
            
            # Score ranges
            self.assertTrue(0.0 <= result["overall_score"] <= 1.0)
        
        # Verify output files generated
        output_files = os.listdir(self.test_dir)
        
        # Should have results files
        json_files = [f for f in output_files if f.endswith('.json')]
        self.assertGreater(len(json_files), 0)
        
        # Should have log file
        log_files = [f for f in output_files if f.endswith('.log')]
        self.assertGreater(len(log_files), 0)


def run_integration_tests():
    """Run all integration tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEndToEndPipeline,
        TestSingleReportScenarios,
        TestErrorRecoveryScenarios,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    
    print(f"\n=== Integration Test Summary ===")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors - skipped}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print(f"Success Rate: {((total_tests - failures - errors) / total_tests * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    # Run integration tests
    success = run_integration_tests()
    
    if success:
        print("\n✅ All integration tests passed!")
        exit(0)
    else:
        print("\n❌ Some integration tests failed!")
        exit(1)
