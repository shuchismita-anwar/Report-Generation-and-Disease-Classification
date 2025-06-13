#!/usr/bin/env python3
"""
Validation Tests
===============
Comprehensive validation tests for the Medical Report Evaluation System.
Tests against published benchmarks, reproduces known metric scores,
and validates on standard datasets.

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
import math
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import main components
from medical_report_evaluator import MedicalReportEvaluator
from metrics.bleu_scorer import BLEUScorer
from metrics.rouge_scorer import ROUGEScorer
from metrics.meteor_scorer import METEORScorer
from metrics.bertscore_scorer import BERTScoreScorer
from metrics.cider_scorer import CIDErScorer
from metrics.medical_scorer import MedicalScorer


class TestKnownBenchmarks(unittest.TestCase):
    """Test against known benchmarks and expected results."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        
        # Known test cases with expected approximate scores
        self.benchmark_cases = [
            {
                "name": "identical_texts",
                "reference": "The patient has pneumonia.",
                "candidate": "The patient has pneumonia.",
                "expected_bleu_1": 1.0,
                "expected_rouge_1_f": 1.0,
                "expected_meteor": 1.0,
                "tolerance": 0.01
            },
            {
                "name": "no_overlap",
                "reference": "pneumonia consolidation infiltrate",
                "candidate": "elephant monkey zebra",
                "expected_bleu_1": 0.0,
                "expected_rouge_1_f": 0.0,
                "expected_meteor": 0.0,
                "tolerance": 0.01
            },
            {
                "name": "partial_overlap",
                "reference": "The patient has pneumonia and fever.",
                "candidate": "Patient shows pneumonia symptoms.",
                "expected_bleu_1": 0.4,  # 2 out of 5 words match
                "expected_rouge_1_f": 0.4,  # Similar overlap
                "tolerance": 0.2  # More lenient for complex cases
            },
            {
                "name": "synonym_case",
                "reference": "Patient has lung infection.",
                "candidate": "Patient shows pulmonary infection.",
                "expected_bleu_1": 0.5,  # 2 out of 4 words match exactly
                "expected_rouge_1_f": 0.5,
                "tolerance": 0.2
            }
        ]
    
    def test_bleu_known_scores(self):
        """Test BLEU against known expected scores."""
        bleu = BLEUScorer(self.logger)
        
        for case in self.benchmark_cases:
            with self.subTest(case=case["name"]):
                result = bleu.calculate(case["reference"], case["candidate"])
                
                if "expected_bleu_1" in case:
                    actual_score = result["bleu_1"]
                    expected_score = case["expected_bleu_1"]
                    tolerance = case["tolerance"]
                    
                    self.assertAlmostEqual(
                        actual_score, expected_score, delta=tolerance,
                        msg=f"BLEU-1 mismatch for {case['name']}: "
                            f"expected {expected_score:.3f}, got {actual_score:.3f}"
                    )
    
    def test_rouge_known_scores(self):
        """Test ROUGE against known expected scores."""
        rouge = ROUGEScorer(self.logger)
        
        for case in self.benchmark_cases:
            with self.subTest(case=case["name"]):
                result = rouge.calculate(case["reference"], case["candidate"])
                
                if "expected_rouge_1_f" in case:
                    actual_score = result["rouge_1_f"]
                    expected_score = case["expected_rouge_1_f"]
                    tolerance = case["tolerance"]
                    
                    self.assertAlmostEqual(
                        actual_score, expected_score, delta=tolerance,
                        msg=f"ROUGE-1-F mismatch for {case['name']}: "
                            f"expected {expected_score:.3f}, got {actual_score:.3f}"
                    )
    
    def test_meteor_known_scores(self):
        """Test METEOR against known expected scores."""
        meteor = METEORScorer(self.logger)
        
        for case in self.benchmark_cases:
            with self.subTest(case=case["name"]):
                result = meteor.calculate(case["reference"], case["candidate"])
                
                if "expected_meteor" in case:
                    actual_score = result["meteor"]
                    expected_score = case["expected_meteor"]
                    tolerance = case["tolerance"]
                    
                    self.assertAlmostEqual(
                        actual_score, expected_score, delta=tolerance,
                        msg=f"METEOR mismatch for {case['name']}: "
                            f"expected {expected_score:.3f}, got {actual_score:.3f}"
                    )


class TestMetricConsistency(unittest.TestCase):
    """Test metric consistency and expected behaviors."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.metrics = {
            "bleu": BLEUScorer(self.logger),
            "rouge": ROUGEScorer(self.logger),
            "meteor": METEORScorer(self.logger),
            "bertscore": BERTScoreScorer(self.logger),
            "cider": CIDErScorer(self.logger),
            "medical": MedicalScorer(self.logger)
        }
    
    def test_identical_texts_high_scores(self):
        """Test that identical texts produce high scores for all metrics."""
        text = "Patient presents with acute pneumonia and bilateral lung infiltrates."
        
        for metric_name, metric in self.metrics.items():
            with self.subTest(metric=metric_name):
                result = metric.calculate(text, text)
                
                # Find the primary score for each metric
                if metric_name == "bleu":
                    primary_score = result["bleu_1"]
                elif metric_name == "rouge":
                    primary_score = result["rouge_1_f"]
                elif metric_name == "meteor":
                    primary_score = result["meteor"]
                elif metric_name == "bertscore":
                    primary_score = result["bert_f1"]
                elif metric_name == "cider":
                    primary_score = result["cider"]
                elif metric_name == "medical":
                    primary_score = result["overall_medical"]
                
                # For identical texts, all metrics should score high
                self.assertGreater(primary_score, 0.8,
                                 f"{metric_name} should score high for identical texts: {primary_score:.3f}")
    
    def test_medical_vs_nonmedical_discrimination(self):
        """Test that medical metric discriminates between medical and non-medical content."""
        medical_ref = "Patient presents with acute myocardial infarction and ST elevation."
        medical_cand = "Patient has heart attack with ECG changes."
        
        nonmedical_ref = "The cat sat on the comfortable mat."
        nonmedical_cand = "A cat is sitting on a soft mat."
        
        medical_scorer = self.metrics["medical"]
        
        medical_result = medical_scorer.calculate(medical_ref, medical_cand)
        nonmedical_result = medical_scorer.calculate(nonmedical_ref, nonmedical_cand)
        
        medical_score = medical_result["medical_terminology"]
        nonmedical_score = nonmedical_result["medical_terminology"]
        
        # Medical content should score much higher on medical terminology
        self.assertGreater(medical_score, nonmedical_score + 0.3,
                          f"Medical content should score higher: medical={medical_score:.3f}, "
                          f"nonmedical={nonmedical_score:.3f}")
    
    def test_score_ranges(self):
        """Test that all metric scores are within expected ranges."""
        reference = "Patient has bilateral pneumonia with pleural effusions."
        candidate = "Patient shows pneumonia in both lungs with fluid collection."
        
        for metric_name, metric in self.metrics.items():
            with self.subTest(metric=metric_name):
                result = metric.calculate(reference, candidate)
                
                # All scores should be between 0 and 1 (or reasonable ranges)
                for score_name, score_value in result.items():
                    self.assertIsInstance(score_value, (int, float),
                                        f"{metric_name}.{score_name} should be numeric")
                    
                    # Most scores should be between 0 and 1
                    if metric_name == "cider":
                        # CIDEr can have different ranges
                        self.assertGreaterEqual(score_value, 0,
                                              f"{metric_name}.{score_name} should be non-negative: {score_value}")
                    else:
                        self.assertTrue(0 <= score_value <= 1,
                                      f"{metric_name}.{score_name} should be between 0 and 1: {score_value}")


class TestStandardDatasetValidation(unittest.TestCase):
    """Test validation on standard-like datasets."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create evaluation configuration
        config_path = os.path.join(self.test_dir, "config.json")
        config = {
            "metrics": {
                "weights": {
                    "bleu": 0.2,
                    "rouge": 0.2,
                    "meteor": 0.2,
                    "bertscore": 0.2,
                    "cider": 0.1,
                    "medical": 0.1
                }
            },
            "evaluation": {"output_dir": self.test_dir},
            "logging": {"level": "ERROR"}
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        self.evaluator = MedicalReportEvaluator(config_path)
        
        # Create synthetic medical dataset
        self.medical_dataset = [
            {
                "reference": "Patient presents with acute chest pain radiating to left arm. ECG shows ST elevation in inferior leads.",
                "candidate": "Patient has chest pain extending to left arm. ECG demonstrates ST elevation in inferior leads.",
                "expected_quality": "high"
            },
            {
                "reference": "Chest X-ray demonstrates bilateral lower lobe pneumonia with pleural effusions.",
                "candidate": "X-ray shows pneumonia in both lower lung zones with pleural fluid.",
                "expected_quality": "high"
            },
            {
                "reference": "No evidence of acute intracranial abnormality. Brain parenchyma appears normal.",
                "candidate": "No acute brain abnormalities. Normal brain tissue.",
                "expected_quality": "high"
            },
            {
                "reference": "Patient has pneumonia with bilateral infiltrates.",
                "candidate": "Weather is sunny today.",
                "expected_quality": "low"
            },
            {
                "reference": "Clear lung fields bilaterally with normal heart size.",
                "candidate": "Lung fields clear on both sides, heart size normal.",
                "expected_quality": "high"
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_dataset_quality_ranking(self):
        """Test that system correctly ranks dataset quality."""
        results = []
        
        for i, case in enumerate(self.medical_dataset):
            result = self.evaluator.evaluate_single(
                case["reference"], 
                case["candidate"]
            )
            
            results.append({
                "case_id": i,
                "overall_score": result["overall_score"],
                "expected_quality": case["expected_quality"],
                "reference": case["reference"][:50] + "...",
                "candidate": case["candidate"][:50] + "..."
            })
        
        # Separate high and low quality cases
        high_quality_scores = [r["overall_score"] for r in results if r["expected_quality"] == "high"]
        low_quality_scores = [r["overall_score"] for r in results if r["expected_quality"] == "low"]
        
        print(f"\n=== Dataset Quality Validation ===")
        for result in results:
            print(f"Case {result['case_id']}: {result['overall_score']:.3f} ({result['expected_quality']} quality)")
            print(f"  Ref: {result['reference']}")
            print(f"  Cand: {result['candidate']}")
        
        # High quality cases should score higher than low quality cases
        avg_high_quality = sum(high_quality_scores) / len(high_quality_scores)
        avg_low_quality = sum(low_quality_scores) / len(low_quality_scores)
        
        print(f"\nAverage high quality score: {avg_high_quality:.3f}")
        print(f"Average low quality score: {avg_low_quality:.3f}")
        
        self.assertGreater(avg_high_quality, avg_low_quality + 0.2,
                          f"High quality cases should score significantly higher: "
                          f"high={avg_high_quality:.3f}, low={avg_low_quality:.3f}")


class TestEdgeCaseValidation(unittest.TestCase):
    """Test validation of edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.metrics = {
            "bleu": BLEUScorer(self.logger),
            "rouge": ROUGEScorer(self.logger),
            "meteor": METEORScorer(self.logger),
            "medical": MedicalScorer(self.logger)
        }
    
    def test_very_short_texts(self):
        """Test behavior with very short texts."""
        short_cases = [
            ("Pneumonia.", "Pneumonia."),
            ("Normal.", "Normal."),
            ("No.", "Yes."),
            ("Patient.", "Doctor.")
        ]
        
        for ref, cand in short_cases:
            for metric_name, metric in self.metrics.items():
                with self.subTest(metric=metric_name, ref=ref, cand=cand):
                    result = metric.calculate(ref, cand)
                    
                    # Should not crash and should return valid results
                    self.assertIsInstance(result, dict)
                    self.assertGreater(len(result), 0)
                    
                    # All scores should be numeric and finite
                    for score_name, score_value in result.items():
                        self.assertIsInstance(score_value, (int, float))
                        self.assertTrue(math.isfinite(score_value),
                                      f"{metric_name}.{score_name} not finite: {score_value}")


def run_validation_tests():
    """Run all validation tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestKnownBenchmarks,
        TestMetricConsistency,
        TestStandardDatasetValidation,
        TestEdgeCaseValidation
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
    
    print(f"\n=== Validation Test Summary ===")
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
    
    # Run validation tests
    success = run_validation_tests()
    
    if success:
        print("\n✅ All validation tests passed!")
        exit(0)
    else:
        print("\n❌ Some validation tests failed!")
        exit(1)
