#!/usr/bin/env python3
"""
Comprehensive Metric Unit Tests
==============================
Complete unit tests for all metric implementations with edge cases,
error conditions, and boundary value testing.

Author: Medical Report Evaluation System
Version: 1.0
"""

import unittest
import sys
import os
import logging
import warnings
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all metrics
from metrics.bleu_scorer import BLEUScorer
from metrics.rouge_scorer import ROUGEScorer
from metrics.meteor_scorer import METEORScorer
from metrics.bertscore_scorer import BERTScoreScorer
from metrics.cider_scorer import CIDErScorer
from metrics.medical_scorer import MedicalScorer


class TestBLEUScorer(unittest.TestCase):
    """Comprehensive tests for BLEU Scorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.bleu = BLEUScorer(self.logger)
    
    def test_bleu_instantiation(self):
        """Test BLEU scorer instantiation."""
        self.assertEqual(self.bleu.get_name(), "BLEU")
        self.assertIn("bleu", self.bleu.get_description().lower())
        self.assertEqual(self.bleu.get_metric_type(), "n-gram")
    
    def test_identical_texts(self):
        """Test BLEU with identical texts."""
        text = "The patient has clear lung fields."
        result = self.bleu.calculate(text, text)
        
        # Should have high scores for identical texts
        self.assertIn("bleu_1", result)
        self.assertIn("bleu_2", result)
        self.assertIn("bleu_3", result)
        self.assertIn("bleu_4", result)
        
        # BLEU-1 should be 1.0 for identical texts
        self.assertAlmostEqual(result["bleu_1"], 1.0, places=2)
    
    def test_completely_different_texts(self):
        """Test BLEU with completely different texts."""
        ref = "The patient has clear lung fields."
        cand = "Elephant dancing in the moonlight."
        
        result = self.bleu.calculate(ref, cand)
        
        # Should have low scores for different texts
        self.assertLess(result["bleu_1"], 0.5)
        self.assertLess(result["bleu_4"], 0.1)
    
    def test_empty_texts(self):
        """Test BLEU with empty texts."""
        with self.assertRaises(ValueError):
            self.bleu.calculate("", "some text")
        
        with self.assertRaises(ValueError):
            self.bleu.calculate("some text", "")
    
    def test_whitespace_only_texts(self):
        """Test BLEU with whitespace-only texts."""
        with self.assertRaises(ValueError):
            self.bleu.calculate("   ", "some text")
        
        with self.assertRaises(ValueError):
            self.bleu.calculate("some text", "\t\n  ")
    
    def test_single_word_texts(self):
        """Test BLEU with single word texts."""
        result = self.bleu.calculate("pneumonia", "pneumonia")
        self.assertAlmostEqual(result["bleu_1"], 1.0, places=2)
        
        result = self.bleu.calculate("pneumonia", "infection")
        self.assertAlmostEqual(result["bleu_1"], 0.0, places=2)
    
    def test_case_sensitivity(self):
        """Test BLEU case sensitivity."""
        ref = "The Patient Has Clear Lung Fields."
        cand = "the patient has clear lung fields."
        
        result = self.bleu.calculate(ref, cand)
        # Should handle case differences well
        self.assertGreater(result["bleu_1"], 0.8)
    
    def test_punctuation_handling(self):
        """Test BLEU punctuation handling."""
        ref = "The patient has clear lung fields."
        cand = "The patient has clear lung fields!"
        
        result = self.bleu.calculate(ref, cand)
        # Should handle punctuation differences
        self.assertGreater(result["bleu_1"], 0.8)
    
    def test_smoothing_methods(self):
        """Test different smoothing methods."""
        ref = "The patient shows improvement."
        cand = "Patient shows some improvement."
        
        # Test with different smoothing methods
        config_add_one = {"smoothing_method": "add_one"}
        self.bleu.configure(config_add_one)
        result1 = self.bleu.calculate(ref, cand)
        
        config_epsilon = {"smoothing_method": "epsilon"}
        self.bleu.configure(config_epsilon)
        result2 = self.bleu.calculate(ref, cand)
        
        # Results should be different with different smoothing
        self.assertNotEqual(result1["bleu_4"], result2["bleu_4"])
    
    def test_boundary_values(self):
        """Test BLEU with boundary values."""
        # Very long text
        long_text = "word " * 1000
        result = self.bleu.calculate(long_text, long_text)
        self.assertAlmostEqual(result["bleu_1"], 1.0, places=2)
        
        # Very short but valid text
        result = self.bleu.calculate("a", "a")
        self.assertAlmostEqual(result["bleu_1"], 1.0, places=2)


class TestROUGEScorer(unittest.TestCase):
    """Comprehensive tests for ROUGE Scorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.rouge = ROUGEScorer(self.logger)
    
    def test_rouge_instantiation(self):
        """Test ROUGE scorer instantiation."""
        self.assertEqual(self.rouge.get_name(), "ROUGE")
        self.assertIn("rouge", self.rouge.get_description().lower())
        self.assertEqual(self.rouge.get_metric_type(), "n-gram")
    
    def test_rouge_calculation(self):
        """Test ROUGE calculation."""
        ref = "The patient has clear lung fields and normal heart size."
        cand = "Patient shows clear lungs and normal heart."
        
        result = self.rouge.calculate(ref, cand)
        
        # Check for all ROUGE variants
        self.assertIn("rouge_1_f", result)
        self.assertIn("rouge_1_p", result)
        self.assertIn("rouge_1_r", result)
        self.assertIn("rouge_2_f", result)
        self.assertIn("rouge_l_f", result)
        
        # Scores should be between 0 and 1
        for score in result.values():
            self.assertTrue(0.0 <= score <= 1.0)
    
    def test_identical_texts_rouge(self):
        """Test ROUGE with identical texts."""
        text = "The patient has clear lung fields."
        result = self.rouge.calculate(text, text)
        
        # Should have perfect scores for identical texts
        self.assertAlmostEqual(result["rouge_1_f"], 1.0, places=2)
        self.assertAlmostEqual(result["rouge_l_f"], 1.0, places=2)
    
    def test_empty_text_rouge(self):
        """Test ROUGE with empty texts."""
        with self.assertRaises(ValueError):
            self.rouge.calculate("", "some text")
        
        with self.assertRaises(ValueError):
            self.rouge.calculate("some text", "")
    
    def test_no_overlap(self):
        """Test ROUGE with no word overlap."""
        ref = "pneumonia consolidation infiltrate"
        cand = "elephant monkey zebra"
        
        result = self.rouge.calculate(ref, cand)
        self.assertAlmostEqual(result["rouge_1_f"], 0.0, places=2)
        self.assertAlmostEqual(result["rouge_2_f"], 0.0, places=2)
    
    def test_partial_overlap(self):
        """Test ROUGE with partial overlap."""
        ref = "The patient has pneumonia and fever."
        cand = "Patient shows pneumonia symptoms."
        
        result = self.rouge.calculate(ref, cand)
        self.assertGreater(result["rouge_1_f"], 0.0)
        self.assertLess(result["rouge_1_f"], 1.0)
    
    def test_lcs_calculation(self):
        """Test Longest Common Subsequence calculation."""
        # Test with known LCS
        ref = "A B C D E"
        cand = "A C E"
        
        result = self.rouge.calculate(ref, cand)
        # LCS should be detected
        self.assertGreater(result["rouge_l_f"], 0.0)
    
    def test_rouge_precision_recall(self):
        """Test ROUGE precision and recall calculation."""
        ref = "patient pneumonia"
        cand = "patient pneumonia infection"
        
        result = self.rouge.calculate(ref, cand)
        
        # Precision should be 2/3 (2 matching words out of 3 in candidate)
        # Recall should be 1.0 (all reference words matched)
        self.assertAlmostEqual(result["rouge_1_r"], 1.0, places=2)
        self.assertAlmostEqual(result["rouge_1_p"], 2/3, places=2)


class TestMETEORScorer(unittest.TestCase):
    """Comprehensive tests for METEOR Scorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.meteor = METEORScorer(self.logger)
    
    def test_meteor_instantiation(self):
        """Test METEOR scorer instantiation."""
        self.assertEqual(self.meteor.get_name(), "METEOR")
        self.assertIn("meteor", self.meteor.get_description().lower())
        self.assertEqual(self.meteor.get_metric_type(), "alignment")
    
    def test_meteor_calculation(self):
        """Test METEOR calculation."""
        ref = "The patient has clear lung fields."
        cand = "Patient shows clear lungs."
        
        result = self.meteor.calculate(ref, cand)
        
        # Check for METEOR score
        self.assertIn("meteor", result)
        self.assertTrue(0.0 <= result["meteor"] <= 1.0)
    
    def test_identical_texts_meteor(self):
        """Test METEOR with identical texts."""
        text = "The patient has pneumonia."
        result = self.meteor.calculate(text, text)
        
        # Should have perfect score for identical texts
        self.assertAlmostEqual(result["meteor"], 1.0, places=2)
    
    def test_empty_text_meteor(self):
        """Test METEOR with empty texts."""
        with self.assertRaises(ValueError):
            self.meteor.calculate("", "some text")
        
        with self.assertRaises(ValueError):
            self.meteor.calculate("some text", "")
    
    def test_stemming_functionality(self):
        """Test METEOR stemming functionality."""
        ref = "The patient is walking slowly."
        cand = "The patient walks slow."
        
        # Configure with stemming
        config = {"use_stemming": True}
        self.meteor.configure(config)
        
        result_with_stem = self.meteor.calculate(ref, cand)
        
        # Configure without stemming
        config = {"use_stemming": False}
        self.meteor.configure(config)
        
        result_without_stem = self.meteor.calculate(ref, cand)
        
        # Stemming should improve score
        self.assertGreaterEqual(result_with_stem["meteor"], result_without_stem["meteor"])
    
    def test_medical_synonyms(self):
        """Test medical synonym handling."""
        ref = "The patient has pneumonia."
        cand = "The patient has lung infection."
        
        # Configure with medical synonyms
        config = {"use_medical_synonyms": True}
        self.meteor.configure(config)
        
        result = self.meteor.calculate(ref, cand)
        self.assertGreater(result["meteor"], 0.0)
    
    @patch('nltk.translate.meteor_score.meteor_score')
    def test_meteor_with_mock(self, mock_meteor):
        """Test METEOR with mocked NLTK function."""
        mock_meteor.return_value = 0.75
        
        ref = "Test reference"
        cand = "Test candidate"
        
        result = self.meteor.calculate(ref, cand)
        self.assertEqual(result["meteor"], 0.75)
    
    def test_meteor_error_handling(self):
        """Test METEOR error handling."""
        # Test with special characters
        ref = "Patient has @@@ symptoms."
        cand = "Patient shows ### signs."
        
        # Should not crash
        result = self.meteor.calculate(ref, cand)
        self.assertIsInstance(result["meteor"], float)


class TestBERTScoreScorer(unittest.TestCase):
    """Comprehensive tests for BERTScore Scorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.bertscore = BERTScoreScorer(self.logger)
    
    def test_bertscore_instantiation(self):
        """Test BERTScore scorer instantiation."""
        self.assertEqual(self.bertscore.get_name(), "BERTScore")
        self.assertIn("bert", self.bertscore.get_description().lower())
        self.assertEqual(self.bertscore.get_metric_type(), "semantic")
    
    @patch('bert_score.score')
    def test_bertscore_calculation(self, mock_bert_score):
        """Test BERTScore calculation."""
        # Mock BERTScore return values
        mock_bert_score.return_value = (
            [0.85],  # Precision
            [0.80],  # Recall
            [0.82]   # F1
        )
        
        ref = "The patient has clear lung fields."
        cand = "Patient shows clear lungs."
        
        result = self.bertscore.calculate(ref, cand)
        
        # Check for all BERTScore components
        self.assertIn("bert_precision", result)
        self.assertIn("bert_recall", result)
        self.assertIn("bert_f1", result)
        
        # Check values
        self.assertEqual(result["bert_precision"], 0.85)
        self.assertEqual(result["bert_recall"], 0.80)
        self.assertEqual(result["bert_f1"], 0.82)
    
    @patch('bert_score.score')
    def test_bertscore_identical_texts(self, mock_bert_score):
        """Test BERTScore with identical texts."""
        mock_bert_score.return_value = ([1.0], [1.0], [1.0])
        
        text = "The patient has pneumonia."
        result = self.bertscore.calculate(text, text)
        
        # Should have perfect scores for identical texts
        self.assertEqual(result["bert_f1"], 1.0)
    
    def test_bertscore_fallback(self):
        """Test BERTScore fallback when library unavailable."""
        # This should use the fallback implementation
        ref = "The patient has pneumonia."
        cand = "Patient shows lung infection."
        
        result = self.bertscore.calculate(ref, cand)
        
        # Should still return valid scores
        self.assertIn("bert_precision", result)
        self.assertIn("bert_recall", result)
        self.assertIn("bert_f1", result)
        
        # Scores should be between 0 and 1
        for score in result.values():
            self.assertTrue(0.0 <= score <= 1.0)
    
    def test_bertscore_empty_text(self):
        """Test BERTScore with empty texts."""
        with self.assertRaises(ValueError):
            self.bertscore.calculate("", "some text")
        
        with self.assertRaises(ValueError):
            self.bertscore.calculate("some text", "")
    
    @patch('bert_score.score')
    def test_bertscore_model_configuration(self, mock_bert_score):
        """Test BERTScore model configuration."""
        mock_bert_score.return_value = ([0.75], [0.75], [0.75])
        
        # Configure with specific model
        config = {"model_type": "distilbert-base-uncased"}
        self.bertscore.configure(config)
        
        ref = "Patient has pneumonia."
        cand = "Patient shows infection."
        
        result = self.bertscore.calculate(ref, cand)
        
        # Should have called with specified model
        mock_bert_score.assert_called_once()
        self.assertEqual(result["bert_f1"], 0.75)


class TestCIDErScorer(unittest.TestCase):
    """Comprehensive tests for CIDEr Scorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.cider = CIDErScorer(self.logger)
    
    def test_cider_instantiation(self):
        """Test CIDEr scorer instantiation."""
        self.assertEqual(self.cider.get_name(), "CIDEr")
        self.assertIn("cider", self.cider.get_description().lower())
        self.assertEqual(self.cider.get_metric_type(), "consensus")
    
    def test_cider_calculation(self):
        """Test CIDEr calculation."""
        ref = "The patient has clear lung fields."
        cand = "Patient shows clear lungs."
        
        result = self.cider.calculate(ref, cand)
        
        # Check for CIDEr score
        self.assertIn("cider", result)
        self.assertIsInstance(result["cider"], float)
    
    def test_cider_identical_texts(self):
        """Test CIDEr with identical texts."""
        text = "The patient has pneumonia."
        result = self.cider.calculate(text, text)
        
        # Should have high score for identical texts
        self.assertGreater(result["cider"], 0.5)
    
    def test_multiple_references(self):
        """Test CIDEr with multiple references."""
        refs = [
            "The patient has pneumonia.",
            "Patient shows lung infection.",
            "Pneumonia detected in patient."
        ]
        cand = "Patient has lung infection."
        
        result = self.cider.calculate(refs, cand)
        
        # Should handle multiple references
        self.assertIn("cider", result)
        self.assertIsInstance(result["cider"], float)
    
    def test_empty_text_cider(self):
        """Test CIDEr with empty texts."""
        with self.assertRaises(ValueError):
            self.cider.calculate("", "some text")
        
        with self.assertRaises(ValueError):
            self.cider.calculate("some text", "")
    
    def test_tf_idf_weighting(self):
        """Test TF-IDF weighting in CIDEr."""
        # Common words should have lower weight
        ref = "the the the patient has pneumonia"
        cand = "the patient shows pneumonia"
        
        result = self.cider.calculate(ref, cand)
        
        # Should still produce valid score
        self.assertIn("cider", result)
        self.assertIsInstance(result["cider"], float)
    
    def test_cider_no_overlap(self):
        """Test CIDEr with no word overlap."""
        ref = "pneumonia consolidation infiltrate"
        cand = "elephant monkey zebra"
        
        result = self.cider.calculate(ref, cand)
        
        # Should have very low score
        self.assertLessEqual(result["cider"], 0.1)


class TestMedicalScorer(unittest.TestCase):
    """Comprehensive tests for Medical Scorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.medical = MedicalScorer(self.logger)
    
    def test_medical_instantiation(self):
        """Test Medical scorer instantiation."""
        self.assertEqual(self.medical.get_name(), "Medical")
        self.assertIn("medical", self.medical.get_description().lower())
        self.assertEqual(self.medical.get_metric_type(), "domain-specific")
    
    def test_medical_terminology_matching(self):
        """Test medical terminology matching."""
        ref = "Patient has pneumonia with bilateral infiltrates."
        cand = "Patient shows pneumonia and bilateral lung infiltrates."
        
        result = self.medical.calculate(ref, cand)
        
        # Check for medical score components
        self.assertIn("medical_terminology", result)
        self.assertIn("clinical_accuracy", result)
        self.assertIn("overall_medical", result)
        
        # Should have high medical terminology score
        self.assertGreater(result["medical_terminology"], 0.7)
    
    def test_negation_detection(self):
        """Test negation detection."""
        ref = "No evidence of pneumonia."
        cand = "Patient has pneumonia."
        
        result = self.medical.calculate(ref, cand)
        
        # Should detect negation mismatch
        self.assertLess(result["clinical_accuracy"], 0.5)
    
    def test_anatomical_terms(self):
        """Test anatomical terms matching."""
        ref = "Left upper lobe consolidation."
        cand = "Right upper lobe consolidation."
        
        result = self.medical.calculate(ref, cand)
        
        # Should detect anatomical difference
        self.assertLess(result["clinical_accuracy"], 0.8)
    
    def test_severity_assessment(self):
        """Test severity assessment."""
        ref = "Mild pneumonia."
        cand = "Severe pneumonia."
        
        result = self.medical.calculate(ref, cand)
        
        # Should detect severity difference
        self.assertLess(result["clinical_accuracy"], 0.7)
    
    def test_abbreviation_normalization(self):
        """Test medical abbreviation normalization."""
        ref = "Patient has CHF and COPD."
        cand = "Patient has congestive heart failure and chronic obstructive pulmonary disease."
        
        result = self.medical.calculate(ref, cand)
        
        # Should recognize abbreviations
        self.assertGreater(result["medical_terminology"], 0.8)
    
    def test_empty_text_medical(self):
        """Test Medical scorer with empty texts."""
        with self.assertRaises(ValueError):
            self.medical.calculate("", "some text")
        
        with self.assertRaises(ValueError):
            self.medical.calculate("some text", "")
    
    def test_non_medical_text(self):
        """Test Medical scorer with non-medical text."""
        ref = "The cat sat on the mat."
        cand = "A cat is sitting on a mat."
        
        result = self.medical.calculate(ref, cand)
        
        # Should have low medical scores
        self.assertLess(result["medical_terminology"], 0.3)
    
    def test_clinical_concept_extraction(self):
        """Test clinical concept extraction."""
        ref = "Patient presents with chest pain, shortness of breath, and fever."
        cand = "Patient has chest pain, dyspnea, and pyrexia."
        
        result = self.medical.calculate(ref, cand)
        
        # Should recognize medical synonyms
        self.assertGreater(result["medical_terminology"], 0.7)


class TestMetricIntegration(unittest.TestCase):
    """Test metric integration and consistency."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.metrics = {
            'bleu': BLEUScorer(self.logger),
            'rouge': ROUGEScorer(self.logger),
            'meteor': METEORScorer(self.logger),
            'bertscore': BERTScoreScorer(self.logger),
            'cider': CIDErScorer(self.logger),
            'medical': MedicalScorer(self.logger)
        }
    
    def test_all_metrics_basic_functionality(self):
        """Test all metrics basic functionality."""
        ref = "The patient has pneumonia with bilateral lung infiltrates."
        cand = "Patient shows pneumonia and bilateral lung infiltration."
        
        results = {}
        
        # Test all metrics
        for name, metric in self.metrics.items():
            try:
                result = metric.calculate(ref, cand)
                results[name] = result
                
                # Each metric should return a dictionary
                self.assertIsInstance(result, dict)
                
                # Each result should have at least one score
                self.assertGreater(len(result), 0)
                
                # All scores should be numeric
                for score in result.values():
                    self.assertIsInstance(score, (int, float))
                    
            except Exception as e:
                self.fail(f"Metric {name} failed: {e}")
        
        # All metrics should have produced results
        self.assertEqual(len(results), len(self.metrics))
    
    def test_metric_consistency(self):
        """Test metric consistency across multiple runs."""
        ref = "Patient has clear lung fields."
        cand = "Patient shows clear lungs."
        
        for name, metric in self.metrics.items():
            # Run multiple times
            results = []
            for _ in range(3):
                result = metric.calculate(ref, cand)
                results.append(result)
            
            # Results should be consistent
            first_result = results[0]
            for result in results[1:]:
                for key in first_result:
                    if key in result:
                        self.assertAlmostEqual(
                            first_result[key], 
                            result[key], 
                            places=3,
                            msg=f"Inconsistent results for {name}.{key}"
                        )
    
    def test_metric_performance_tracking(self):
        """Test metric performance tracking."""
        import time
        
        ref = "The patient presents with acute chest pain, shortness of breath, and diaphoresis."
        cand = "Patient has acute chest pain, dyspnea, and sweating."
        
        performance = {}
        
        for name, metric in self.metrics.items():
            start_time = time.time()
            
            # Run metric multiple times
            for _ in range(10):
                result = metric.calculate(ref, cand)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            performance[name] = avg_time
            
            # Each metric should complete in reasonable time (<1 second per call)
            self.assertLess(avg_time, 1.0, f"Metric {name} too slow: {avg_time:.3f}s")
        
        # Log performance results
        print("\n=== Metric Performance ===")
        for name, time_taken in sorted(performance.items(), key=lambda x: x[1]):
            print(f"{name:12s}: {time_taken:.4f}s per evaluation")


def run_comprehensive_metric_tests():
    """Run all comprehensive metric tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestBLEUScorer,
        TestROUGEScorer,
        TestMETEORScorer,
        TestBERTScoreScorer,
        TestCIDErScorer,
        TestMedicalScorer,
        TestMetricIntegration
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
    
    print(f"\n=== Test Summary ===")
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
    
    # Run comprehensive tests
    success = run_comprehensive_metric_tests()
    
    if success:
        print("\n✅ All comprehensive metric tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1) 