#!/usr/bin/env python3
"""
Individual Metrics Tests
=======================
Unit tests for all individual metric implementations.

Author: Medical Report Evaluation System
Version: 1.0
"""

import unittest
import sys
import os
import logging
import warnings
from unittest.mock import Mock, patch, MagicMock

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
    """Test cases for BLEU Scorer."""
    
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


class TestROUGEScorer(unittest.TestCase):
    """Test cases for ROUGE Scorer."""
    
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
    
    def test_lcs_calculation(self):
        """Test Longest Common Subsequence calculation."""
        # Test with known LCS
        ref = "ABCD"
        cand = "ACBD"
        
        # Should find common subsequence
        result = self.rouge.calculate(ref, cand)
        self.assertGreater(result["rouge_l_f"], 0.0)


class TestMETEORScorer(unittest.TestCase):
    """Test cases for METEOR Scorer."""
    
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


class TestBERTScoreScorer(unittest.TestCase):
    """Test cases for BERTScore Scorer."""
    
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
        """Test BERTScore calculation with mocked bert_score."""
        # Mock the bert_score function
        mock_bert_score.return_value = (
            [0.85],  # precision
            [0.80],  # recall
            [0.82]   # f1
        )
        
        ref = "The patient has clear lung fields."
        cand = "Patient shows clear lungs."
        
        result = self.bertscore.calculate(ref, cand)
        
        # Check for BERTScore components
        self.assertIn("bertscore_precision", result)
        self.assertIn("bertscore_recall", result)
        self.assertIn("bertscore_f1", result)
        
        self.assertEqual(result["bertscore_precision"], 0.85)
        self.assertEqual(result["bertscore_recall"], 0.80)
        self.assertEqual(result["bertscore_f1"], 0.82)
    
    def test_bertscore_fallback(self):
        """Test BERTScore fallback when library not available."""
        # This will test the actual implementation's fallback mechanism
        ref = "Test reference"
        cand = "Test candidate"
        
        try:
            result = self.bertscore.calculate(ref, cand)
            # If it works, check the results
            self.assertIn("bertscore_f1", result)
        except ImportError:
            # If bert_score is not available, this is expected
            pass


class TestCIDErScorer(unittest.TestCase):
    """Test cases for CIDEr Scorer."""
    
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
        ref = "The patient has clear lung fields and normal heart size."
        cand = "Patient shows clear lungs and normal heart."
        
        result = self.cider.calculate(ref, cand)
        
        # Check for CIDEr score
        self.assertIn("cider", result)
        self.assertIsInstance(result["cider"], float)
    
    def test_multiple_references(self):
        """Test CIDEr with multiple references."""
        refs = [
            "The patient has clear lung fields.",
            "Lung fields are clear bilaterally.",
            "Normal chest radiograph."
        ]
        cand = "Patient shows clear lungs."
        
        # Test with multiple references
        result = self.cider.calculate(refs, cand)
        self.assertIn("cider", result)
    
    def test_tf_idf_weighting(self):
        """Test TF-IDF weighting in CIDEr."""
        # Test with common vs rare words
        common_ref = "The patient has the condition."
        rare_ref = "The patient has pneumothorax."
        cand = "The patient has pneumothorax."
        
        result1 = self.cider.calculate(common_ref, cand)
        result2 = self.cider.calculate(rare_ref, cand)
        
        # Rare word match should have higher score
        self.assertGreaterEqual(result2["cider"], result1["cider"])


class TestMedicalScorer(unittest.TestCase):
    """Test cases for Medical Scorer."""
    
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
        ref = "The patient has pneumonia and pleural effusion."
        cand = "Patient shows lung infection and fluid in pleura."
        
        result = self.medical.calculate(ref, cand)
        
        # Check for medical-specific scores
        self.assertIn("medical_score", result)
        self.assertIn("terminology_match", result)
        self.assertIn("clinical_accuracy", result)
        
        # Scores should be between 0 and 1
        for score in result.values():
            if isinstance(score, float):
                self.assertTrue(0.0 <= score <= 1.0)
    
    def test_negation_detection(self):
        """Test negation detection."""
        ref = "No evidence of pneumonia."
        cand = "Pneumonia is not present."
        
        result = self.medical.calculate(ref, cand)
        
        # Should detect similar negation patterns
        self.assertGreater(result["medical_score"], 0.5)
    
    def test_anatomical_terms(self):
        """Test anatomical term matching."""
        ref = "Right upper lobe consolidation."
        cand = "Consolidation in the right upper lung lobe."
        
        result = self.medical.calculate(ref, cand)
        
        # Should match anatomical terms
        self.assertGreater(result["terminology_match"], 0.0)
    
    def test_severity_assessment(self):
        """Test severity level matching."""
        ref = "Severe pneumonia with complications."
        cand = "Serious lung infection with complications."
        
        result = self.medical.calculate(ref, cand)
        
        # Should recognize severity levels
        self.assertIn("severity_match", result)
    
    def test_abbreviation_normalization(self):
        """Test medical abbreviation normalization."""
        ref = "Patient has COPD and CHF."
        cand = "Patient has chronic obstructive pulmonary disease and congestive heart failure."
        
        result = self.medical.calculate(ref, cand)
        
        # Should match abbreviated and full forms
        self.assertGreater(result["medical_score"], 0.3)


class TestMetricIntegration(unittest.TestCase):
    """Test integration between different metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.metrics = [
            BLEUScorer(self.logger),
            ROUGEScorer(self.logger),
            METEORScorer(self.logger),
            BERTScoreScorer(self.logger),
            CIDErScorer(self.logger),
            MedicalScorer(self.logger)
        ]
    
    def test_all_metrics_basic_functionality(self):
        """Test that all metrics can perform basic calculations."""
        ref = "The patient has clear lung fields and normal heart size."
        cand = "Patient shows clear lungs and normal heart."
        
        for metric in self.metrics:
            try:
                result = metric.calculate(ref, cand)
                self.assertIsInstance(result, dict)
                self.assertGreater(len(result), 0)
                
                # All scores should be numeric
                for score in result.values():
                    self.assertIsInstance(score, (int, float))
                    
            except ImportError as e:
                # Some metrics might not have dependencies installed
                self.logger.warning(f"Skipping {metric.get_name()} due to missing dependency: {e}")
                continue
    
    def test_metric_consistency(self):
        """Test that metrics produce consistent results."""
        ref = "Normal chest X-ray."
        cand = "Normal chest X-ray."
        
        for metric in self.metrics:
            try:
                result1 = metric.calculate(ref, cand)
                result2 = metric.calculate(ref, cand)
                
                # Results should be identical for same inputs
                self.assertEqual(result1, result2)
                
            except ImportError:
                continue
    
    def test_metric_performance_tracking(self):
        """Test that all metrics track performance."""
        ref = "Test reference"
        cand = "Test candidate"
        
        for metric in self.metrics:
            try:
                initial_count = metric.calculation_count
                metric.calculate_with_timing(ref, cand)
                
                # Count should increase
                self.assertEqual(metric.calculation_count, initial_count + 1)
                
                # Should have performance stats
                stats = metric.get_performance_stats()
                self.assertIn("calculation_count", stats)
                self.assertIn("total_time", stats)
                
            except ImportError:
                continue


def run_individual_metrics_tests():
    """Run all individual metrics tests and return results."""
    print("🧪 Running Individual Metrics Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
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
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 Individual Metrics Tests Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {result.testsRun - len(result.failures) - len(result.errors) - result.testsRun}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}")
            print(f"     {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}")
            error_msg = traceback.split('\n')[-2] if '\n' in traceback else traceback
            print(f"     {error_msg}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ All individual metrics tests passed!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} tests failed!")
    
    return success, result


if __name__ == "__main__":
    success, _ = run_individual_metrics_tests()
    exit(0 if success else 1) 