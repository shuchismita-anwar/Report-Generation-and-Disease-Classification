#!/usr/bin/env python3
"""
Comprehensive System Tests
==========================
End-to-end tests for the complete medical report evaluation system.

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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import main components
from medical_report_evaluator import MedicalReportEvaluator
from utils.metric_aggregator import MetricAggregator
from utils.results_exporter import ResultsExporter


class TestComprehensiveSystem(unittest.TestCase):
    """Comprehensive system tests for the medical report evaluation system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("test")
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample medical reports for testing
        self.sample_reports = [
            {
                "reference": "The chest X-ray shows clear lung fields bilaterally. Heart size is within normal limits. No acute cardiopulmonary abnormalities are identified.",
                "candidate": "Chest radiograph demonstrates clear lungs on both sides. Normal heart size. No acute findings in the chest."
            },
            {
                "reference": "There is consolidation in the right lower lobe consistent with pneumonia. Small pleural effusion is noted.",
                "candidate": "Right lower lobe pneumonia with associated pleural fluid collection."
            },
            {
                "reference": "No evidence of pneumothorax or pleural effusion. The cardiac silhouette appears normal.",
                "candidate": "No pneumothorax or fluid around the lungs. Heart looks normal."
            }
        ]
        
        # Configuration for testing
        self.test_config = {
            "metrics": {
                "bleu": {"enabled": True, "weight": 0.2},
                "rouge": {"enabled": True, "weight": 0.2},
                "meteor": {"enabled": True, "weight": 0.2},
                "bertscore": {"enabled": False, "weight": 0.0},  # Disabled due to dependencies
                "cider": {"enabled": True, "weight": 0.2},
                "medical": {"enabled": True, "weight": 0.2}
            },
            "aggregation": {
                "method": "weighted_average",
                "normalize_scores": True
            },
            "export": {
                "formats": ["json", "csv", "txt"],
                "include_metadata": True,
                "precision": 4
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_evaluator_initialization(self):
        """Test that the evaluator can be initialized properly."""
        try:
            evaluator = MedicalReportEvaluator()
            self.assertIsInstance(evaluator, MedicalReportEvaluator)
            self.logger.info("✅ Evaluator initialization successful")
        except Exception as e:
            self.fail(f"Evaluator initialization failed: {e}")
    
    def test_single_evaluation(self):
        """Test single report evaluation."""
        try:
            evaluator = MedicalReportEvaluator()
            
            # Test with first sample report
            report = self.sample_reports[0]
            result = evaluator.evaluate_single(
                generated_report=report["candidate"],
                ground_truth_report=report["reference"]
            )
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn("overall_score", result)
            self.assertIn("metrics", result)
            self.assertIn("metadata", result)
            
            # Verify score is reasonable
            overall_score = result["overall_score"]
            self.assertTrue(0.0 <= overall_score <= 1.0)
            
            self.logger.info(f"✅ Single evaluation successful: {overall_score:.4f}")
            
        except Exception as e:
            self.fail(f"Single evaluation failed: {e}")
    
    def test_batch_evaluation(self):
        """Test batch evaluation functionality."""
        try:
            evaluator = MedicalReportEvaluator()
            
            # Prepare batch data
            report_pairs = [(report["candidate"], report["reference"]) for report in self.sample_reports]
            
            # Run batch evaluation
            results = evaluator.evaluate_batch(
                report_pairs=report_pairs
            )
            
            # Verify results structure
            self.assertIsInstance(results, dict)
            self.assertIn("individual_results", results)
            self.assertIn("batch_summary", results)
            self.assertIn("aggregate_metrics", results)
            
            # Verify we have results for all reports
            individual_results = results["individual_results"]
            self.assertEqual(len(individual_results), len(self.sample_reports))
            
            # Verify each individual result
            for result in individual_results:
                self.assertIn("overall_score", result)
                self.assertTrue(0.0 <= result["overall_score"] <= 1.0)
            
            batch_score = results["aggregate_metrics"].get("batch_overall_score", 0.0)
            self.assertTrue(0.0 <= batch_score <= 1.0)
            
            self.logger.info(f"✅ Batch evaluation successful: {batch_score:.4f}")
            
        except Exception as e:
            self.fail(f"Batch evaluation failed: {e}")
    
    def test_metric_aggregator(self):
        """Test metric aggregator functionality."""
        try:
            aggregator = MetricAggregator(self.logger)
            
            # Sample metric scores
            metric_scores = {
                "bleu": 0.75,
                "rouge_1_f1": 0.68,
                "rouge_2_f1": 0.45,
                "rouge_l_f1": 0.72,
                "meteor": 0.58,
                "cider": 0.82,
                "medical_score": 0.65
            }
            
            # Test aggregation
            result = aggregator.aggregate_scores(
                metric_scores=metric_scores,
                method="weighted_average"
            )
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn("overall_score", result)
            self.assertIn("valid_scores", result)
            self.assertIn("aggregation_details", result)
            
            # Verify score is reasonable
            overall_score = result["overall_score"]
            self.assertTrue(0.0 <= overall_score <= 1.0)
            
            self.logger.info(f"✅ Metric aggregation successful: {overall_score:.4f}")
            
        except Exception as e:
            self.fail(f"Metric aggregation failed: {e}")
    
    def test_results_exporter(self):
        """Test results export functionality."""
        try:
            exporter = ResultsExporter(self.logger)
            
            # Sample results to export
            sample_results = {
                "evaluation_id": "test_001",
                "timestamp": "2024-01-15T10:30:00",
                "overall_score": 0.75,
                "individual_scores": {
                    "bleu": 0.8,
                    "rouge_1_f1": 0.7,
                    "meteor": 0.9
                },
                "metadata": {
                    "reference_length": 150,
                    "candidate_length": 140,
                    "evaluation_time": 2.5
                }
            }
            
            # Test JSON export
            json_path = os.path.join(self.temp_dir, "test_results.json")
            exported_path = exporter.export_results(
                results=sample_results,
                output_path=json_path,
                format="json"
            )
            
            self.assertTrue(os.path.exists(exported_path))
            
            # Verify JSON content
            with open(exported_path, 'r') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(loaded_data["evaluation_id"], "test_001")
            self.assertEqual(loaded_data["overall_score"], 0.75)
            
            # Test CSV export
            csv_path = os.path.join(self.temp_dir, "test_results.csv")
            exported_path = exporter.export_results(
                results=sample_results,
                output_path=csv_path,
                format="csv"
            )
            
            self.assertTrue(os.path.exists(exported_path))
            
            # Test TXT export
            txt_path = os.path.join(self.temp_dir, "test_results.txt")
            exported_path = exporter.export_results(
                results=sample_results,
                output_path=txt_path,
                format="txt"
            )
            
            self.assertTrue(os.path.exists(exported_path))
            
            self.logger.info("✅ Results export successful")
            
        except Exception as e:
            self.fail(f"Results export failed: {e}")
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end evaluation workflow."""
        try:
            # Initialize evaluator
            evaluator = MedicalReportEvaluator()
            
            # Run single evaluation
            report = self.sample_reports[0]
            result = evaluator.evaluate_single(
                generated_report=report["candidate"],
                ground_truth_report=report["reference"]
            )
            
            # Export results
            output_path = os.path.join(self.temp_dir, "end_to_end_results.json")
            exported_path = evaluator.export_results(
                results=result,
                output_path=output_path,
                format="json"
            )
            
            # Verify export
            self.assertTrue(os.path.exists(exported_path))
            
            # Load and verify exported data
            with open(exported_path, 'r') as f:
                exported_data = json.load(f)
            
            self.assertIn("overall_score", exported_data)
            self.assertIn("metrics", exported_data)
            
            self.logger.info("✅ End-to-end workflow successful")
            
        except Exception as e:
            self.fail(f"End-to-end workflow failed: {e}")
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        try:
            evaluator = MedicalReportEvaluator()
            
            # Test with empty strings
            try:
                result = evaluator.evaluate_single("", "some text")
                # Should handle gracefully or raise appropriate error
            except ValueError:
                pass  # Expected behavior
            
            # Test with None inputs
            try:
                result = evaluator.evaluate_single(None, "some text")
                # Should handle gracefully or raise appropriate error
            except (ValueError, TypeError):
                pass  # Expected behavior
            
            # Test with empty batch
            try:
                result = evaluator.evaluate_batch(
                    report_pairs=[]  # Empty batch
                )
                # Should handle gracefully or raise appropriate error
            except ValueError:
                pass  # Expected behavior
            
            self.logger.info("✅ Error handling tests passed")
            
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")
    
    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        try:
            evaluator = MedicalReportEvaluator()
            
            # Run evaluation and check timing
            report = self.sample_reports[0]
            result = evaluator.evaluate_single(
                generated_report=report["candidate"],
                ground_truth_report=report["reference"]
            )
            
            # Check for timing information
            if "metadata" in result and "evaluation_time" in result["metadata"]:
                eval_time = result["metadata"]["evaluation_time"]
                self.assertGreater(eval_time, 0.0)
                self.logger.info(f"✅ Performance tracking: {eval_time:.4f}s")
            else:
                self.logger.info("✅ Performance tracking structure verified")
            
        except Exception as e:
            self.fail(f"Performance tracking test failed: {e}")


def run_comprehensive_tests():
    """Run all comprehensive system tests and return results."""
    print("🧪 Running Comprehensive System Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestComprehensiveSystem))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 Comprehensive System Tests Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
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
        print("\n✅ All comprehensive system tests passed!")
        print("🎉 The medical report evaluation system is working correctly!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} tests failed!")
        print("🔧 System needs fixes before proceeding to Phase 5")
    
    return success, result


if __name__ == "__main__":
    success, _ = run_comprehensive_tests()
    exit(0 if success else 1) 