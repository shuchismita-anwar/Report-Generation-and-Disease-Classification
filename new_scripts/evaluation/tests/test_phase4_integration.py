#!/usr/bin/env python3
"""
Phase 4 Integration Tests
========================
Comprehensive tests for Phase 4: Integration & Core Functionality

Tests:
- Single report evaluation
- Batch evaluation
- Metric aggregation
- Results export
- Error handling and recovery

Author: Medical Report Evaluation System
Version: 1.0
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add the evaluation directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_report_evaluator import MedicalReportEvaluator
from utils.metric_aggregator import MetricAggregator
from utils.results_exporter import ResultsExporter


class TestPhase4Integration:
    """Test suite for Phase 4 integration functionality."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_dir = None
        self.evaluator = None
        self.sample_reports = self._create_sample_reports()
        
    def setup(self):
        """Set up test environment."""
        # Create temporary directory for tests
        self.test_dir = tempfile.mkdtemp(prefix="eval_test_")
        print(f"Test directory: {self.test_dir}")
        
        # Create test config
        self._create_test_config()
        
        # Initialize evaluator
        config_path = os.path.join(self.test_dir, "test_config.json")
        self.evaluator = MedicalReportEvaluator(config_path=config_path)
        
        print("✅ Test setup completed")
    
    def teardown(self):
        """Clean up test environment."""
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        print("✅ Test cleanup completed")
    
    def test_single_evaluation(self):
        """Test single report evaluation."""
        print("\n🧪 Testing single report evaluation...")
        
        try:
            # Test with sample reports
            generated_report = self.sample_reports["generated"][0]
            ground_truth_report = self.sample_reports["ground_truth"][0]
            
            result = self.evaluator.evaluate_single(
                generated_report=generated_report,
                ground_truth_report=ground_truth_report,
                image_id="test_001"
            )
            
            # Validate result structure
            assert "image_id" in result
            assert "overall_score" in result
            assert "quality_level" in result
            assert "metrics" in result
            assert "timestamp" in result
            
            # Check that we have metric results
            assert len(result["metrics"]) > 0
            
            # Check overall score is reasonable
            assert 0.0 <= result["overall_score"] <= 1.0
            
            print(f"✅ Single evaluation successful - Score: {result['overall_score']:.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Single evaluation failed: {e}")
            return False
    
    def test_batch_evaluation(self):
        """Test batch report evaluation."""
        print("\n🧪 Testing batch report evaluation...")
        
        try:
            # Create report pairs
            report_pairs = list(zip(
                self.sample_reports["generated"],
                self.sample_reports["ground_truth"]
            ))
            
            image_ids = [f"test_{i:03d}" for i in range(len(report_pairs))]
            
            # Run batch evaluation
            batch_result = self.evaluator.evaluate_batch(
                report_pairs=report_pairs,
                image_ids=image_ids,
                save_intermediate=False
            )
            
            # Validate batch result structure
            assert "batch_id" in batch_result
            assert "batch_size" in batch_result
            assert "individual_results" in batch_result
            assert "batch_summary" in batch_result
            assert "aggregate_metrics" in batch_result
            
            # Check batch size matches
            assert batch_result["batch_size"] == len(report_pairs)
            
            # Check individual results
            assert len(batch_result["individual_results"]) <= len(report_pairs)
            
            # Check batch summary
            summary = batch_result["batch_summary"]
            assert "successful" in summary
            assert "failed" in summary
            assert "success_rate" in summary
            
            print(f"✅ Batch evaluation successful - {summary['successful']}/{summary['total_evaluated']} reports")
            return True
            
        except Exception as e:
            print(f"❌ Batch evaluation failed: {e}")
            return False
    
    def test_metric_aggregation(self):
        """Test metric aggregation functionality."""
        print("\n🧪 Testing metric aggregation...")
        
        try:
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
            
            # Validate aggregation result
            assert "overall_score" in result
            assert "method" in result
            assert "valid_scores" in result
            assert "aggregation_details" in result
            
            # Check score is reasonable
            assert 0.0 <= result["overall_score"] <= 1.0
            
            # Test different aggregation methods
            methods = ['simple_average', 'harmonic_mean', 'geometric_mean', 'median']
            for method in methods:
                method_result = aggregator.aggregate_scores(test_scores, method=method)
                assert 0.0 <= method_result["overall_score"] <= 1.0
            
            print(f"✅ Metric aggregation successful - Score: {result['overall_score']:.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Metric aggregation failed: {e}")
            return False
    
    def test_results_export(self):
        """Test results export functionality."""
        print("\n🧪 Testing results export...")
        
        try:
            exporter = ResultsExporter()
            
            # Create test result
            test_result = {
                "image_id": "test_export_001",
                "overall_score": 0.7234,
                "quality_level": "good",
                "metrics": {
                    "bleu": {"bleu": 0.45, "bleu_4": 0.3},
                    "rouge": {"rouge_1": 0.62, "rouge_l": 0.58},
                    "meteor": {"meteor": 0.51}
                },
                "timestamp": "2025-06-13T10:30:00Z"
            }
            
            # Test all export formats
            formats = ['json', 'csv', 'txt', 'html']
            exported_files = []
            
            for format in formats:
                output_path = os.path.join(self.test_dir, f"test_result.{format}")
                exported_path = exporter.export_results(
                    test_result, output_path, format=format
                )
                
                # Check file was created
                assert os.path.exists(exported_path)
                assert os.path.getsize(exported_path) > 0
                
                exported_files.append(exported_path)
            
            print(f"✅ Results export successful - {len(exported_files)} formats")
            return True
            
        except Exception as e:
            print(f"❌ Results export failed: {e}")
            return False
    
    def test_error_handling(self):
        """Test error handling and recovery."""
        print("\n🧪 Testing error handling...")
        
        try:
            # Test with invalid inputs
            test_cases = [
                ("", "valid ground truth"),  # Empty generated report
                ("valid generated", ""),     # Empty ground truth
                (None, "valid ground truth"), # None generated report
                ("valid generated", None),   # None ground truth
            ]
            
            error_count = 0
            for generated, ground_truth in test_cases:
                try:
                    self.evaluator.evaluate_single(generated, ground_truth)
                    print(f"⚠️ Expected error for inputs: {generated}, {ground_truth}")
                except (ValueError, TypeError):
                    error_count += 1
                    # Expected error
                    pass
                except Exception as e:
                    print(f"⚠️ Unexpected error type: {e}")
            
            # Test batch with mixed valid/invalid pairs
            mixed_pairs = [
                ("Valid generated report about chest X-ray findings.", "Valid ground truth report."),
                ("", "Valid ground truth"),  # This should fail
                ("Another valid generated report.", "Another valid ground truth."),
            ]
            
            batch_result = self.evaluator.evaluate_batch(
                report_pairs=mixed_pairs,
                save_intermediate=False
            )
            
            # Should have some successful and some failed
            assert batch_result["batch_summary"]["successful"] > 0
            assert batch_result["batch_summary"]["failed"] > 0
            
            print(f"✅ Error handling successful - {error_count} errors caught properly")
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            return False
    
    def test_integration_workflow(self):
        """Test complete integration workflow."""
        print("\n🧪 Testing complete integration workflow...")
        
        try:
            # Step 1: Single evaluation
            generated_report = self.sample_reports["generated"][0]
            ground_truth_report = self.sample_reports["ground_truth"][0]
            
            single_result = self.evaluator.evaluate_single(
                generated_report=generated_report,
                ground_truth_report=ground_truth_report,
                image_id="workflow_test_001"
            )
            
            # Step 2: Export single result
            single_export_path = self.evaluator.export_results(
                single_result, format='json'
            )
            assert os.path.exists(single_export_path)
            
            # Step 3: Batch evaluation
            report_pairs = list(zip(
                self.sample_reports["generated"][:3],
                self.sample_reports["ground_truth"][:3]
            ))
            
            batch_result = self.evaluator.evaluate_batch(
                report_pairs=report_pairs,
                save_intermediate=False
            )
            
            # Step 4: Export batch results in multiple formats
            for format in ['json', 'csv', 'txt']:
                batch_export_path = self.evaluator.export_results(
                    batch_result, format=format
                )
                assert os.path.exists(batch_export_path)
            
            print("✅ Complete integration workflow successful")
            return True
            
        except Exception as e:
            print(f"❌ Integration workflow failed: {e}")
            return False
    
    def _create_sample_reports(self):
        """Create sample reports for testing."""
        return {
            "generated": [
                "The chest X-ray shows clear lungs with no acute findings. Heart size is normal. No pleural effusion or pneumothorax is seen.",
                "Bilateral lower lobe opacities consistent with pneumonia. Heart size is mildly enlarged. Small bilateral pleural effusions.",
                "Normal chest radiograph. Lungs are clear. Heart and mediastinum are normal. No acute cardiopulmonary process.",
                "Left lower lobe consolidation suggestive of pneumonia. Right lung is clear. Heart size is normal.",
                "Cardiomegaly with pulmonary edema. Bilateral pleural effusions. Findings consistent with congestive heart failure."
            ],
            "ground_truth": [
                "Normal chest X-ray. Clear lungs bilaterally. Normal heart size and mediastinal contours. No acute findings.",
                "Bilateral pneumonia with lower lobe predominance. Mild cardiomegaly. Bilateral pleural effusions present.",
                "No acute cardiopulmonary abnormality. Lungs are clear. Normal heart size and mediastinal silhouette.",
                "Left lower lobe pneumonia. Right lung clear. Normal cardiac silhouette. No pleural effusion.",
                "Severe cardiomegaly with acute pulmonary edema. Bilateral pleural effusions. Findings of acute heart failure."
            ]
        }
    
    def _create_test_config(self):
        """Create test configuration file."""
        config = {
            "evaluation_settings": {
                "version": "1.0",
                "description": "Test Configuration"
            },
            "paths": {
                "evaluation_results": os.path.join(self.test_dir, "results"),
                "logs": os.path.join(self.test_dir, "logs")
            },
            "metrics": {
                "weights": {
                    "bleu": 0.2,
                    "rouge_l": 0.2,
                    "meteor": 0.2,
                    "bert_score": 0.2,
                    "medical": 0.2
                },
                "enabled": {
                    "bleu": True,
                    "rouge": True,
                    "meteor": True,
                    "bert_score": False,  # Disable to avoid model loading in tests
                    "cider": True,
                    "medical": True
                },
                "parameters": {
                    "bleu": {"max_n": 4, "smoothing": True},
                    "rouge": {"rouge_types": ["rouge1", "rougeL"]},
                    "meteor": {"alpha": 0.9}
                }
            },
            "thresholds": {
                "quality_levels": {
                    "excellent": 0.8,
                    "good": 0.65,
                    "fair": 0.5,
                    "poor": 0.35
                }
            },
            "output_formats": {
                "precision": 4,
                "default": "json"
            },
            "processing": {
                "batch_size": 10,
                "timeout_seconds": 300
            },
            "logging": {
                "level": "INFO",
                "console_output": True
            }
        }
        
        # Create directories
        os.makedirs(config["paths"]["evaluation_results"], exist_ok=True)
        os.makedirs(config["paths"]["logs"], exist_ok=True)
        
        # Save config
        config_path = os.path.join(self.test_dir, "test_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def run_all_tests(self):
        """Run all Phase 4 integration tests."""
        print("🚀 Starting Phase 4 Integration Tests")
        print("=" * 50)
        
        self.setup()
        
        tests = [
            ("Single Evaluation", self.test_single_evaluation),
            ("Batch Evaluation", self.test_batch_evaluation),
            ("Metric Aggregation", self.test_metric_aggregation),
            ("Results Export", self.test_results_export),
            ("Error Handling", self.test_error_handling),
            ("Integration Workflow", self.test_integration_workflow),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                failed += 1
        
        self.teardown()
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 All Phase 4 integration tests passed!")
        else:
            print(f"⚠️ {failed} tests failed. Please review the issues above.")
        
        return failed == 0


def main():
    """Main test runner."""
    test_suite = TestPhase4Integration()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n✅ Phase 4 integration is ready!")
        return 0
    else:
        print("\n❌ Phase 4 integration needs fixes!")
        return 1


if __name__ == "__main__":
    exit(main()) 