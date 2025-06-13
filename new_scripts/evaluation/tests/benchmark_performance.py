#!/usr/bin/env python3
"""
Performance Benchmarks
======================
Comprehensive performance benchmarks for the Medical Report Evaluation System.
Measures evaluation timing, monitors memory usage, identifies bottlenecks,
and tests with large datasets.

Author: Medical Report Evaluation System
Version: 1.0
"""

import unittest
import sys
import os
import json
import time
import tempfile
import shutil
import logging
import warnings
import gc
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from statistics import mean, median, stdev

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


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time


class TestSingleMetricPerformance(unittest.TestCase):
    """Test performance of individual metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.ERROR)  # Reduce logging overhead
        
        # Initialize all metrics
        self.metrics = {
            "bleu": BLEUScorer(self.logger),
            "rouge": ROUGEScorer(self.logger),
            "meteor": METEORScorer(self.logger),
            "bertscore": BERTScoreScorer(self.logger),
            "cider": CIDErScorer(self.logger),
            "medical": MedicalScorer(self.logger)
        }
        
        # Test data of varying lengths
        self.test_data = {
            "short": {
                "reference": "Patient has pneumonia.",
                "candidate": "Patient shows pneumonia."
            },
            "medium": {
                "reference": "Patient presents with acute chest pain and shortness of breath. Chest X-ray shows bilateral lung infiltrates consistent with pneumonia.",
                "candidate": "Patient has chest pain and dyspnea. X-ray reveals bilateral pneumonia infiltrates."
            },
            "long": {
                "reference": " ".join([
                    "Patient presents with acute onset chest pain radiating to left arm associated with diaphoresis and nausea.",
                    "Physical examination reveals elevated blood pressure and heart rate.",
                    "Electrocardiogram demonstrates ST elevation in leads II, III, and aVF consistent with inferior wall myocardial infarction.",
                    "Chest X-ray shows clear lung fields with normal cardiac silhouette.",
                    "Laboratory studies reveal elevated troponin levels confirming myocardial injury.",
                    "Patient was immediately started on dual antiplatelet therapy and anticoagulation.",
                    "Emergent cardiac catheterization revealed complete occlusion of right coronary artery.",
                    "Percutaneous coronary intervention was performed with successful restoration of flow."
                ]),
                "candidate": " ".join([
                    "Patient has acute chest pain extending to left arm with sweating and nausea.",
                    "Vital signs show high blood pressure and heart rate.",
                    "ECG shows ST elevation in inferior leads suggesting inferior STEMI.",
                    "Chest X-ray demonstrates clear lungs and normal heart size.",
                    "Blood tests show elevated cardiac enzymes indicating heart muscle damage.",
                    "Treatment initiated with antiplatelet medications and blood thinners.",
                    "Emergency heart catheterization showed blocked right coronary artery.",
                    "Balloon angioplasty successfully reopened the blocked vessel."
                ])
            }
        }
    
    def test_individual_metric_speed(self):
        """Test speed of individual metrics."""
        results = {}
        
        for metric_name, metric in self.metrics.items():
            metric_results = {}
            
            for data_size, data in self.test_data.items():
                times = []
                
                # Warm up
                for _ in range(3):
                    metric.calculate(data["reference"], data["candidate"])
                
                # Actual timing
                for _ in range(10):
                    with PerformanceTimer() as timer:
                        result = metric.calculate(data["reference"], data["candidate"])
                    times.append(timer.duration)
                    
                    # Verify result is valid
                    self.assertIsInstance(result, dict)
                    self.assertGreater(len(result), 0)
                
                metric_results[data_size] = {
                    "mean_time": mean(times),
                    "median_time": median(times),
                    "std_time": stdev(times) if len(times) > 1 else 0,
                    "min_time": min(times),
                    "max_time": max(times)
                }
            
            results[metric_name] = metric_results
        
        # Print results
        print(f"\n=== Individual Metric Performance ===")
        for metric_name, metric_results in results.items():
            print(f"\n{metric_name.upper()}:")
            for data_size, stats in metric_results.items():
                print(f"  {data_size:6s}: {stats['mean_time']:.4f}s ± {stats['std_time']:.4f}s "
                      f"(min: {stats['min_time']:.4f}s, max: {stats['max_time']:.4f}s)")
        
        # Performance assertions
        for metric_name, metric_results in results.items():
            # Each metric should complete in reasonable time
            for data_size, stats in metric_results.items():
                max_time = 2.0 if data_size == "long" else 1.0
                self.assertLess(stats['mean_time'], max_time, 
                              f"{metric_name} too slow for {data_size}: {stats['mean_time']:.3f}s")


class TestSystemPerformance(unittest.TestCase):
    """Test overall system performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create minimal configuration for speed
        config_path = os.path.join(self.test_dir, "config.json")
        config = {
            "metrics": {
                "weights": {
                    "bleu": 0.25,
                    "rouge": 0.25,
                    "meteor": 0.25,
                    "medical": 0.25
                    # Disable slower metrics for performance testing
                }
            },
            "evaluation": {
                "output_dir": self.test_dir,
                "save_individual_scores": False,
                "save_aggregated_scores": False
            },
            "logging": {
                "level": "ERROR"  # Reduce logging overhead
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        self.evaluator = MedicalReportEvaluator(config_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_single_evaluation_performance(self):
        """Test single evaluation performance."""
        # Test data
        reference = "Patient presents with acute chest pain and shortness of breath. Chest X-ray shows bilateral lung infiltrates consistent with pneumonia."
        candidate = "Patient has chest pain and dyspnea. X-ray reveals bilateral pneumonia infiltrates."
        
        times = []
        
        # Warm up
        for _ in range(5):
            self.evaluator.evaluate_single(reference, candidate)
        
        # Actual timing
        for _ in range(20):
            with PerformanceTimer() as timer:
                result = self.evaluator.evaluate_single(reference, candidate)
            times.append(timer.duration)
            
            # Verify result
            self.assertIsInstance(result, dict)
            self.assertIn("overall_score", result)
        
        # Calculate statistics
        mean_time = mean(times)
        median_time = median(times)
        std_time = stdev(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n=== Single Evaluation Performance ===")
        print(f"Mean time: {mean_time:.4f}s ± {std_time:.4f}s")
        print(f"Median time: {median_time:.4f}s")
        print(f"Range: {min_time:.4f}s - {max_time:.4f}s")
        
        # Performance assertions
        self.assertLess(mean_time, 0.1, f"Single evaluation too slow: {mean_time:.3f}s")
    
    def test_batch_evaluation_performance(self):
        """Test batch evaluation performance."""
        # Generate test data
        base_cases = [
            ("Patient has pneumonia.", "Patient shows pneumonia."),
            ("No abnormalities found.", "No abnormalities detected."),
            ("Bilateral infiltrates present.", "Bilateral lung infiltrates seen."),
            ("Clear lung fields.", "Lungs appear clear."),
            ("Heart size normal.", "Normal cardiac silhouette.")
        ]
        
        # Test different batch sizes
        batch_sizes = [10, 50, 100]
        
        for batch_size in batch_sizes:
            # Create batch
            references = []
            candidates = []
            case_ids = []
            
            for i in range(batch_size):
                ref, cand = base_cases[i % len(base_cases)]
                references.append(ref)
                candidates.append(cand)
                case_ids.append(f"case_{i:03d}")
            
            # Time batch evaluation
            with PerformanceTimer() as timer:
                results = self.evaluator.evaluate_batch(
                    references=references,
                    candidates=candidates,
                    case_ids=case_ids,
                    save_results=False
                )
            
            # Verify results
            self.assertEqual(len(results), batch_size)
            
            # Calculate per-case metrics
            per_case_time = timer.duration / batch_size
            
            print(f"\n=== Batch Size {batch_size} ===")
            print(f"Total time: {timer.duration:.2f}s")
            print(f"Per case: {per_case_time:.4f}s")
            print(f"Cases/second: {batch_size / timer.duration:.1f}")
            
            # Performance assertions
            self.assertLess(per_case_time, 0.05, 
                          f"Batch per-case time too slow: {per_case_time:.3f}s")


class TestBottleneckIdentification(unittest.TestCase):
    """Identify performance bottlenecks in the system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create configuration with all metrics enabled
        config_path = os.path.join(self.test_dir, "config.json")
        config = {
            "metrics": {
                "weights": {
                    "bleu": 0.16,
                    "rouge": 0.16,
                    "meteor": 0.16,
                    "bertscore": 0.17,
                    "cider": 0.17,
                    "medical": 0.18
                }
            },
            "evaluation": {"output_dir": self.test_dir},
            "logging": {"level": "ERROR"}
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        self.evaluator = MedicalReportEvaluator(config_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_identify_slowest_metrics(self):
        """Identify which metrics are the bottlenecks."""
        reference = "Patient presents with acute chest pain and shortness of breath. Chest X-ray shows bilateral lung infiltrates consistent with pneumonia."
        candidate = "Patient has chest pain and dyspnea. X-ray reveals bilateral pneumonia infiltrates."
        
        # Time each metric individually
        metric_times = {}
        
        for metric_name, metric in self.evaluator.metrics.items():
            times = []
            
            # Warm up
            for _ in range(3):
                metric.calculate(reference, candidate)
            
            # Time multiple runs
            for _ in range(10):
                with PerformanceTimer() as timer:
                    result = metric.calculate(reference, candidate)
                times.append(timer.duration)
            
            metric_times[metric_name] = {
                "mean": mean(times),
                "median": median(times),
                "std": stdev(times) if len(times) > 1 else 0
            }
        
        # Sort by mean time
        sorted_metrics = sorted(metric_times.items(), 
                              key=lambda x: x[1]["mean"], 
                              reverse=True)
        
        print(f"\n=== Metric Performance Ranking (Slowest First) ===")
        total_time = sum(stats["mean"] for _, stats in metric_times.items())
        
        for i, (metric_name, stats) in enumerate(sorted_metrics, 1):
            percentage = (stats["mean"] / total_time) * 100
            print(f"{i}. {metric_name:12s}: {stats['mean']:.4f}s ± {stats['std']:.4f}s "
                  f"({percentage:.1f}% of total)")
        
        print(f"\nTotal combined time: {total_time:.4f}s")
        
        # Identify bottlenecks
        slowest_metric = sorted_metrics[0]
        fastest_metric = sorted_metrics[-1]
        
        speed_ratio = slowest_metric[1]["mean"] / fastest_metric[1]["mean"]
        
        print(f"\nBottleneck Analysis:")
        print(f"Slowest: {slowest_metric[0]} ({slowest_metric[1]['mean']:.4f}s)")
        print(f"Fastest: {fastest_metric[0]} ({fastest_metric[1]['mean']:.4f}s)")
        print(f"Speed ratio: {speed_ratio:.1f}x")
        
        # Performance insights
        if slowest_metric[1]["mean"] > 0.1:
            print(f"WARNING: {slowest_metric[0]} is significantly slow!")
        
        if speed_ratio > 10:
            print(f"WARNING: Large performance variance between metrics!")


def run_performance_benchmarks():
    """Run all performance benchmarks."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSingleMetricPerformance,
        TestSystemPerformance,
        TestBottleneckIdentification
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
    
    print(f"\n=== Performance Benchmark Summary ===")
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
    
    # Run performance benchmarks
    print("Starting Performance Benchmarks...")
    print("This may take a few minutes to complete.")
    
    success = run_performance_benchmarks()
    
    if success:
        print("\n✅ All performance benchmarks passed!")
        exit(0)
    else:
        print("\n❌ Some performance benchmarks failed!")
        exit(1)
