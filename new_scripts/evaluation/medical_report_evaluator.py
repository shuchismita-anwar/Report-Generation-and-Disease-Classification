#!/usr/bin/env python3
"""
Medical Report Evaluator - Main Evaluation Class
===============================================
Comprehensive evaluation system for LLM-generated medical reports
using multiple metrics (BLEU, ROUGE, METEOR, BERTScore, medical-specific metrics).

Author: Medical Report Evaluation System
Version: 1.0
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import time
from datetime import datetime
import traceback

# Import all metric classes
from metrics.base_metric import BaseMetric
from metrics.bleu_scorer import BLEUScorer
from metrics.rouge_scorer import ROUGEScorer
from metrics.meteor_scorer import METEORScorer
from metrics.chexpert_scorer import ChexpertScorer
from metrics.radgraph_f1_scorer import RadGraphF1Scorer
from metrics.medical_scorer import MedicalScorer

# Import utilities
from utils.metric_aggregator import MetricAggregator
from utils.results_exporter import ResultsExporter


class MedicalReportEvaluator:
    """
    Main class for evaluating LLM-generated medical reports against ground truth.
    
    This class orchestrates the entire evaluation pipeline including:
    - Loading and managing ground truth data
    - Computing multiple evaluation metrics
    - Aggregating results and generating reports
    - Exporting results in various formats
    """
    
    def __init__(self, config_path: str = "config/evaluation_config.json", log_level: int = logging.INFO):
        """
        Initialize the Medical Report Evaluator.
        
        Args:
            config_path (str): Path to the configuration file
            log_level (int): Logging level (default: INFO)
        """
        self.config_path = config_path
        self.config = None
        self.ground_truth_data = {}
        self.metrics = {}
        self.logger = None
        self.evaluation_session_id = None
        
        # Initialize logging
        self._setup_logging(log_level)
        
        # Load configuration
        self._load_configuration()
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Initialize metric aggregator
        self.aggregator = MetricAggregator(logger=self.logger)
        
        # Initialize results exporter
        self.exporter = ResultsExporter(logger=self.logger)
        
        # Initialize session
        self.evaluation_session_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.logger:
            self.logger.info(f"Initialized Medical Report Evaluator - Session: {self.evaluation_session_id}")
            self.logger.info(f"Loaded {len(self.metrics)} metrics: {list(self.metrics.keys())}")

    def _setup_logging(self, log_level: int) -> None:
        """
        Set up logging configuration.
        
        Args:
            log_level (int): Logging level
        """
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Create logger
        self.logger = logging.getLogger("MedicalReportEvaluator")
        self.logger.setLevel(log_level)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = os.path.join(logs_dir, "evaluation.log")
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            # Fallback to basic file handler if rotation fails
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
        
        self.logger.info("Logging system initialized successfully")
    
    def _load_configuration(self) -> None:
        """
        Load and validate configuration from JSON file.
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is malformed
            ValueError: If required configuration sections are missing
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Malformed configuration file: {e}")
        
        # Validate required sections
        required_sections = [
            "evaluation_settings", "paths", "metrics", "thresholds",
            "output_formats", "processing", "ground_truth", "logging"
        ]
        
        missing_sections = [section for section in required_sections if section not in self.config]
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")
        
        if self.logger:
            self.logger.info(f"Configuration loaded successfully from {self.config_path}")

    def _initialize_metrics(self) -> None:
        """
        Initialize all evaluation metrics based on configuration.
        """
        self.metrics = {}
        
        if not self.config or 'metrics' not in self.config:
            self.logger.warning("No metrics configuration found, using defaults")
            return
        
        enabled_metrics = self.config['metrics'].get('enabled', {})
        metric_params = self.config['metrics'].get('parameters', {})
        
        try:
            # Initialize BLEU scorer
            if enabled_metrics.get('bleu', True):
                bleu_config = metric_params.get('bleu', {})
                self.metrics['bleu'] = BLEUScorer(logger=self.logger)
                self.metrics['bleu'].configure(bleu_config)
                self.logger.debug("Initialized BLEU scorer")
            
            # Initialize ROUGE scorer
            if enabled_metrics.get('rouge', True):
                rouge_config = metric_params.get('rouge', {})
                self.metrics['rouge'] = ROUGEScorer(logger=self.logger)
                self.metrics['rouge'].configure(rouge_config)
                self.logger.debug("Initialized ROUGE scorer")
            
            # Initialize METEOR scorer
            if enabled_metrics.get('meteor', True):
                meteor_config = metric_params.get('meteor', {})
                self.metrics['meteor'] = METEORScorer(logger=self.logger)
                self.metrics['meteor'].configure(meteor_config)
                self.logger.debug("Initialized METEOR scorer")
            
            # Initialize CheXpert scorer (pathology accuracy)
            if enabled_metrics.get('chexpert', True) or enabled_metrics.get('clinical_accuracy', True):
                chexpert_config = metric_params.get('chexpert', {})
                self.metrics['chexpert'] = ChexpertScorer(logger=self.logger)
                self.metrics['chexpert'].configure(chexpert_config)
                self.logger.debug("Initialized CheXpert scorer")
            
            # Initialize RadGraph F1 scorer (relation correctness)
            if enabled_metrics.get('radgraph_f1', True):
                radgraph_config = metric_params.get('radgraph_f1', {})
                self.metrics['radgraph_f1'] = RadGraphF1Scorer(logger=self.logger)
                self.metrics['radgraph_f1'].configure(radgraph_config)
                self.logger.debug("Initialized RadGraph F1 scorer")
            
            # Initialize Medical scorer
            if enabled_metrics.get('medical_terminology', True):
                medical_config = metric_params.get('medical', {})
                # Add medical-specific configuration from config
                if 'medical_specific' in self.config:
                    medical_config.update(self.config['medical_specific'])
                self.metrics['medical'] = MedicalScorer(logger=self.logger)
                self.metrics['medical'].configure(medical_config)
                self.logger.debug("Initialized Medical scorer")
            
            self.logger.info(f"Successfully initialized {len(self.metrics)} metrics")
            
        except Exception as e:
            self.logger.error(f"Error initializing metrics: {e}")
            self.logger.debug(f"Metrics initialization traceback: {traceback.format_exc()}")
            raise

    def load_ground_truth(self, source_dirs: Optional[List[str]] = None, 
                         force_reload: bool = False) -> Dict:
        """
        Load ground truth radiologist reports from MIMIC dataset.
        
        Args:
            source_dirs (Optional[List[str]]): List of directories to search for ground truth.
                                             If None, uses config defaults.
            force_reload (bool): Force reload even if already loaded
            
        Returns:
            Dict: Dictionary mapping image IDs to ground truth reports
            
        Raises:
            FileNotFoundError: If ground truth directories don't exist
            ValueError: If no valid ground truth reports found
        """
        # Implementation will be added in Phase 2
        pass
    
    def evaluate_single(self, generated_report: str, ground_truth_report: str,
                       image_id: str = None, metadata: Dict = None) -> Dict:
        """
        Evaluate a single generated report against ground truth.
        
        Args:
            generated_report (str): The LLM-generated medical report
            ground_truth_report (str): The radiologist ground truth report
            image_id (str, optional): Identifier for the image/case
            metadata (Dict, optional): Additional metadata for the evaluation
            
        Returns:
            Dict: Comprehensive evaluation results with all metric scores
            
        Example:
            {
                "image_id": "002da0d9-ce49c30d-...",
                "metrics": {
                    "bleu": {"score": 0.45, "details": {...}},
                    "rouge": {"rouge_1": 0.62, "rouge_2": 0.34, "rouge_l": 0.58},
                    "meteor": {"score": 0.51},
                    "bert_score": {"precision": 0.78, "recall": 0.72, "f1": 0.75}
                },
                "overall_score": 0.67,
                "quality_level": "good",
                "timestamp": "2025-06-13T10:30:00Z"
            }
        """
        start_time = time.time()
        
        # Validate inputs
        if not generated_report or not ground_truth_report:
            raise ValueError("Both generated_report and ground_truth_report must be non-empty strings")
        
        if not generated_report.strip() or not ground_truth_report.strip():
            raise ValueError("Reports cannot be empty or contain only whitespace")
        
        # Initialize result structure
        result = {
            "image_id": image_id or f"eval_{int(time.time())}",
            "metrics": {},
            "metric_details": {},
            "performance": {},
            "overall_score": 0.0,
            "quality_level": "unknown",
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.logger.info(f"Evaluating single report - ID: {result['image_id']}")
        
        # Calculate individual metrics
        metric_scores = {}
        metric_weights = self.config.get('metrics', {}).get('weights', {})
        
        for metric_name, metric_instance in self.metrics.items():
            try:
                self.logger.debug(f"Calculating {metric_name} metric")
                
                # Calculate metric with timing
                scores, calc_time = metric_instance.calculate_with_timing(
                    ground_truth_report, generated_report
                )
                
                # Store results
                result["metrics"][metric_name] = scores
                result["performance"][metric_name] = {
                    "calculation_time": calc_time,
                    "status": "success"
                }
                
                # Extract primary score for aggregation
                primary_score = self._extract_primary_score(metric_name, scores)
                if primary_score is not None:
                    metric_scores[metric_name] = primary_score
                
                self.logger.debug(f"{metric_name} completed: {primary_score:.4f} in {calc_time:.4f}s")
                
            except Exception as e:
                self.logger.error(f"Error calculating {metric_name}: {e}")
                result["metrics"][metric_name] = {"error": str(e)}
                result["performance"][metric_name] = {
                    "calculation_time": 0.0,
                    "status": "error",
                    "error": str(e)
                }
                # Continue with other metrics
        
        # Calculate overall score
        result["overall_score"] = self._calculate_overall_score(metric_scores, metric_weights)
        
        # Determine quality level
        result["quality_level"] = self._determine_quality_level(result["overall_score"])
        
        # Add evaluation metadata
        total_time = time.time() - start_time
        result["evaluation_time"] = total_time
        result["session_id"] = self.evaluation_session_id
        result["evaluator_version"] = "1.0"
        
        self.logger.info(f"Evaluation completed - Overall Score: {result['overall_score']:.4f}, "
                        f"Quality: {result['quality_level']}, Time: {total_time:.4f}s")
        
        return result
    
    def _extract_primary_score(self, metric_name: str, scores: Dict[str, float]) -> Optional[float]:
        """
        Extract the primary score from metric results for aggregation.
        
        Args:
            metric_name (str): Name of the metric
            scores (Dict[str, float]): Metric scores dictionary
            
        Returns:
            Optional[float]: Primary score for aggregation
        """
        if not scores or isinstance(scores, dict) and "error" in scores:
            return None
        
        # Define primary score extraction logic for each metric
        if metric_name == "bleu":
            return scores.get("bleu_4", scores.get("bleu", 0.0))
        elif metric_name == "rouge":
            # Use ROUGE-L as primary score, fallback to ROUGE-1
            return scores.get("rouge_l", scores.get("rouge_1", 0.0))
        elif metric_name == "meteor":
            return scores.get("meteor", scores.get("score", 0.0))
        elif metric_name == "chexpert":
            # Use CheXpert F1 as primary score for pathology accuracy
            return scores.get("chexpert_f1", scores.get("f1", 0.0))
        elif metric_name == "radgraph_f1":
            return scores.get("radgraph_f1", scores.get("f1", 0.0))
        elif metric_name == "medical":
            return scores.get("overall_score", scores.get("medical_score", 0.0))
        else:
            # Generic fallback - try common score field names
            for field in ["score", "f1", "overall", "primary"]:
                if field in scores:
                    return scores[field]
            # If all else fails, return the first numeric value
            for value in scores.values():
                if isinstance(value, (int, float)):
                    return float(value)
        
        return 0.0
    
    def _calculate_overall_score(self, metric_scores: Dict[str, float], 
                                metric_weights: Dict[str, float]) -> float:
        """
        Calculate weighted overall score from individual metric scores.
        
        Args:
            metric_scores (Dict[str, float]): Individual metric scores
            metric_weights (Dict[str, float]): Weights for each metric
            
        Returns:
            float: Weighted overall score
        """
        if not metric_scores:
            return 0.0
        
        # Handle ROUGE sub-metrics separately
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_name, score in metric_scores.items():
            # Map metric names to weight keys
            weight_key = metric_name
            if metric_name == "rouge":
                # For ROUGE, we might want to use rouge_l weight or a general rouge weight
                weight_key = "rouge_l"  # or could be split into rouge_1, rouge_2, rouge_l
            elif metric_name == "medical":
                weight_key = "medical_terminology"
            elif metric_name == "chexpert":
                weight_key = "clinical_accuracy"
            elif metric_name == "radgraph_f1":
                weight_key = "radgraph_f1"
            
            weight = metric_weights.get(weight_key, 0.1)  # Default weight if not specified
            weighted_sum += score * weight
            total_weight += weight
        
        # Normalize by total weight to get score between 0 and 1
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback to simple average
            return sum(metric_scores.values()) / len(metric_scores)
    
    def _determine_quality_level(self, overall_score: float) -> str:
        """
        Determine quality level based on overall score.
        
        Args:
            overall_score (float): Overall evaluation score
            
        Returns:
            str: Quality level (excellent, good, fair, poor)
        """
        thresholds = self.config.get('thresholds', {}).get('quality_levels', {
            'excellent': 0.8,
            'good': 0.65,
            'fair': 0.5,
            'poor': 0.35
        })
        
        if overall_score >= thresholds.get('excellent', 0.8):
            return "excellent"
        elif overall_score >= thresholds.get('good', 0.65):
            return "good"
        elif overall_score >= thresholds.get('fair', 0.5):
            return "fair"
        elif overall_score >= thresholds.get('poor', 0.35):
            return "poor"
        else:
            return "very_poor"
    
    def evaluate_batch(self, report_pairs: List[Tuple[str, str]], 
                      image_ids: Optional[List[str]] = None,
                      save_intermediate: bool = True,
                      progress_callback: Optional[callable] = None) -> Dict:
        """
        Evaluate multiple report pairs in batch mode.
        
        Args:
            report_pairs (List[Tuple[str, str]]): List of (generated, ground_truth) report pairs
            image_ids (Optional[List[str]]): Corresponding image IDs
            save_intermediate (bool): Save results after each evaluation
            progress_callback (Optional[callable]): Callback function for progress updates
            
        Returns:
            Dict: Batch evaluation results with individual and aggregate scores
            
        Example:
            {
                "batch_summary": {
                    "total_evaluated": 150,
                    "successful": 148,
                    "failed": 2,
                    "average_scores": {...},
                    "processing_time": 45.2
                },
                "individual_results": [...],
                "aggregate_metrics": {...}
            }
        """
        start_time = time.time()
        
        # Validate inputs
        if not report_pairs:
            raise ValueError("report_pairs cannot be empty")
        
        if image_ids and len(image_ids) != len(report_pairs):
            raise ValueError(f"Length mismatch: {len(image_ids)} image_ids vs {len(report_pairs)} report_pairs")
        
        # Generate batch ID
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(report_pairs)}"
        
        # Initialize batch result structure
        batch_result = {
            "batch_id": batch_id,
            "session_id": self.evaluation_session_id,
            "batch_size": len(report_pairs),
            "individual_results": [],
            "batch_summary": {
                "total_evaluated": len(report_pairs),
                "successful": 0,
                "failed": 0,
                "processing_time": 0.0,
                "average_time_per_report": 0.0,
                "success_rate": 0.0
            },
            "aggregate_metrics": {},
            "processing_info": {
                "start_time": datetime.now().isoformat(),
                "errors": []
            },
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Starting batch evaluation - ID: {batch_id}, Size: {len(report_pairs)}")
        
        # Process each report pair
        successful_results = []
        
        for i, (generated_report, ground_truth_report) in enumerate(report_pairs):
            try:
                # Get image ID for this pair
                current_image_id = image_ids[i] if image_ids else f"{batch_id}_item_{i:04d}"
                
                # Progress callback
                if progress_callback:
                    progress_callback(i, len(report_pairs), current_image_id)
                
                # Evaluate single report
                individual_result = self.evaluate_single(
                    generated_report=generated_report,
                    ground_truth_report=ground_truth_report,
                    image_id=current_image_id,
                    metadata={"batch_id": batch_id, "batch_index": i}
                )
                
                successful_results.append(individual_result)
                batch_result["individual_results"].append(individual_result)
                batch_result["batch_summary"]["successful"] += 1
                
                # Log progress for large batches
                if (i + 1) % 10 == 0 or i == len(report_pairs) - 1:
                    self.logger.info(f"Processed {i + 1}/{len(report_pairs)} reports")
                
                # Save intermediate results if requested
                if save_intermediate and (i + 1) % 50 == 0:
                    self._save_intermediate_results(batch_result, batch_id)
                
            except Exception as e:
                error_info = {
                    "index": i,
                    "image_id": image_ids[i] if image_ids else f"{batch_id}_item_{i:04d}",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
                batch_result["batch_summary"]["failed"] += 1
                batch_result["processing_info"]["errors"].append(error_info)
                
                self.logger.error(f"Error processing report {i}: {e}")
                
                # Continue processing other reports
                continue
        
        # Calculate batch statistics using aggregator
        if successful_results:
            batch_stats = self.aggregator.aggregate_batch_results(successful_results)
            batch_result["aggregate_metrics"] = batch_stats
            
            # Update batch summary with aggregated results
            batch_result["batch_summary"]["average_scores"] = {
                "overall_score": batch_stats.get("batch_overall_score", 0.0),
                "statistics": batch_stats.get("batch_statistics", {})
            }
            
        else:
            batch_result["aggregate_metrics"] = {
                "error": "No successful evaluations",
                "batch_overall_score": 0.0
            }
        
        # Finalize processing info
        total_time = time.time() - start_time
        batch_result["processing_info"]["end_time"] = datetime.now().isoformat()
        batch_result["batch_summary"]["processing_time"] = total_time
        batch_result["batch_summary"]["average_time_per_report"] = total_time / len(report_pairs) if report_pairs else 0.0
        batch_result["batch_summary"]["success_rate"] = batch_result["batch_summary"]["successful"] / len(report_pairs) if report_pairs else 0.0
        
        # Save final results if requested
        if save_intermediate:
            self._save_batch_results(batch_result, batch_id)
        
        self.logger.info(f"Batch evaluation completed - ID: {batch_id}")
        self.logger.info(f"Success: {batch_result['batch_summary']['successful']}/{len(report_pairs)}, "
                        f"Time: {total_time:.2f}s, "
                        f"Avg: {total_time/len(report_pairs):.3f}s/report")
        
        if batch_result["aggregate_metrics"].get("batch_overall_score"):
            self.logger.info(f"Batch Overall Score: {batch_result['aggregate_metrics']['batch_overall_score']:.4f}")
        
        return batch_result
    
    def _save_intermediate_results(self, batch_result: Dict, batch_id: str) -> None:
        """
        Save intermediate batch results to file.
        
        Args:
            batch_result (Dict): Current batch results
            batch_id (str): Batch identifier
        """
        try:
            # Create results directory if it doesn't exist
            results_dir = self.config.get('paths', {}).get('evaluation_results', 'evaluation_results')
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Save intermediate results
            intermediate_file = os.path.join(results_dir, f"{batch_id}_intermediate.json")
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(batch_result, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved intermediate results to {intermediate_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {e}")
    
    def _save_batch_results(self, batch_result: Dict, batch_id: str) -> None:
        """
        Save final batch results to file.
        
        Args:
            batch_result (Dict): Final batch results
            batch_id (str): Batch identifier
        """
        try:
            # Create results directory if it doesn't exist
            results_dir = self.config.get('paths', {}).get('evaluation_results', 'evaluation_results')
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Save final results
            final_file = os.path.join(results_dir, f"{batch_id}_final.json")
            with open(final_file, 'w', encoding='utf-8') as f:
                json.dump(batch_result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved batch results to {final_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving batch results: {e}")
    
    def export_results(self, results: Dict, output_path: str = None,
                      format: str = "json", include_metadata: bool = True) -> str:
        """
        Export evaluation results to various formats.
        
        Args:
            results (Dict): Evaluation results to export
            output_path (str, optional): Output file path. If None, generates automatic path.
            format (str): Output format ("json", "csv", "txt", "html")
            include_metadata (bool): Include evaluation metadata
            
        Returns:
            str: Path to the exported file
            
        Raises:
            ValueError: If unsupported format specified
            IOError: If unable to write to output path
        """
        try:
            # Auto-generate output path if not provided
            if output_path is None:
                results_dir = self.config.get('paths', {}).get('evaluation_results', 'evaluation_results')
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                
                # Generate filename based on result type
                if 'batch_id' in results:
                    filename = f"{results['batch_id']}_results.{format}"
                elif 'image_id' in results:
                    filename = f"{results['image_id']}_result.{format}"
                else:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"evaluation_result_{timestamp}.{format}"
                
                output_path = os.path.join(results_dir, filename)
            
            # Use the results exporter
            exported_path = self.exporter.export_results(
                results=results,
                output_path=output_path,
                format=format,
                include_metadata=include_metadata,
                precision=self.config.get('output_formats', {}).get('precision', 4)
            )
            
            self.logger.info(f"Results exported to {exported_path}")
            return exported_path
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            raise
    
    def get_supported_metrics(self) -> List[str]:
        """
        Get list of supported evaluation metrics.
        
        Returns:
            List[str]: List of metric names
        """
        if not self.config:
            return []
        
        return list(self.config.get("metrics", {}).get("weights", {}).keys())
    
    def get_system_status(self) -> Dict:
        """
        Get current system status and health check.
        
        Returns:
            Dict: System status information
        """
        status = {
            "session_id": self.evaluation_session_id,
            "config_loaded": self.config is not None,
            "ground_truth_loaded": len(self.ground_truth_data) > 0,
            "available_metrics": self.get_supported_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
        return status
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Validate the current configuration.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        if not self.config:
            issues.append("No configuration loaded")
            return False, issues
        
        # Check metric weights sum to 1.0
        weights = self.config.get("metrics", {}).get("weights", {})
        if weights:
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                issues.append(f"Metric weights sum to {total_weight:.3f}, should be 1.0")
        
        # Check required paths exist
        paths = self.config.get("paths", {})
        for path_name, path_value in paths.items():
            if path_name != "logs" and not os.path.exists(path_value):
                issues.append(f"Path does not exist: {path_name} = {path_value}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def __str__(self) -> str:
        """String representation of the evaluator."""
        status = self.get_system_status()
        return f"MedicalReportEvaluator(session={status['session_id']}, metrics={len(status['available_metrics'])})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"MedicalReportEvaluator(config_path='{self.config_path}', session_id='{self.evaluation_session_id}')"


# Utility functions for testing and validation
def test_evaluator_instantiation() -> bool:
    """
    Test basic evaluator instantiation.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        evaluator = MedicalReportEvaluator()
        print(f"✅ Evaluator created successfully: {evaluator}")
        
        # Test configuration validation
        is_valid, issues = evaluator.validate_configuration()
        if not is_valid:
            print(f"⚠️ Configuration issues found: {issues}")
        else:
            print("✅ Configuration is valid")
        
        # Test system status
        status = evaluator.get_system_status()
        print(f"✅ System status: {status['available_metrics']} metrics available")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create evaluator: {e}")
        return False


def main():
    """Main entry point with command-line argument parsing."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Medical Report Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python medical_report_evaluator.py                    # System health check
  python medical_report_evaluator.py --auto             # Full automation
  python medical_report_evaluator.py --discover         # Discovery only
  python medical_report_evaluator.py --auto --verbose   # Full automation with detailed logging
        """
    )
    
    parser.add_argument('--auto', action='store_true',
                       help='Run full automation pipeline')
    parser.add_argument('--discover', action='store_true',
                       help='Run discovery and matching only')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--config', type=str, default='config/evaluation_config.json',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    if args.auto:
        run_full_automation(args)
    elif args.discover:
        run_discovery_only(args)
    else:
        run_health_check(args)

def run_full_automation(args):
    """Execute complete automation pipeline."""
    try:
        from automation.evaluation_controller import AutomatedEvaluationController
        
        print("🚀 Starting full automation pipeline...")
        controller = AutomatedEvaluationController(
            config_path=args.config
        )
        results = controller.run_full_automation()
        
        print(f"\n🎉 Automation completed successfully!")
        print(f"📊 Processed {results.get('evaluation_results', {}).get('total_pairs', 0)} report pairs")
        print(f"📁 Results saved to output/ directory")
        
    except ImportError as e:
        print(f"❌ Automation components not yet implemented: {e}")
        print("💡 Please complete Phase A and B implementation first")
    except Exception as e:
        print(f"❌ Automation failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()

def run_discovery_only(args):
    """Run discovery and matching only."""
    try:
        from automation.data_discovery import discover_and_match_reports
        
        print("🔍 Starting data discovery and matching...")
        matched_pairs, unmatched_llm, unmatched_gt = discover_and_match_reports()
        
        print(f"\n🔍 Discovery completed:")
        print(f"✅ Found {len(matched_pairs)} matched pairs")
        print(f"⚠️  {len(unmatched_llm)} unmatched LLM reports")
        print(f"⚠️  {len(unmatched_gt)} unmatched ground truth reports")
        
        if matched_pairs:
            print(f"\n📋 Sample matched pairs:")
            for i, pair in enumerate(matched_pairs[:3]):
                print(f"  {i+1}. Study ID: {pair['study_id']}")
        
    except ImportError as e:
        print(f"❌ Discovery components not yet implemented: {e}")
        print("💡 Please complete Phase A implementation first")
    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()

def run_health_check(args):
    """Run system health check and validation."""
    print("🧪 Testing Medical Report Evaluator...")
    success = test_evaluator_instantiation()
    
    if success:
        print("\n🎉 All basic tests passed!")
        print("\nUsage:")
        print("  python medical_report_evaluator.py --auto     # Run full automation")
        print("  python medical_report_evaluator.py --discover # Discovery only")
        print("  python medical_report_evaluator.py --help     # Show all options")
    else:
        print("\n❌ Some tests failed!")

if __name__ == "__main__":
    main() 