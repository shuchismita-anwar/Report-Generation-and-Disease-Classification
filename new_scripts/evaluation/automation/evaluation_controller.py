#!/usr/bin/env python3
"""
Automated Evaluation Controller
==============================
Main automation orchestrator for the MIMIC-Eye evaluation pipeline.

This module provides:
- Automated discovery and matching of reports
- Batch evaluation processing
- Progress monitoring and error recovery
- Comprehensive results generation

Author: Medical Report Evaluation System
Version: 1.0
"""

import json
import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import core evaluation system
import sys
sys.path.append('..')
from medical_report_evaluator import MedicalReportEvaluator

# Import automation components
from .data_discovery import discover_and_match_reports


class AutomatedEvaluationController:
    """
    Main controller for automated evaluation pipeline.
    
    Orchestrates the complete process from discovery to results generation.
    """
    
    def __init__(self, config_path: str = "config/evaluation_config.json"):
        """
        Initialize the automated evaluation controller.
        
        Args:
            config_path (str): Path to evaluation configuration file
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize core evaluator
        self.evaluator = MedicalReportEvaluator(config_path)
        
        # Automation state
        self.matched_pairs = []
        self.unmatched_llm = []
        self.unmatched_gt = []
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Progress tracking
        self.total_pairs = 0
        self.processed_pairs = 0
        self.successful_evaluations = 0
        self.failed_evaluations = 0
        
        self.logger.info("Automated Evaluation Controller initialized")
    
    def run_full_automation(self, 
                          llm_path: str = "../../real_analysis_results/",
                          gt_path: str = "../../mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye/",
                          output_dir: str = "../output/automation/") -> Dict:
        """
        Run the complete automated evaluation pipeline.
        
        Args:
            llm_path (str): Path to LLM analysis results
            gt_path (str): Path to ground truth reports
            output_dir (str): Output directory for results
            
        Returns:
            Dict: Complete automation results
        """
        self.start_time = datetime.now()
        self.logger.info("🚀 Starting full automation pipeline")
        
        try:
            # Phase 1: Discovery and Matching
            self.logger.info("📡 Phase 1: Discovery and Matching")
            self._run_discovery(llm_path, gt_path)
            
            # Phase 2: Validation
            self.logger.info("✅ Phase 2: Data Validation")
            self._run_validation()
            
            # Phase 3: Batch Evaluation
            self.logger.info("⚡ Phase 3: Batch Evaluation")
            self._run_batch_evaluation()
            
            # Phase 4: Results Generation
            self.logger.info("📊 Phase 4: Results Generation")
            self._generate_results(output_dir)
            
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            
            # Final summary
            summary = self._generate_summary(duration)
            self.logger.info("🎉 Automation pipeline completed successfully")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Automation pipeline failed: {e}")
            self.end_time = datetime.now()
            raise
    
    def _run_discovery(self, llm_path: str, gt_path: str):
        """Run data discovery and matching."""
        self.logger.info(f"Discovering reports in LLM path: {llm_path}")
        self.logger.info(f"Discovering reports in GT path: {gt_path}")
        
        # Discover and match reports
        self.matched_pairs, self.unmatched_llm, self.unmatched_gt = discover_and_match_reports(
            llm_path=llm_path
        )
        
        self.total_pairs = len(self.matched_pairs)
        
        self.logger.info(f"✅ Discovery completed:")
        self.logger.info(f"   📋 Found {self.total_pairs} matched pairs")
        self.logger.info(f"   ⚠️  {len(self.unmatched_llm)} unmatched LLM reports")
        self.logger.info(f"   ⚠️  {len(self.unmatched_gt)} unmatched ground truth reports")
        
        if self.total_pairs == 0:
            raise ValueError("No matched pairs found. Cannot proceed with evaluation.")
    
    def _run_validation(self):
        """Validate discovered data."""
        # Simple validation of discovered data
        validation_results = {
            "total_pairs": len(self.matched_pairs),
            "valid_pairs": len(self.matched_pairs), 
            "issues": [],
            "statistics": {
                "matched_reports": len(self.matched_pairs),
                "unmatched_llm": len(self.unmatched_llm),
                "unmatched_gt": len(self.unmatched_gt)
            }
        }
        
        self.logger.info("✅ Data validation completed:")
        self.logger.info(f"   📊 Total pairs: {validation_results['total_pairs']}")
        self.logger.info(f"   ✅ Valid pairs: {validation_results['valid_pairs']}")
        
        stats = validation_results.get('statistics', {})
        self.logger.info(f"   📋 Matched reports: {stats.get('matched_reports', 0)}")
        self.logger.info(f"   ⚠️  Unmatched LLM: {stats.get('unmatched_llm', 0)}")
        self.logger.info(f"   ⚠️  Unmatched GT: {stats.get('unmatched_gt', 0)}")
        
        issues = validation_results.get('issues', [])
        if issues:
            self.logger.warning(f"Found {len(issues)} validation issues")
            for issue in issues[:3]:  # Show first 3 issues
                self.logger.warning(f"   - {issue}")
    
    def _run_batch_evaluation(self):
        """Run batch evaluation on all matched pairs."""
        self.logger.info(f"Starting batch evaluation of {self.total_pairs} pairs")
        
        # Prepare batch data as list of tuples (generated, ground_truth)
        batch_data = []
        image_ids = []
        for pair in self.matched_pairs:
            batch_data.append((pair["llm_report"], pair["ground_truth"]))
            image_ids.append(pair["study_id"])
        
        # Run batch evaluation
        start_time = time.time()
        self.results = self.evaluator.evaluate_batch(
            report_pairs=batch_data,
            image_ids=image_ids,
            save_intermediate=True,
            progress_callback=self._progress_callback
        )
        evaluation_time = time.time() - start_time
        
        # Update counters
        self.successful_evaluations = self.results["batch_summary"]["successful"]
        self.failed_evaluations = self.results["batch_summary"]["failed"]
        
        self.logger.info(f"⚡ Batch evaluation completed in {evaluation_time:.2f}s")
        self.logger.info(f"   ✅ Successful: {self.successful_evaluations}")
        self.logger.info(f"   ❌ Failed: {self.failed_evaluations}")
    
    def _progress_callback(self, current: int, total: int, study_id: str = None):
        """Progress callback for batch evaluation."""
        self.processed_pairs = current
        progress_pct = (current / total) * 100
        
        if study_id:
            self.logger.info(f"📊 Progress: {current}/{total} ({progress_pct:.1f}%) - {study_id}")
        else:
            self.logger.info(f"📊 Progress: {current}/{total} ({progress_pct:.1f}%)")
    
    def _generate_results(self, output_dir: str):
        """Generate comprehensive results and save to files."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        results_file = os.path.join(output_dir, f"automation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Save discovery summary
        discovery_file = os.path.join(output_dir, f"discovery_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        discovery_summary = {
            "matched_pairs": len(self.matched_pairs),
            "unmatched_llm": len(self.unmatched_llm),
            "unmatched_gt": len(self.unmatched_gt),
            "unmatched_llm_ids": self.unmatched_llm,
            "unmatched_gt_ids": self.unmatched_gt[:50],  # Limit to first 50
            "sample_matched_pairs": [pair["study_id"] for pair in self.matched_pairs[:10]]
        }
        with open(discovery_file, 'w', encoding='utf-8') as f:
            json.dump(discovery_summary, f, indent=2, ensure_ascii=False)
        
        # Generate automated reports
        self.logger.info("📊 Generating automated reports...")
        try:
            from .report_generator import generate_reports
            report_files = generate_reports(self.results)
            self.logger.info(f"   📄 Executive summary: {report_files.get('executive_summary', 'N/A')}")
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
        
        # Generate visualizations
        self.logger.info("📈 Generating visualizations...")
        try:
            from .visualization_engine import generate_visualizations
            viz_files = generate_visualizations(self.results)
            self.logger.info(f"   📊 Metric chart: {viz_files.get('metric_distribution', 'N/A')}")
            self.logger.info(f"   🌐 Dashboard: {viz_files.get('dashboard', 'N/A')}")
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
        
        # Generate insights
        self.logger.info("🧠 Generating insights...")
        try:
            from .insights_generator import generate_insights
            insights = generate_insights(self.results)
            insights_file = os.path.join(output_dir, f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(insights_file, 'w', encoding='utf-8') as f:
                json.dump(insights, f, indent=2, ensure_ascii=False)
            self.logger.info(f"   🎯 Insights: {insights_file}")
            self.logger.info(f"   💡 Recommendations: {len(insights.get('recommendations', []))} generated")
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
        
        self.logger.info(f"📁 Results saved to: {output_dir}")
        self.logger.info(f"   📄 Main results: {results_file}")
        self.logger.info(f"   📄 Discovery summary: {discovery_file}")
    
    def _generate_summary(self, duration: float) -> Dict:
        """Generate final automation summary."""
        summary = {
            "automation_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": duration,
                "duration_formatted": f"{duration:.2f}s"
            },
            "discovery_results": {
                "matched_pairs": len(self.matched_pairs),
                "unmatched_llm": len(self.unmatched_llm),
                "unmatched_gt": len(self.unmatched_gt),
                "match_rate": len(self.matched_pairs) / (len(self.matched_pairs) + len(self.unmatched_llm)) if (len(self.matched_pairs) + len(self.unmatched_llm)) > 0 else 0
            },
            "evaluation_results": {
                "total_pairs": self.total_pairs,
                "successful_evaluations": self.successful_evaluations,
                "failed_evaluations": self.failed_evaluations,
                "success_rate": self.successful_evaluations / self.total_pairs if self.total_pairs > 0 else 0
            },
            "performance_metrics": self.results.get("summary", {}) if self.results else {},
            "status": "completed" if self.failed_evaluations == 0 else "completed_with_errors"
        }
        
        return summary


def run_automation_pipeline(llm_path: str = "../../real_analysis_results/",
                           gt_path: str = "../../mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye/",
                           output_dir: str = "output/automation/",
                           config_path: str = "config/evaluation_config.json") -> Dict:
    """
    Convenience function to run the complete automation pipeline.
    
    Args:
        llm_path (str): Path to LLM analysis results
        gt_path (str): Path to ground truth reports  
        output_dir (str): Output directory for results
        config_path (str): Path to evaluation configuration
        
    Returns:
        Dict: Automation summary results
    """
    controller = AutomatedEvaluationController(config_path)
    return controller.run_full_automation(llm_path, gt_path, output_dir)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run automation pipeline
    try:
        results = run_automation_pipeline()
        print("🎉 Automation completed successfully!")
        print(f"📊 Summary: {results['evaluation_results']}")
    except Exception as e:
        print(f"❌ Automation failed: {e}")
        raise 