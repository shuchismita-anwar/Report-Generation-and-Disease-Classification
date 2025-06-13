"""
MIMIC-Eye Automation Package
===========================
Automated evaluation pipeline for LLM-generated medical reports.

This package provides:
- Automated data discovery and matching
- Batch evaluation processing
- Comprehensive reporting and visualization
- Error recovery and monitoring

Author: Medical Report Evaluation System
Version: 1.0
"""

__version__ = "1.0.0"
__author__ = "Medical Report Evaluation System"

# Package imports for easy access
from .data_discovery import discover_llm_reports, discover_ground_truth_reports, discover_and_match_reports
from .evaluation_controller import AutomatedEvaluationController, run_automation_pipeline

__all__ = [
    'discover_llm_reports',
    'discover_ground_truth_reports', 
    'discover_and_match_reports',
    'AutomatedEvaluationController',
    'run_automation_pipeline'
] 