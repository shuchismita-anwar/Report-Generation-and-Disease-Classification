#!/usr/bin/env python3
"""
MIMIC-Eye Evaluation Automation Package
======================================
Comprehensive automation suite for medical report evaluation.

Author: Medical Report Evaluation System
Version: 1.0
"""

# Import automation components
from .progress_monitor import create_progress_monitor, EvaluationMonitor
from .error_recovery import create_recovery_manager, ErrorRecoveryManager
from .batch_processor import create_batch_processor, BatchProcessor
from .report_generator import AutomatedReportGenerator
from .visualization_engine import AutomatedVisualizationEngine
from .insights_generator import AutomatedInsightsGenerator
from .data_discovery import discover_and_match_reports, load_ground_truth_report
from .evaluation_controller import AutomatedEvaluationController, run_automation_pipeline

__all__ = [
    # Progress monitoring
    'create_progress_monitor',
    'EvaluationMonitor',
    
    # Error recovery
    'create_recovery_manager', 
    'ErrorRecoveryManager',
    
    # Batch processing
    'create_batch_processor',
    'BatchProcessor',
    
    # Report generation
    'AutomatedReportGenerator',
    
    # Visualization
    'AutomatedVisualizationEngine',
    
    # Insights generation
    'AutomatedInsightsGenerator',
    
    # Data discovery
    'discover_and_match_reports',
    'load_ground_truth_report',
    
    # Evaluation controller
    'AutomatedEvaluationController',
    'run_automation_pipeline'
] 