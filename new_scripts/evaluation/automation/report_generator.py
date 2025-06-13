#!/usr/bin/env python3
"""
Automated Report Generator
==========================
Comprehensive report generation for MIMIC-Eye evaluation results.

Author: Medical Report Evaluation System
Version: 1.0
"""

import json
import os
from datetime import datetime
from typing import Dict, List


class AutomatedReportGenerator:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, output_dir: str = "output/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_executive_summary(self, results: Dict) -> str:
        """Generate executive summary report."""
        summary = f"""
# MIMIC-Eye Evaluation Executive Summary
Generated: {datetime.now().isoformat()}

## Overview
- **Total Evaluations**: {results.get('batch_summary', {}).get('total_evaluated', 0)}
- **Success Rate**: {results.get('batch_summary', {}).get('success_rate', 0)*100:.1f}%
- **Overall Score**: {results.get('aggregate_metrics', {}).get('batch_overall_score', 0):.4f}

## Key Findings
- Primary strength: {'High semantic similarity' if results.get('aggregate_metrics', {}).get('batch_overall_score', 0) > 0.5 else 'Room for improvement in alignment'}
- Processing efficiency: {results.get('batch_summary', {}).get('processing_time', 0):.2f}s total

## Recommendations
- {'Continue current approach' if results.get('batch_summary', {}).get('success_rate', 0) > 0.9 else 'Review error patterns and optimize'}
"""
        return summary
    
    def save_report(self, content: str, filename: str) -> str:
        """Save report to file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath


def generate_reports(results: Dict) -> Dict:
    """Generate all reports for automation results."""
    generator = AutomatedReportGenerator()
    
    executive_summary = generator.generate_executive_summary(results)
    exec_file = generator.save_report(executive_summary, f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    
    return {"executive_summary": exec_file} 