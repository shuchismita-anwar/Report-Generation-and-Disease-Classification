#!/usr/bin/env python3
"""
Automated Visualization Engine  
=============================
Automated visualization generation for MIMIC-Eye evaluation results.

Author: Medical Report Evaluation System
Version: 1.0
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List
import numpy as np


class AutomatedVisualizationEngine:
    """Generate automated visualizations for evaluation results."""
    
    def __init__(self, output_dir: str = "output/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_metric_distribution(self, results: Dict) -> str:
        """Generate metric distribution visualization."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract metric scores from results
            metrics_data = results.get('aggregate_metrics', {}).get('batch_statistics', {}).get('per_metric', {})
            
            if metrics_data:
                metrics = list(metrics_data.keys())
                scores = [metrics_data[m].get('mean', 0) for m in metrics]
                
                bars = ax.bar(metrics, scores)
                ax.set_title('Metric Score Distribution')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            filename = f"metric_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Error generating metric distribution: {e}")
            return ""
    
    def generate_dashboard_html(self, results: Dict) -> str:
        """Generate HTML dashboard."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MIMIC-Eye Evaluation Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .score {{ font-size: 24px; font-weight: bold; color: #2E8B57; }}
    </style>
</head>
<body>
    <h1>MIMIC-Eye Evaluation Dashboard</h1>
    <p>Generated: {datetime.now().isoformat()}</p>
    
    <div class="metric">
        <h3>Overall Performance</h3>
        <div class="score">{results.get('aggregate_metrics', {}).get('batch_overall_score', 0):.4f}</div>
    </div>
    
    <div class="metric">
        <h3>Processing Summary</h3>
        <p>Total Evaluations: {results.get('batch_summary', {}).get('total_evaluated', 0)}</p>
        <p>Success Rate: {results.get('batch_summary', {}).get('success_rate', 0)*100:.1f}%</p>
        <p>Processing Time: {results.get('batch_summary', {}).get('processing_time', 0):.2f}s</p>
    </div>
</body>
</html>
"""
        
        filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath

    def generate_all_visualizations(self, results: Dict) -> Dict:
        """Generate all visualizations for automation results."""
        metric_chart = self.generate_metric_distribution(results)
        dashboard = self.generate_dashboard_html(results)
        
        return {
            "metric_distribution": metric_chart,
            "dashboard": dashboard
        }


def generate_visualizations(results: Dict) -> Dict:
    """Generate all visualizations for automation results."""
    engine = AutomatedVisualizationEngine()
    return engine.generate_all_visualizations(results)