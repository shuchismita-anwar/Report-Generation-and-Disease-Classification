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
        # Extract key metrics
        batch_summary = results.get("batch_summary", {})
        aggregate_metrics = results.get("aggregate_metrics", {})
        individual_results = results.get("individual_results", [])
        
        total_evaluated = batch_summary.get("total", len(individual_results))
        success_rate = batch_summary.get("success_rate", 0) * 100
        overall_score = aggregate_metrics.get("batch_overall_score", 0)
        processing_time = batch_summary.get("processing_time", 0)
        
        # Quality assessment
        quality_level = "Excellent" if overall_score > 0.8 else \
                       "Good" if overall_score > 0.6 else \
                       "Fair" if overall_score > 0.4 else \
                       "Needs Improvement" if overall_score > 0.2 else \
                       "Poor"
        
        # Individual report quality breakdown
        quality_breakdown = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        metric_details = {}
        
        for result in individual_results:
            score = result.get("overall_score", 0)
            if score > 0.8:
                quality_breakdown["excellent"] += 1
            elif score > 0.6:
                quality_breakdown["good"] += 1
            elif score > 0.4:
                quality_breakdown["fair"] += 1
            else:
                quality_breakdown["poor"] += 1
            
            # Collect metric information
            metrics = result.get("metrics", {})
            for metric_name, metric_data in metrics.items():
                if metric_name not in metric_details:
                    metric_details[metric_name] = []
                
                if isinstance(metric_data, dict):
                    if metric_name == "bleu":
                        metric_details[metric_name].append(metric_data.get("bleu", 0))
                    elif metric_name == "rouge":
                        metric_details[metric_name].append(metric_data.get("rouge_avg_f1", 0))
                    elif metric_name == "meteor":
                        metric_details[metric_name].append(metric_data.get("meteor", 0))
                    elif metric_name == "bert_score":
                        metric_details[metric_name].append(metric_data.get("bertscore_f1", 0))
                    elif metric_name == "chexpert":
                        metric_details[metric_name].append(metric_data.get("chexpert_f1", 0))
                    elif metric_name == "radgraph_f1":
                        metric_details[metric_name].append(metric_data.get("radgraph_f1", 0))
                    elif metric_name == "cider":
                        metric_details[metric_name].append(metric_data.get("cider", 0))
                    elif metric_name == "medical":
                        metric_details[metric_name].append(metric_data.get("medical_score", 0))
        
        # Calculate metric averages and standard deviations
        import statistics
        metric_statistics = {}
        for metric, scores in metric_details.items():
            if scores:  # Include all metrics including BERTScore
                metric_statistics[metric] = {
                    'mean': sum(scores) / len(scores),
                    'std': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    'count': len(scores)
                }
        
        # Best and worst performing reports
        sorted_results = sorted(individual_results, key=lambda x: x.get("overall_score", 0), reverse=True)
        best_report = sorted_results[0] if sorted_results else None
        worst_report = sorted_results[-1] if sorted_results else None
        
        # Calculate overall score standard deviation
        overall_scores = [result.get("overall_score", 0) for result in individual_results]
        overall_std = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0
        
        summary = f"""
# 📊 MIMIC-Eye Medical Report Evaluation Summary
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Overall Performance
- **Reports Evaluated**: {total_evaluated}
- **Overall Quality Score (Mean)**: {overall_score:.4f} ({quality_level})
- **Overall Quality Score (Std Dev)**: ±{overall_std:.4f}
- **Success Rate**: {success_rate:.1f}%
- **Processing Time**: {processing_time:.3f} seconds

## 📈 Quality Distribution
- **Excellent (>80%)**: {quality_breakdown['excellent']} reports
- **Good (60-80%)**: {quality_breakdown['good']} reports  
- **Fair (40-60%)**: {quality_breakdown['fair']} reports
- **Poor (<40%)**: {quality_breakdown['poor']} reports

## 📋 Individual Metric Performance
*Note: Values shown as Mean ± Standard Deviation across all {total_evaluated} patients*

"""
        
        # Add metric details with mean and std
        for metric, stats in metric_statistics.items():
            metric_name = metric.upper()
            mean_score = stats['mean']
            std_score = stats['std']
            performance = "Excellent" if mean_score > 0.8 else \
                         "Good" if mean_score > 0.6 else \
                         "Fair" if mean_score > 0.4 else \
                         "Needs Work" if mean_score > 0.2 else \
                         "Poor"
            summary += f"- **{metric_name}**: {mean_score:.4f} ± {std_score:.4f} ({performance})\n"
        
        # Add best/worst analysis
        if best_report and worst_report and len(individual_results) > 1:
            summary += f"""
## 🏆 Report Analysis
- **Best Performing**: {best_report.get('image_id', 'N/A')} (Score: {best_report.get('overall_score', 0):.4f})
- **Needs Most Improvement**: {worst_report.get('image_id', 'N/A')} (Score: {worst_report.get('overall_score', 0):.4f})
"""
        
        # Add recommendations based on scores
        summary += f"""
## 💡 Key Insights & Recommendations

"""
        
        if overall_score < 0.3:
            summary += """
**🚨 Critical Areas for Improvement:**
- Overall quality is below acceptable threshold
- Consider reviewing LLM training data and prompts
- Focus on medical terminology and clinical accuracy
- Evaluate report structure and length optimization

"""
        elif overall_score < 0.6:
            summary += """
**⚠️ Areas for Improvement:**
- Moderate performance with room for enhancement
- Review specific weak metrics identified above
- Consider fine-tuning on medical report datasets
- Improve alignment with radiologist writing patterns

"""
        else:
            summary += """
**✅ Strong Performance:**
- Quality scores are within acceptable range
- Continue current approach with minor optimizations
- Monitor consistency across different report types

"""
        
        # Add metric-specific recommendations
        if metric_statistics:
            weakest_metric = min(metric_statistics.keys(), key=lambda k: metric_statistics[k]['mean'])
            strongest_metric = max(metric_statistics.keys(), key=lambda k: metric_statistics[k]['mean'])
            
            summary += f"""
**Metric-Specific Insights:**
- **Strongest**: {strongest_metric.upper()} (Mean: {metric_statistics[strongest_metric]['mean']:.4f} ± {metric_statistics[strongest_metric]['std']:.4f}) - maintain this strength
- **Weakest**: {weakest_metric.upper()} (Mean: {metric_statistics[weakest_metric]['mean']:.4f} ± {metric_statistics[weakest_metric]['std']:.4f}) - focus improvement here

"""
        
        summary += f"""
## 📁 Complete Results Location
- **Main Results**: `evaluation_results/` directory
- **Visualizations**: `output/visualizations/` directory  
- **Detailed Analysis**: `output/automation/` directory

---
*This summary provides a high-level overview. Check the HTML dashboard and detailed JSON files for complete metrics.*
"""
        
        return summary
    
    def generate_detailed_report(self, results: Dict) -> str:
        """Generate detailed technical report."""
        report = f"""
# Detailed MIMIC-Eye Evaluation Report
Generated: {datetime.now().isoformat()}

## Batch Information
- Batch ID: {results.get('batch_id', 'N/A')}
- Session ID: {results.get('session_id', 'N/A')}
- Total Reports: {len(results.get('individual_results', []))}

## Individual Results Summary
"""
        
        for i, result in enumerate(results.get('individual_results', []), 1):
            metrics = result.get('metrics', {})
            report += f"""
### Report {i}: {result.get('image_id', 'Unknown')}
- Overall Score: {result.get('overall_score', 0):.4f}
- Quality Level: {result.get('quality_level', 'unknown')}
- Processing Time: {result.get('evaluation_time', 0):.3f}s

**Individual Metrics:**"""
            
            # Add all available metrics
            if 'bleu' in metrics:
                bleu_score = metrics['bleu'].get('bleu', 0) if isinstance(metrics['bleu'], dict) else metrics['bleu']
                report += f"\n- BLEU: {bleu_score:.4f}"
            
            if 'rouge' in metrics:
                rouge_score = metrics['rouge'].get('rouge_avg_f1', 0) if isinstance(metrics['rouge'], dict) else metrics['rouge']
                report += f"\n- ROUGE: {rouge_score:.4f}"
            
            if 'meteor' in metrics:
                meteor_score = metrics['meteor'].get('meteor', 0) if isinstance(metrics['meteor'], dict) else metrics['meteor']
                report += f"\n- METEOR: {meteor_score:.4f}"
            
            if 'bert_score' in metrics:
                bert_score = metrics['bert_score'].get('bertscore_f1', 0) if isinstance(metrics['bert_score'], dict) else metrics['bert_score']
                report += f"\n- BERTScore: {bert_score:.4f}"
            
            if 'chexpert' in metrics:
                chexpert_score = metrics['chexpert'].get('chexpert_f1', 0) if isinstance(metrics['chexpert'], dict) else metrics['chexpert']
                report += f"\n- CheXpert: {chexpert_score:.4f}"
            
            if 'radgraph_f1' in metrics:
                radgraph_score = metrics['radgraph_f1'].get('radgraph_f1', 0) if isinstance(metrics['radgraph_f1'], dict) else metrics['radgraph_f1']
                report += f"\n- RadGraph: {radgraph_score:.4f}"
            
            if 'medical' in metrics:
                medical_score = metrics['medical'].get('medical_score', 0) if isinstance(metrics['medical'], dict) else metrics['medical']
                report += f"\n- Medical: {medical_score:.4f}"
            
            report += "\n"
        
        return report
    
    def save_reports(self, results: Dict) -> Dict[str, str]:
        """Save all reports and return file paths."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate and save executive summary
        executive_summary = self.generate_executive_summary(results)
        exec_file = os.path.join(self.output_dir, f"executive_summary_{timestamp}.md")
        with open(exec_file, 'w', encoding='utf-8') as f:
            f.write(executive_summary)
        
        # Generate and save detailed report
        detailed_report = self.generate_detailed_report(results)
        detail_file = os.path.join(self.output_dir, f"detailed_report_{timestamp}.md")
        with open(detail_file, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        
        return {
            "executive_summary": exec_file,
            "detailed_report": detail_file
        }


def generate_reports(results: Dict) -> Dict:
    """Generate all reports for automation results."""
    generator = AutomatedReportGenerator()
    return generator.save_reports(results) 