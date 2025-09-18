"""
Evaluation Report Generator for Medical Report Evaluation

This module generates comprehensive evaluation reports including executive summaries,
detailed metrics analysis, visualizations, and actionable recommendations.

Author: Medical Report Evaluation System
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
import json
from datetime import datetime
from jinja2 import Template
import base64
from io import BytesIO

class ReportGenerator:
    """
    Generates comprehensive evaluation reports.
    
    Features:
    - Executive summary generation
    - Detailed metrics analysis
    - Visualization integration
    - HTML and PDF report generation
    - Actionable recommendations
    """
    
    def __init__(self, output_dir: str = "evaluation_reports"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ReportGenerator initialized with output directory: {output_dir}")
        
        # HTML template for reports
        self.html_template = self._get_html_template()
    
    def generate_comprehensive_report(self, 
                                    evaluation_results: Dict[str, Any],
                                    analysis_results: Optional[Dict[str, Any]] = None,
                                    comparison_results: Optional[Dict[str, Any]] = None,
                                    report_title: str = "Medical Report Evaluation Analysis",
                                    include_visualizations: bool = True) -> Dict[str, str]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            evaluation_results: Main evaluation results
            analysis_results: Optional distribution/trend analysis results
            comparison_results: Optional comparative analysis results
            report_title: Title for the report
            include_visualizations: Whether to include visualizations
            
        Returns:
            Dictionary with paths to generated reports
        """
        try:
            # Generate report components
            executive_summary = self._generate_executive_summary(
                evaluation_results, analysis_results, comparison_results
            )
            
            detailed_analysis = self._generate_detailed_analysis(evaluation_results)
            
            recommendations = self._generate_recommendations(
                evaluation_results, analysis_results, comparison_results
            )
            
            # Generate visualizations if requested
            visualizations = {}
            if include_visualizations:
                visualizations = self._generate_report_visualizations(evaluation_results)
            
            # Compile report data
            report_data = {
                'title': report_title,
                'generation_timestamp': datetime.now().isoformat(),
                'executive_summary': executive_summary,
                'detailed_analysis': detailed_analysis,
                'recommendations': recommendations,
                'visualizations': visualizations,
                'raw_data': {
                    'evaluation_results': evaluation_results,
                    'analysis_results': analysis_results,
                    'comparison_results': comparison_results
                }
            }
            
            # Generate different report formats
            report_paths = {}
            
            # JSON report
            json_path = self._generate_json_report(report_data)
            report_paths['json'] = json_path
            
            # HTML report
            html_path = self._generate_html_report(report_data)
            report_paths['html'] = html_path
            
            # Text report
            text_path = self._generate_text_report(report_data)
            report_paths['text'] = text_path
            
            self.logger.info(f"Comprehensive report generated: {len(report_paths)} formats")
            return report_paths
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            raise
    
    def _generate_executive_summary(self, 
                                  evaluation_results: Dict[str, Any],
                                  analysis_results: Optional[Dict[str, Any]] = None,
                                  comparison_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate executive summary."""
        try:
            # Extract key metrics from evaluation results
            if 'batch_results' in evaluation_results:
                results_data = evaluation_results['batch_results']
            elif 'results' in evaluation_results:
                results_data = evaluation_results['results']
            else:
                results_data = evaluation_results
            
            # Convert to DataFrame for analysis
            if isinstance(results_data, dict):
                df = pd.DataFrame(results_data).T
            else:
                df = pd.DataFrame(results_data)
            
            # Calculate summary statistics
            overall_scores = df.mean(axis=1) if len(df.columns) > 1 else df.iloc[:, 0]
            
            summary_stats = {
                'total_reports_evaluated': len(df),
                'metrics_analyzed': list(df.columns),
                'overall_performance': {
                    'mean_score': float(overall_scores.mean()),
                    'median_score': float(overall_scores.median()),
                    'std_score': float(overall_scores.std()),
                    'min_score': float(overall_scores.min()),
                    'max_score': float(overall_scores.max())
                },
                'metric_performance': {
                    metric: {
                        'mean': float(df[metric].mean()),
                        'std': float(df[metric].std())
                    } for metric in df.columns
                }
            }
            
            # Performance assessment
            mean_score = summary_stats['overall_performance']['mean_score']
            if mean_score >= 0.8:
                performance_level = "Excellent"
                performance_description = "The model demonstrates exceptional performance across all metrics."
            elif mean_score >= 0.7:
                performance_level = "Good"
                performance_description = "The model shows strong performance with room for minor improvements."
            elif mean_score >= 0.6:
                performance_level = "Satisfactory"
                performance_description = "The model performs adequately but has significant room for improvement."
            else:
                performance_level = "Needs Improvement"
                performance_description = "The model requires substantial improvements to meet quality standards."
            
            # Key findings
            key_findings = []
            
            # Best and worst performing metrics with std dev
            metric_means = {metric: float(df[metric].mean()) for metric in df.columns}
            metric_stds = {metric: float(df[metric].std()) for metric in df.columns}
            best_metric = max(metric_means, key=metric_means.get)
            worst_metric = min(metric_means, key=metric_means.get)
            
            key_findings.append(f"Best performing metric: {best_metric} (Mean: {metric_means[best_metric]:.3f} ± {metric_stds[best_metric]:.3f})")
            key_findings.append(f"Lowest performing metric: {worst_metric} (Mean: {metric_means[worst_metric]:.3f} ± {metric_stds[worst_metric]:.3f})")
            
            # Consistency analysis
            consistency = 1 - (summary_stats['overall_performance']['std_score'] / summary_stats['overall_performance']['mean_score'])
            if consistency > 0.8:
                key_findings.append("High consistency across reports (low variance)")
            elif consistency > 0.6:
                key_findings.append("Moderate consistency across reports")
            else:
                key_findings.append("High variance in performance across reports")
            
            # Add analysis-specific findings
            if analysis_results:
                if 'outliers' in analysis_results:
                    outlier_count = analysis_results['outliers'].get('count', 0)
                    if outlier_count > 0:
                        key_findings.append(f"Identified {outlier_count} outlier reports requiring attention")
            
            # Add comparison-specific findings
            if comparison_results:
                if 'summary' in comparison_results:
                    verdict = comparison_results['summary'].get('overall_verdict', 'Unknown')
                    key_findings.append(f"Comparison analysis: {verdict}")
            
            executive_summary = {
                'performance_level': performance_level,
                'performance_description': performance_description,
                'summary_statistics': summary_stats,
                'key_findings': key_findings,
                'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return executive_summary
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            raise
    
    def _generate_detailed_analysis(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed analysis section."""
        try:
            # Extract results data
            if 'batch_results' in evaluation_results:
                results_data = evaluation_results['batch_results']
            elif 'results' in evaluation_results:
                results_data = evaluation_results['results']
            else:
                results_data = evaluation_results
            
            # Convert to DataFrame
            if isinstance(results_data, dict):
                df = pd.DataFrame(results_data).T
            else:
                df = pd.DataFrame(results_data)
            
            detailed_analysis = {}
            
            # Metric-by-metric analysis
            for metric in df.columns:
                metric_scores = df[metric]
                
                # Statistical analysis
                stats_analysis = {
                    'descriptive_statistics': {
                        'count': len(metric_scores),
                        'mean': float(metric_scores.mean()),
                        'median': float(metric_scores.median()),
                        'std': float(metric_scores.std()),
                        'min': float(metric_scores.min()),
                        'max': float(metric_scores.max()),
                        'q1': float(metric_scores.quantile(0.25)),
                        'q3': float(metric_scores.quantile(0.75)),
                        'iqr': float(metric_scores.quantile(0.75) - metric_scores.quantile(0.25))
                    }
                }
                
                # Quality distribution
                excellent_count = len(metric_scores[metric_scores >= 0.8])
                good_count = len(metric_scores[(metric_scores >= 0.6) & (metric_scores < 0.8)])
                fair_count = len(metric_scores[(metric_scores >= 0.4) & (metric_scores < 0.6)])
                poor_count = len(metric_scores[metric_scores < 0.4])
                
                quality_distribution = {
                    'excellent': {'count': excellent_count, 'percentage': (excellent_count / len(metric_scores)) * 100},
                    'good': {'count': good_count, 'percentage': (good_count / len(metric_scores)) * 100},
                    'fair': {'count': fair_count, 'percentage': (fair_count / len(metric_scores)) * 100},
                    'poor': {'count': poor_count, 'percentage': (poor_count / len(metric_scores)) * 100}
                }
                
                # Performance insights
                insights = []
                
                mean_score = stats_analysis['descriptive_statistics']['mean']
                std_score = stats_analysis['descriptive_statistics']['std']
                
                if mean_score >= 0.8:
                    insights.append(f"{metric} shows excellent performance (Mean: {mean_score:.3f} ± {std_score:.3f})")
                elif mean_score >= 0.7:
                    insights.append(f"{metric} shows good performance (Mean: {mean_score:.3f} ± {std_score:.3f})")
                else:
                    insights.append(f"{metric} needs improvement (Mean: {mean_score:.3f} ± {std_score:.3f})")
                
                if std_score < 0.1:
                    insights.append(f"{metric} shows high consistency (Std Dev: {std_score:.3f})")
                elif std_score > 0.2:
                    insights.append(f"{metric} shows high variability (Std Dev: {std_score:.3f})")
                
                detailed_analysis[metric] = {
                    'statistical_analysis': stats_analysis,
                    'quality_distribution': quality_distribution,
                    'insights': insights
                }
            
            # Cross-metric analysis
            correlation_matrix = df.corr().to_dict()
            
            # Find highly correlated metrics
            high_correlations = []
            for i, metric1 in enumerate(df.columns):
                for j, metric2 in enumerate(df.columns):
                    if i < j:  # Avoid duplicates
                        corr = correlation_matrix[metric1][metric2]
                        if abs(corr) > 0.7:
                            high_correlations.append({
                                'metric1': metric1,
                                'metric2': metric2,
                                'correlation': float(corr),
                                'interpretation': 'Strong positive' if corr > 0.7 else 'Strong negative'
                            })
            
            detailed_analysis['cross_metric_analysis'] = {
                'correlation_matrix': correlation_matrix,
                'high_correlations': high_correlations
            }
            
            return detailed_analysis
            
        except Exception as e:
            self.logger.error(f"Error generating detailed analysis: {e}")
            raise
    
    def _generate_recommendations(self, 
                                evaluation_results: Dict[str, Any],
                                analysis_results: Optional[Dict[str, Any]] = None,
                                comparison_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate actionable recommendations."""
        try:
            recommendations = {
                'immediate_actions': [],
                'short_term_improvements': [],
                'long_term_strategies': [],
                'monitoring_suggestions': []
            }
            
            # Extract results data
            if 'batch_results' in evaluation_results:
                results_data = evaluation_results['batch_results']
            elif 'results' in evaluation_results:
                results_data = evaluation_results['results']
            else:
                results_data = evaluation_results
            
            # Convert to DataFrame
            if isinstance(results_data, dict):
                df = pd.DataFrame(results_data).T
            else:
                df = pd.DataFrame(results_data)
            
            # Overall performance assessment
            overall_mean = df.mean(axis=1).mean()
            
            # Metric-specific recommendations
            for metric in df.columns:
                metric_mean = df[metric].mean()
                metric_std = df[metric].std()
                
                if metric_mean < 0.6:
                    recommendations['immediate_actions'].append(
                        f"Address {metric} performance immediately (current: {metric_mean:.3f})"
                    )
                elif metric_mean < 0.7:
                    recommendations['short_term_improvements'].append(
                        f"Focus on improving {metric} in next iteration (current: {metric_mean:.3f})"
                    )
                
                if metric_std > 0.2:
                    recommendations['monitoring_suggestions'].append(
                        f"Monitor {metric} consistency - high variability detected (std: {metric_std:.3f})"
                    )
            
            # Overall system recommendations
            if overall_mean < 0.6:
                recommendations['immediate_actions'].append(
                    "System requires immediate attention - overall performance below acceptable threshold"
                )
                recommendations['long_term_strategies'].append(
                    "Consider comprehensive model retraining or architecture changes"
                )
            elif overall_mean < 0.7:
                recommendations['short_term_improvements'].append(
                    "System performance is adequate but has room for improvement"
                )
                recommendations['long_term_strategies'].append(
                    "Implement targeted improvements for underperforming metrics"
                )
            else:
                recommendations['monitoring_suggestions'].append(
                    "Maintain current performance levels through regular monitoring"
                )
            
            # Analysis-based recommendations
            if analysis_results:
                if 'outliers' in analysis_results:
                    outlier_count = analysis_results['outliers'].get('count', 0)
                    if outlier_count > 0:
                        recommendations['immediate_actions'].append(
                            f"Investigate {outlier_count} outlier reports for data quality issues"
                        )
                
                if 'quality_categories' in analysis_results:
                    poor_percentage = analysis_results['quality_categories'].get('poor', {}).get('percentage', 0)
                    if poor_percentage > 20:
                        recommendations['immediate_actions'].append(
                            f"Address {poor_percentage:.1f}% of reports with poor quality scores"
                        )
            
            # Comparison-based recommendations
            if comparison_results:
                if 'summary' in comparison_results:
                    verdict = comparison_results['summary'].get('overall_verdict', '')
                    if verdict == 'Regression':
                        recommendations['immediate_actions'].append(
                            "Do not deploy current model - performance regression detected"
                        )
                    elif verdict == 'Improvement':
                        recommendations['short_term_improvements'].append(
                            "Consider deploying improved model version"
                        )
                
                if 'significant_regressions' in comparison_results.get('summary', {}):
                    regressed_metrics = comparison_results['summary']['significant_regressions']
                    if regressed_metrics:
                        recommendations['immediate_actions'].append(
                            f"Investigate regression in metrics: {', '.join(regressed_metrics)}"
                        )
            
            # General best practices
            recommendations['long_term_strategies'].extend([
                "Implement continuous monitoring and evaluation pipeline",
                "Establish regular model performance review cycles",
                "Maintain comprehensive evaluation dataset for consistent benchmarking"
            ])
            
            recommendations['monitoring_suggestions'].extend([
                "Set up automated alerts for performance degradation",
                "Track metric trends over time",
                "Monitor for data drift in input reports"
            ])
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            raise
    
    def _generate_report_visualizations(self, evaluation_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualizations for the report."""
        try:
            visualizations = {}
            
            # Extract results data
            if 'batch_results' in evaluation_results:
                results_data = evaluation_results['batch_results']
            elif 'results' in evaluation_results:
                results_data = evaluation_results['results']
            else:
                results_data = evaluation_results
            
            # Convert to DataFrame
            if isinstance(results_data, dict):
                df = pd.DataFrame(results_data).T
            else:
                df = pd.DataFrame(results_data)
            
            # 1. Metric comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            metric_means = df.mean()
            metric_means.plot(kind='bar', ax=ax, color='skyblue', alpha=0.8)
            ax.set_title('Average Performance by Metric', fontweight='bold')
            ax.set_ylabel('Average Score')
            ax.set_xlabel('Metrics')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save as base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            visualizations['metric_comparison'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # 2. Score distribution histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            overall_scores = df.mean(axis=1)
            ax.hist(overall_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.set_title('Overall Score Distribution', fontweight='bold')
            ax.set_xlabel('Overall Score')
            ax.set_ylabel('Frequency')
            ax.axvline(overall_scores.mean(), color='red', linestyle='--', 
                      label=f'Mean: {overall_scores.mean():.3f}')
            ax.legend()
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            visualizations['score_distribution'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # 3. Correlation heatmap
            if len(df.columns) > 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                correlation_matrix = df.corr()
                im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                
                # Add correlation values
                for i in range(len(correlation_matrix.columns)):
                    for j in range(len(correlation_matrix.columns)):
                        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black")
                
                ax.set_xticks(range(len(correlation_matrix.columns)))
                ax.set_yticks(range(len(correlation_matrix.columns)))
                ax.set_xticklabels(correlation_matrix.columns, rotation=45)
                ax.set_yticklabels(correlation_matrix.columns)
                ax.set_title('Metric Correlation Matrix', fontweight='bold')
                
                plt.colorbar(im)
                plt.tight_layout()
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                visualizations['correlation_heatmap'] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            return {}
    
    def _generate_json_report(self, report_data: Dict[str, Any]) -> str:
        """Generate JSON format report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = self.output_dir / f"evaluation_report_{timestamp}.json"
            
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            return str(json_path)
            
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {e}")
            raise
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML format report."""
        try:
            template = Template(self.html_template)
            html_content = template.render(**report_data)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = self.output_dir / f"evaluation_report_{timestamp}.html"
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(html_path)
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            raise
    
    def _generate_text_report(self, report_data: Dict[str, Any]) -> str:
        """Generate text format report."""
        try:
            text_content = []
            
            # Header
            text_content.append("=" * 80)
            text_content.append(f"MEDICAL REPORT EVALUATION ANALYSIS")
            text_content.append(f"Generated: {report_data['generation_timestamp']}")
            text_content.append("=" * 80)
            text_content.append("")
            
            # Executive Summary
            text_content.append("EXECUTIVE SUMMARY")
            text_content.append("-" * 40)
            exec_summary = report_data['executive_summary']
            text_content.append(f"Performance Level: {exec_summary['performance_level']}")
            text_content.append(f"Description: {exec_summary['performance_description']}")
            text_content.append("")
            
            # Key Statistics
            stats = exec_summary['summary_statistics']['overall_performance']
            text_content.append("Key Statistics:")
            text_content.append(f"  - Total Reports: {exec_summary['summary_statistics']['total_reports_evaluated']}")
            text_content.append(f"  - Mean Score: {stats['mean_score']:.3f}")
            text_content.append(f"  - Median Score: {stats['median_score']:.3f}")
            text_content.append(f"  - Score Range: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
            text_content.append("")
            
            # Key Findings
            text_content.append("Key Findings:")
            for finding in exec_summary['key_findings']:
                text_content.append(f"  - {finding}")
            text_content.append("")
            
            # Recommendations
            text_content.append("RECOMMENDATIONS")
            text_content.append("-" * 40)
            recommendations = report_data['recommendations']
            
            if recommendations['immediate_actions']:
                text_content.append("Immediate Actions:")
                for action in recommendations['immediate_actions']:
                    text_content.append(f"  - {action}")
                text_content.append("")
            
            if recommendations['short_term_improvements']:
                text_content.append("Short-term Improvements:")
                for improvement in recommendations['short_term_improvements']:
                    text_content.append(f"  - {improvement}")
                text_content.append("")
            
            if recommendations['long_term_strategies']:
                text_content.append("Long-term Strategies:")
                for strategy in recommendations['long_term_strategies']:
                    text_content.append(f"  - {strategy}")
                text_content.append("")
            
            # Detailed Analysis
            text_content.append("DETAILED ANALYSIS")
            text_content.append("-" * 40)
            detailed = report_data['detailed_analysis']
            
            for metric, analysis in detailed.items():
                if metric == 'cross_metric_analysis':
                    continue
                    
                text_content.append(f"{metric} Analysis:")
                stats = analysis['statistical_analysis']['descriptive_statistics']
                text_content.append(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
                text_content.append(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
                
                for insight in analysis['insights']:
                    text_content.append(f"  - {insight}")
                text_content.append("")
            
            # Footer
            text_content.append("=" * 80)
            text_content.append("End of Report")
            text_content.append("=" * 80)
            
            # Save text report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            text_path = self.output_dir / f"evaluation_report_{timestamp}.txt"
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(text_content))
            
            return str(text_path)
            
        except Exception as e:
            self.logger.error(f"Error generating text report: {e}")
            raise
    
    def _get_html_template(self) -> str:
        """Get HTML template for report generation."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { text-align: center; border-bottom: 3px solid #333; padding-bottom: 20px; }
        .section { margin: 30px 0; }
        .section h2 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        .metric-stats { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .recommendation { background: #e8f4fd; padding: 10px; border-left: 4px solid #2196F3; margin: 5px 0; }
        .finding { background: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 5px 0; }
        .performance-excellent { color: #28a745; font-weight: bold; }
        .performance-good { color: #17a2b8; font-weight: bold; }
        .performance-satisfactory { color: #ffc107; font-weight: bold; }
        .performance-needs-improvement { color: #dc3545; font-weight: bold; }
        .visualization { text-align: center; margin: 20px 0; }
        .visualization img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated on: {{ generation_timestamp }}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metric-stats">
            <h3>Performance Assessment</h3>
            <p class="performance-{{ executive_summary.performance_level.lower().replace(' ', '-') }}">
                Level: {{ executive_summary.performance_level }}
            </p>
            <p>{{ executive_summary.performance_description }}</p>
            
            <h4>Key Statistics</h4>
            <ul>
                <li>Total Reports Evaluated: {{ executive_summary.summary_statistics.total_reports_evaluated }}</li>
                <li>Mean Overall Score: {{ "%.3f"|format(executive_summary.summary_statistics.overall_performance.mean_score) }}</li>
                <li>Score Range: {{ "%.3f"|format(executive_summary.summary_statistics.overall_performance.min_score) }} - {{ "%.3f"|format(executive_summary.summary_statistics.overall_performance.max_score) }}</li>
            </ul>
        </div>
        
        <h3>Key Findings</h3>
        {% for finding in executive_summary.key_findings %}
        <div class="finding">{{ finding }}</div>
        {% endfor %}
    </div>

    {% if visualizations %}
    <div class="section">
        <h2>Performance Visualizations</h2>
        {% if visualizations.metric_comparison %}
        <div class="visualization">
            <h3>Metric Comparison</h3>
            <img src="data:image/png;base64,{{ visualizations.metric_comparison }}" alt="Metric Comparison Chart">
        </div>
        {% endif %}
        
        {% if visualizations.score_distribution %}
        <div class="visualization">
            <h3>Score Distribution</h3>
            <img src="data:image/png;base64,{{ visualizations.score_distribution }}" alt="Score Distribution">
        </div>
        {% endif %}
        
        {% if visualizations.correlation_heatmap %}
        <div class="visualization">
            <h3>Metric Correlations</h3>
            <img src="data:image/png;base64,{{ visualizations.correlation_heatmap }}" alt="Correlation Heatmap">
        </div>
        {% endif %}
    </div>
    {% endif %}

    <div class="section">
        <h2>Recommendations</h2>
        
        {% if recommendations.immediate_actions %}
        <h3>Immediate Actions Required</h3>
        {% for action in recommendations.immediate_actions %}
        <div class="recommendation" style="border-left-color: #dc3545;">{{ action }}</div>
        {% endfor %}
        {% endif %}
        
        {% if recommendations.short_term_improvements %}
        <h3>Short-term Improvements</h3>
        {% for improvement in recommendations.short_term_improvements %}
        <div class="recommendation" style="border-left-color: #ffc107;">{{ improvement }}</div>
        {% endfor %}
        {% endif %}
        
        {% if recommendations.long_term_strategies %}
        <h3>Long-term Strategies</h3>
        {% for strategy in recommendations.long_term_strategies %}
        <div class="recommendation" style="border-left-color: #28a745;">{{ strategy }}</div>
        {% endfor %}
        {% endif %}
    </div>

    <div class="section">
        <h2>Detailed Metric Analysis</h2>
        {% for metric, analysis in detailed_analysis.items() %}
        {% if metric != 'cross_metric_analysis' %}
        <div class="metric-stats">
            <h3>{{ metric }}</h3>
            <table>
                <tr>
                    <th>Statistic</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td>{{ "%.3f"|format(analysis.statistical_analysis.descriptive_statistics.mean) }}</td>
                </tr>
                <tr>
                    <td>Standard Deviation</td>
                    <td>{{ "%.3f"|format(analysis.statistical_analysis.descriptive_statistics.std) }}</td>
                </tr>
                <tr>
                    <td>Minimum</td>
                    <td>{{ "%.3f"|format(analysis.statistical_analysis.descriptive_statistics.min) }}</td>
                </tr>
                <tr>
                    <td>Maximum</td>
                    <td>{{ "%.3f"|format(analysis.statistical_analysis.descriptive_statistics.max) }}</td>
                </tr>
            </table>
            
            <h4>Quality Distribution</h4>
            <ul>
                <li>Excellent (≥0.8): {{ analysis.quality_distribution.excellent.count }} reports ({{ "%.1f"|format(analysis.quality_distribution.excellent.percentage) }}%)</li>
                <li>Good (0.6-0.8): {{ analysis.quality_distribution.good.count }} reports ({{ "%.1f"|format(analysis.quality_distribution.good.percentage) }}%)</li>
                <li>Fair (0.4-0.6): {{ analysis.quality_distribution.fair.count }} reports ({{ "%.1f"|format(analysis.quality_distribution.fair.percentage) }}%)</li>
                <li>Poor (<0.4): {{ analysis.quality_distribution.poor.count }} reports ({{ "%.1f"|format(analysis.quality_distribution.poor.percentage) }}%)</li>
            </ul>
        </div>
        {% endif %}
        {% endfor %}
    </div>
</body>
</html>
        """

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample evaluation results for testing
    sample_evaluation_results = {
        'batch_results': {
            'report_1': {'BLEU': 0.65, 'ROUGE': 0.72, 'METEOR': 0.68, 'BERTScore': 0.75},
            'report_2': {'BLEU': 0.58, 'ROUGE': 0.69, 'METEOR': 0.62, 'BERTScore': 0.71},
            'report_3': {'BLEU': 0.71, 'ROUGE': 0.78, 'METEOR': 0.74, 'BERTScore': 0.82},
            'report_4': {'BLEU': 0.63, 'ROUGE': 0.70, 'METEOR': 0.66, 'BERTScore': 0.73},
            'report_5': {'BLEU': 0.69, 'ROUGE': 0.76, 'METEOR': 0.72, 'BERTScore': 0.79}
        }
    }
    
    # Test the report generator
    generator = ReportGenerator()
    
    print("Generating comprehensive evaluation report...")
    report_paths = generator.generate_comprehensive_report(
        sample_evaluation_results,
        report_title="Sample Medical Report Evaluation Analysis"
    )
    
    print("Report generation completed!")
    for format_type, path in report_paths.items():
        print(f"  {format_type.upper()}: {path}")
    
    print("✅ ReportGenerator testing completed successfully!") 