"""
Comparative Analysis System for Medical Report Evaluation

This module provides comprehensive comparison capabilities for evaluation results
across different model versions, including delta analysis, improvement tracking,
and regression detection.

Author: Medical Report Evaluation System
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
from datetime import datetime
from scipy import stats

class ComparativeAnalyzer:
    """
    Compares evaluation results across different model versions or time periods.
    
    Features:
    - Delta analysis between model versions
    - Improvement/regression tracking
    - Statistical significance testing
    - Comparative visualizations
    - Performance trend analysis
    """
    
    def __init__(self, output_dir: str = "comparative_analysis"):
        """
        Initialize the comparative analyzer.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ComparativeAnalyzer initialized with output directory: {output_dir}")
    
    def compare_model_versions(self, 
                             baseline_results: Dict[str, Dict[str, float]], 
                             comparison_results: Dict[str, Dict[str, float]],
                             baseline_name: str = "Baseline",
                             comparison_name: str = "Comparison") -> Dict[str, Any]:
        """
        Compare results between two model versions.
        
        Args:
            baseline_results: Results from baseline model
            comparison_results: Results from comparison model
            baseline_name: Name for baseline model
            comparison_name: Name for comparison model
            
        Returns:
            Dictionary with comprehensive comparison analysis
        """
        try:
            # Convert to DataFrames
            baseline_df = pd.DataFrame(baseline_results).T
            comparison_df = pd.DataFrame(comparison_results).T
            
            # Ensure same reports are compared
            common_reports = set(baseline_df.index) & set(comparison_df.index)
            if not common_reports:
                raise ValueError("No common reports found between baseline and comparison results")
            
            baseline_df = baseline_df.loc[list(common_reports)]
            comparison_df = comparison_df.loc[list(common_reports)]
            
            # Ensure same metrics are compared
            common_metrics = set(baseline_df.columns) & set(comparison_df.columns)
            if not common_metrics:
                raise ValueError("No common metrics found between baseline and comparison results")
            
            baseline_df = baseline_df[list(common_metrics)]
            comparison_df = comparison_df[list(common_metrics)]
            
            # Calculate deltas
            delta_df = comparison_df - baseline_df
            
            # Metric-level analysis
            metric_analysis = {}
            for metric in common_metrics:
                baseline_scores = baseline_df[metric]
                comparison_scores = comparison_df[metric]
                deltas = delta_df[metric]
                
                # Statistical tests
                t_stat, p_value = stats.ttest_rel(comparison_scores, baseline_scores)
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(comparison_scores, baseline_scores)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(baseline_scores) - 1) * baseline_scores.var() + 
                                    (len(comparison_scores) - 1) * comparison_scores.var()) / 
                                   (len(baseline_scores) + len(comparison_scores) - 2))
                cohens_d = (comparison_scores.mean() - baseline_scores.mean()) / pooled_std
                
                # Improvement/regression analysis
                improvements = deltas[deltas > 0]
                regressions = deltas[deltas < 0]
                no_change = deltas[deltas == 0]
                
                metric_analysis[metric] = {
                    'baseline_stats': {
                        'mean': float(baseline_scores.mean()),
                        'std': float(baseline_scores.std()),
                        'median': float(baseline_scores.median())
                    },
                    'comparison_stats': {
                        'mean': float(comparison_scores.mean()),
                        'std': float(comparison_scores.std()),
                        'median': float(comparison_scores.median())
                    },
                    'delta_analysis': {
                        'mean_delta': float(deltas.mean()),
                        'median_delta': float(deltas.median()),
                        'std_delta': float(deltas.std()),
                        'min_delta': float(deltas.min()),
                        'max_delta': float(deltas.max()),
                        'absolute_mean_delta': float(deltas.abs().mean())
                    },
                    'statistical_tests': {
                        'paired_t_test': {
                            'statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        },
                        'wilcoxon_test': {
                            'statistic': float(wilcoxon_stat),
                            'p_value': float(wilcoxon_p),
                            'significant': wilcoxon_p < 0.05
                        },
                        'effect_size': {
                            'cohens_d': float(cohens_d),
                            'interpretation': self._interpret_effect_size(cohens_d)
                        }
                    },
                    'change_distribution': {
                        'improvements': {
                            'count': len(improvements),
                            'percentage': (len(improvements) / len(deltas)) * 100,
                            'mean_improvement': float(improvements.mean()) if len(improvements) > 0 else 0
                        },
                        'regressions': {
                            'count': len(regressions),
                            'percentage': (len(regressions) / len(deltas)) * 100,
                            'mean_regression': float(regressions.mean()) if len(regressions) > 0 else 0
                        },
                        'no_change': {
                            'count': len(no_change),
                            'percentage': (len(no_change) / len(deltas)) * 100
                        }
                    }
                }
            
            # Overall analysis
            overall_baseline = baseline_df.mean(axis=1)
            overall_comparison = comparison_df.mean(axis=1)
            overall_delta = overall_comparison - overall_baseline
            
            # Overall statistical tests
            overall_t_stat, overall_p_value = stats.ttest_rel(overall_comparison, overall_baseline)
            overall_wilcoxon_stat, overall_wilcoxon_p = stats.wilcoxon(overall_comparison, overall_baseline)
            
            # Overall effect size
            overall_pooled_std = np.sqrt(((len(overall_baseline) - 1) * overall_baseline.var() + 
                                        (len(overall_comparison) - 1) * overall_comparison.var()) / 
                                       (len(overall_baseline) + len(overall_comparison) - 2))
            overall_cohens_d = (overall_comparison.mean() - overall_baseline.mean()) / overall_pooled_std
            
            comparison_result = {
                'comparison_info': {
                    'baseline_name': baseline_name,
                    'comparison_name': comparison_name,
                    'timestamp': datetime.now().isoformat(),
                    'common_reports': len(common_reports),
                    'common_metrics': list(common_metrics)
                },
                'metric_analysis': metric_analysis,
                'overall_analysis': {
                    'baseline_overall': {
                        'mean': float(overall_baseline.mean()),
                        'std': float(overall_baseline.std()),
                        'median': float(overall_baseline.median())
                    },
                    'comparison_overall': {
                        'mean': float(overall_comparison.mean()),
                        'std': float(overall_comparison.std()),
                        'median': float(overall_comparison.median())
                    },
                    'overall_delta': {
                        'mean': float(overall_delta.mean()),
                        'median': float(overall_delta.median()),
                        'std': float(overall_delta.std())
                    },
                    'statistical_tests': {
                        'paired_t_test': {
                            'statistic': float(overall_t_stat),
                            'p_value': float(overall_p_value),
                            'significant': overall_p_value < 0.05
                        },
                        'wilcoxon_test': {
                            'statistic': float(overall_wilcoxon_stat),
                            'p_value': float(overall_wilcoxon_p),
                            'significant': overall_wilcoxon_p < 0.05
                        },
                        'effect_size': {
                            'cohens_d': float(overall_cohens_d),
                            'interpretation': self._interpret_effect_size(overall_cohens_d)
                        }
                    }
                },
                'summary': self._generate_comparison_summary(metric_analysis, overall_delta.mean())
            }
            
            self.logger.info(f"Model comparison completed: {baseline_name} vs {comparison_name}")
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"Error comparing model versions: {e}")
            raise
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "Negligible effect"
        elif abs_d < 0.5:
            return "Small effect"
        elif abs_d < 0.8:
            return "Medium effect"
        else:
            return "Large effect"
    
    def _generate_comparison_summary(self, metric_analysis: Dict[str, Any], overall_delta: float) -> Dict[str, Any]:
        """Generate summary of comparison results."""
        significant_improvements = []
        significant_regressions = []
        
        for metric, analysis in metric_analysis.items():
            delta = analysis['delta_analysis']['mean_delta']
            significant = analysis['statistical_tests']['paired_t_test']['significant']
            
            if significant:
                if delta > 0:
                    significant_improvements.append(metric)
                elif delta < 0:
                    significant_regressions.append(metric)
        
        if overall_delta > 0.01:
            overall_verdict = "Improvement"
        elif overall_delta < -0.01:
            overall_verdict = "Regression"
        else:
            overall_verdict = "No significant change"
        
        return {
            'overall_verdict': overall_verdict,
            'overall_delta': float(overall_delta),
            'significant_improvements': significant_improvements,
            'significant_regressions': significant_regressions,
            'metrics_improved': len(significant_improvements),
            'metrics_regressed': len(significant_regressions),
            'recommendation': self._generate_recommendation(overall_verdict, significant_improvements, significant_regressions)
        }
    
    def _generate_recommendation(self, verdict: str, improvements: List[str], regressions: List[str]) -> str:
        """Generate recommendation based on comparison results."""
        if verdict == "Improvement" and not regressions:
            return "Strong recommendation to deploy the new model version."
        elif verdict == "Improvement" and regressions:
            return "Consider deployment with monitoring of regressed metrics."
        elif verdict == "Regression":
            return "Do not deploy. Investigate causes of performance regression."
        else:
            return "No significant difference. Consider other factors for deployment decision."
    
    def create_comparison_visualizations(self, 
                                       comparison_results: Dict[str, Any],
                                       save_path: Optional[str] = None) -> str:
        """
        Create comprehensive comparison visualizations.
        
        Args:
            comparison_results: Results from compare_model_versions
            save_path: Optional custom save path
            
        Returns:
            Path to saved visualizations
        """
        try:
            baseline_name = comparison_results['comparison_info']['baseline_name']
            comparison_name = comparison_results['comparison_info']['comparison_name']
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Metric comparison bar chart
            metrics = list(comparison_results['metric_analysis'].keys())
            baseline_means = [comparison_results['metric_analysis'][m]['baseline_stats']['mean'] for m in metrics]
            comparison_means = [comparison_results['metric_analysis'][m]['comparison_stats']['mean'] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, baseline_means, width, label=baseline_name, alpha=0.8)
            ax1.bar(x + width/2, comparison_means, width, label=comparison_name, alpha=0.8)
            
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Average Score')
            ax1.set_title('Metric Comparison', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Delta analysis
            deltas = [comparison_results['metric_analysis'][m]['delta_analysis']['mean_delta'] for m in metrics]
            colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in deltas]
            
            ax2.bar(metrics, deltas, color=colors, alpha=0.7)
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Delta (Comparison - Baseline)')
            ax2.set_title('Performance Delta Analysis', fontweight='bold')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 3. Effect size visualization
            effect_sizes = [comparison_results['metric_analysis'][m]['statistical_tests']['effect_size']['cohens_d'] for m in metrics]
            
            ax3.barh(metrics, effect_sizes, color='skyblue', alpha=0.7)
            ax3.set_xlabel("Cohen's d (Effect Size)")
            ax3.set_title('Effect Size Analysis', fontweight='bold')
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax3.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Small')
            ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Medium')
            ax3.axvline(x=0.8, color='darkred', linestyle='--', alpha=0.5, label='Large')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Statistical significance
            p_values = [comparison_results['metric_analysis'][m]['statistical_tests']['paired_t_test']['p_value'] for m in metrics]
            significance = ['Significant' if p < 0.05 else 'Not Significant' for p in p_values]
            
            sig_counts = pd.Series(significance).value_counts()
            colors_pie = ['lightcoral', 'lightblue']
            
            ax4.pie(sig_counts.values, labels=sig_counts.index, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Statistical Significance Distribution', fontweight='bold')
            
            # Main title
            fig.suptitle(f'Comparative Analysis: {baseline_name} vs {comparison_name}', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save visualization
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.output_dir / f"comparison_visualization_{timestamp}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Comparison visualizations saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Error creating comparison visualizations: {e}")
            raise
    
    def detect_regressions(self, 
                         current_results: Dict[str, Dict[str, float]], 
                         reference_results: Dict[str, Dict[str, float]],
                         threshold: float = 0.05) -> Dict[str, Any]:
        """
        Detect performance regressions compared to reference results.
        
        Args:
            current_results: Current model results
            reference_results: Reference (baseline) results
            threshold: Minimum delta to consider as regression
            
        Returns:
            Dictionary with regression detection results
        """
        try:
            # Convert to DataFrames
            current_df = pd.DataFrame(current_results).T
            reference_df = pd.DataFrame(reference_results).T
            
            # Find common reports and metrics
            common_reports = set(current_df.index) & set(reference_df.index)
            common_metrics = set(current_df.columns) & set(reference_df.columns)
            
            if not common_reports or not common_metrics:
                raise ValueError("No common reports or metrics found for regression detection")
            
            current_df = current_df.loc[list(common_reports), list(common_metrics)]
            reference_df = reference_df.loc[list(common_reports), list(common_metrics)]
            
            # Calculate deltas
            delta_df = current_df - reference_df
            
            # Detect regressions
            regressions = {}
            for metric in common_metrics:
                metric_deltas = delta_df[metric]
                
                # Find significant regressions
                significant_regressions = metric_deltas[metric_deltas < -threshold]
                
                if len(significant_regressions) > 0:
                    regressions[metric] = {
                        'regression_count': len(significant_regressions),
                        'regression_percentage': (len(significant_regressions) / len(metric_deltas)) * 100,
                        'average_regression': float(significant_regressions.mean()),
                        'worst_regression': float(significant_regressions.min()),
                        'regressed_reports': significant_regressions.index.tolist(),
                        'regression_values': significant_regressions.tolist()
                    }
            
            # Overall regression analysis
            overall_current = current_df.mean(axis=1)
            overall_reference = reference_df.mean(axis=1)
            overall_delta = overall_current - overall_reference
            
            overall_regressions = overall_delta[overall_delta < -threshold]
            
            regression_result = {
                'detection_info': {
                    'threshold': threshold,
                    'timestamp': datetime.now().isoformat(),
                    'common_reports': len(common_reports),
                    'common_metrics': list(common_metrics)
                },
                'metric_regressions': regressions,
                'overall_regression': {
                    'regression_count': len(overall_regressions),
                    'regression_percentage': (len(overall_regressions) / len(overall_delta)) * 100,
                    'average_regression': float(overall_regressions.mean()) if len(overall_regressions) > 0 else 0,
                    'worst_regression': float(overall_regressions.min()) if len(overall_regressions) > 0 else 0,
                    'regressed_reports': overall_regressions.index.tolist()
                },
                'summary': {
                    'total_metrics_regressed': len(regressions),
                    'total_reports_regressed': len(overall_regressions),
                    'severity': self._assess_regression_severity(regressions, overall_regressions, threshold),
                    'recommendation': self._generate_regression_recommendation(regressions, overall_regressions)
                }
            }
            
            self.logger.info(f"Regression detection completed. Found {len(regressions)} regressed metrics")
            return regression_result
            
        except Exception as e:
            self.logger.error(f"Error detecting regressions: {e}")
            raise
    
    def _assess_regression_severity(self, metric_regressions: Dict[str, Any], 
                                  overall_regressions: pd.Series, threshold: float) -> str:
        """Assess the severity of detected regressions."""
        if not metric_regressions and len(overall_regressions) == 0:
            return "No regressions detected"
        
        # Calculate severity based on number of regressed metrics and magnitude
        regressed_metrics_count = len(metric_regressions)
        total_regression_magnitude = sum(abs(reg['average_regression']) for reg in metric_regressions.values())
        
        if regressed_metrics_count == 0:
            return "Low"
        elif regressed_metrics_count <= 2 and total_regression_magnitude < threshold * 3:
            return "Medium"
        else:
            return "High"
    
    def _generate_regression_recommendation(self, metric_regressions: Dict[str, Any], 
                                          overall_regressions: pd.Series) -> str:
        """Generate recommendation based on regression analysis."""
        if not metric_regressions and len(overall_regressions) == 0:
            return "No action needed. No significant regressions detected."
        elif len(metric_regressions) <= 1:
            return "Monitor closely. Minor regressions detected in limited metrics."
        else:
            return "Investigate immediately. Significant regressions detected across multiple metrics."
    
    def export_comparative_report(self, 
                                analysis_results: Dict[str, Any], 
                                filename: str = None) -> str:
        """
        Export comprehensive comparative analysis report.
        
        Args:
            analysis_results: Results from comparative analysis methods
            filename: Optional custom filename
            
        Returns:
            Path to exported report
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comparative_analysis_report_{timestamp}.json"
            
            report_path = self.output_dir / filename
            
            with open(report_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            self.logger.info(f"Comparative analysis report exported to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting comparative report: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample data for testing
    np.random.seed(42)
    
    # Generate baseline results
    baseline_results = {}
    for i in range(20):
        report_id = f"report_{i+1}"
        baseline_results[report_id] = {
            'BLEU': np.random.normal(0.65, 0.08),
            'ROUGE': np.random.normal(0.70, 0.06),
            'METEOR': np.random.normal(0.68, 0.07),
            'BERTScore': np.random.normal(0.72, 0.05)
        }
    
    # Generate improved results (with some regressions)
    comparison_results = {}
    for i in range(20):
        report_id = f"report_{i+1}"
        # Generally improved, but with some regressions
        improvement_factor = 1.05 if i < 15 else 0.95  # Most improve, some regress
        
        comparison_results[report_id] = {
            'BLEU': baseline_results[report_id]['BLEU'] * improvement_factor + np.random.normal(0, 0.02),
            'ROUGE': baseline_results[report_id]['ROUGE'] * improvement_factor + np.random.normal(0, 0.02),
            'METEOR': baseline_results[report_id]['METEOR'] * improvement_factor + np.random.normal(0, 0.02),
            'BERTScore': baseline_results[report_id]['BERTScore'] * improvement_factor + np.random.normal(0, 0.02)
        }
    
    # Test the analyzer
    analyzer = ComparativeAnalyzer()
    
    print("Comparing model versions...")
    comparison_analysis = analyzer.compare_model_versions(
        baseline_results, comparison_results, 
        "Baseline Model", "Improved Model"
    )
    print(f"Comparison completed: {comparison_analysis['summary']['overall_verdict']}")
    
    print("Creating comparison visualizations...")
    analyzer.create_comparison_visualizations(comparison_analysis)
    
    print("Detecting regressions...")
    regression_analysis = analyzer.detect_regressions(comparison_results, baseline_results)
    print(f"Regressions detected: {regression_analysis['summary']['severity']}")
    
    print("Exporting comparative report...")
    report_path = analyzer.export_comparative_report({
        'comparison_analysis': comparison_analysis,
        'regression_analysis': regression_analysis
    })
    
    print("✅ ComparativeAnalyzer testing completed successfully!") 