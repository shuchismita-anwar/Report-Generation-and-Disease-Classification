"""
Report Quality Distribution Analyzer for Medical Report Evaluation

This module provides comprehensive analysis of report quality score distributions,
including statistical analysis, outlier detection, and quality trend identification.

Author: Medical Report Evaluation System
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from datetime import datetime

class DistributionAnalyzer:
    """
    Analyzes distribution of report quality scores.
    
    Features:
    - Score distribution analysis
    - Statistical summaries
    - Outlier detection
    - Quality trend analysis
    - Comparative distribution analysis
    """
    
    def __init__(self, output_dir: str = "distribution_analysis"):
        """
        Initialize the distribution analyzer.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DistributionAnalyzer initialized with output directory: {output_dir}")
    
    def analyze_score_distribution(self, 
                                 results: Dict[str, Dict[str, float]], 
                                 metric: str = None) -> Dict[str, Any]:
        """
        Analyze the distribution of scores for a specific metric or overall scores.
        
        Args:
            results: Dictionary with report_id -> {metric_name: score}
            metric: Specific metric to analyze (None for overall scores)
            
        Returns:
            Dictionary with distribution analysis results
        """
        try:
            df = pd.DataFrame(results).T
            
            if metric is None:
                # Analyze overall scores (mean across all metrics)
                scores = df.mean(axis=1)
                analysis_name = "Overall Scores"
            else:
                if metric not in df.columns:
                    raise ValueError(f"Metric '{metric}' not found in results")
                scores = df[metric]
                analysis_name = f"{metric} Scores"
            
            # Basic statistics
            basic_stats = {
                'count': len(scores),
                'mean': float(scores.mean()),
                'median': float(scores.median()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'range': float(scores.max() - scores.min()),
                'q1': float(scores.quantile(0.25)),
                'q3': float(scores.quantile(0.75)),
                'iqr': float(scores.quantile(0.75) - scores.quantile(0.25))
            }
            
            # Distribution shape analysis
            skewness = float(stats.skew(scores))
            kurtosis = float(stats.kurtosis(scores))
            
            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(scores)
            
            # Outlier detection using IQR method
            q1, q3 = basic_stats['q1'], basic_stats['q3']
            iqr = basic_stats['iqr']
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = scores[(scores < lower_bound) | (scores > upper_bound)]
            outlier_indices = outliers.index.tolist()
            
            # Quality categories
            quality_categories = self._categorize_quality(scores)
            
            analysis_result = {
                'analysis_name': analysis_name,
                'metric': metric,
                'timestamp': datetime.now().isoformat(),
                'basic_statistics': basic_stats,
                'distribution_shape': {
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'skewness_interpretation': self._interpret_skewness(skewness),
                    'kurtosis_interpretation': self._interpret_kurtosis(kurtosis)
                },
                'normality_test': {
                    'shapiro_wilk_statistic': float(shapiro_stat),
                    'shapiro_wilk_p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                },
                'outliers': {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(scores)) * 100,
                    'indices': outlier_indices,
                    'values': outliers.tolist(),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                },
                'quality_categories': quality_categories
            }
            
            self.logger.info(f"Distribution analysis completed for {analysis_name}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing score distribution: {e}")
            raise
    
    def _categorize_quality(self, scores: pd.Series) -> Dict[str, Any]:
        """
        Categorize scores into quality levels.
        
        Args:
            scores: Series of scores
            
        Returns:
            Dictionary with quality categorization
        """
        # Define quality thresholds
        excellent_threshold = 0.8
        good_threshold = 0.6
        fair_threshold = 0.4
        
        excellent = scores[scores >= excellent_threshold]
        good = scores[(scores >= good_threshold) & (scores < excellent_threshold)]
        fair = scores[(scores >= fair_threshold) & (scores < good_threshold)]
        poor = scores[scores < fair_threshold]
        
        total_count = len(scores)
        
        return {
            'excellent': {
                'count': len(excellent),
                'percentage': (len(excellent) / total_count) * 100,
                'threshold': f">= {excellent_threshold}"
            },
            'good': {
                'count': len(good),
                'percentage': (len(good) / total_count) * 100,
                'threshold': f"{good_threshold} - {excellent_threshold}"
            },
            'fair': {
                'count': len(fair),
                'percentage': (len(fair) / total_count) * 100,
                'threshold': f"{fair_threshold} - {good_threshold}"
            },
            'poor': {
                'count': len(poor),
                'percentage': (len(poor) / total_count) * 100,
                'threshold': f"< {fair_threshold}"
            }
        }
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value."""
        if abs(skewness) < 0.5:
            return "Approximately symmetric"
        elif skewness > 0.5:
            return "Right-skewed (positive skew)"
        else:
            return "Left-skewed (negative skew)"
    
    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis value."""
        if abs(kurtosis) < 0.5:
            return "Mesokurtic (normal-like)"
        elif kurtosis > 0.5:
            return "Leptokurtic (heavy-tailed)"
        else:
            return "Platykurtic (light-tailed)"
    
    def create_distribution_plots(self, 
                                results: Dict[str, Dict[str, float]], 
                                metric: str = None,
                                save_path: Optional[str] = None) -> str:
        """
        Create comprehensive distribution plots.
        
        Args:
            results: Dictionary with report_id -> {metric_name: score}
            metric: Specific metric to analyze (None for overall scores)
            save_path: Optional custom save path
            
        Returns:
            Path to saved plots
        """
        try:
            df = pd.DataFrame(results).T
            
            if metric is None:
                scores = df.mean(axis=1)
                plot_title = "Overall Score Distribution"
                file_suffix = "overall"
            else:
                if metric not in df.columns:
                    raise ValueError(f"Metric '{metric}' not found in results")
                scores = df[metric]
                plot_title = f"{metric} Score Distribution"
                file_suffix = metric.lower().replace(' ', '_')
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Histogram with KDE
            ax1.hist(scores, bins=20, alpha=0.7, density=True, edgecolor='black')
            scores.plot.kde(ax=ax1, color='red', linewidth=2)
            ax1.set_title(f'{plot_title} - Histogram with KDE', fontweight='bold')
            ax1.set_xlabel('Score')
            ax1.set_ylabel('Density')
            ax1.grid(True, alpha=0.3)
            
            # Add mean and median lines
            ax1.axvline(scores.mean(), color='blue', linestyle='--', label=f'Mean: {scores.mean():.3f}')
            ax1.axvline(scores.median(), color='green', linestyle='--', label=f'Median: {scores.median():.3f}')
            ax1.legend()
            
            # 2. Box plot with outliers
            box_plot = ax2.boxplot(scores, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            ax2.set_title(f'{plot_title} - Box Plot', fontweight='bold')
            ax2.set_ylabel('Score')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f"Mean: {scores.mean():.3f}\n"
            stats_text += f"Std: {scores.std():.3f}\n"
            stats_text += f"Skew: {stats.skew(scores):.3f}\n"
            stats_text += f"Kurt: {stats.kurtosis(scores):.3f}"
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 3. Q-Q plot for normality assessment
            stats.probplot(scores, dist="norm", plot=ax3)
            ax3.set_title(f'{plot_title} - Q-Q Plot (Normality)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # 4. Quality categories pie chart
            quality_cats = self._categorize_quality(scores)
            categories = ['Excellent', 'Good', 'Fair', 'Poor']
            sizes = [quality_cats[cat.lower()]['count'] for cat in categories]
            colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=categories, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            ax4.set_title(f'{plot_title} - Quality Categories', fontweight='bold')
            
            plt.tight_layout()
            
            # Save plots
            if save_path is None:
                save_path = self.output_dir / f"distribution_analysis_{file_suffix}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Distribution plots saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Error creating distribution plots: {e}")
            raise
    
    def detect_outliers(self, 
                       results: Dict[str, Dict[str, float]], 
                       method: str = 'iqr') -> Dict[str, Any]:
        """
        Detect outliers in the evaluation results.
        
        Args:
            results: Dictionary with report_id -> {metric_name: score}
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            Dictionary with outlier detection results
        """
        try:
            df = pd.DataFrame(results).T
            outlier_results = {}
            
            for metric in df.columns:
                scores = df[metric]
                
                if method == 'iqr':
                    outliers = self._detect_outliers_iqr(scores)
                elif method == 'zscore':
                    outliers = self._detect_outliers_zscore(scores)
                elif method == 'isolation_forest':
                    outliers = self._detect_outliers_isolation_forest(scores)
                else:
                    raise ValueError(f"Unknown outlier detection method: {method}")
                
                outlier_results[metric] = outliers
            
            # Overall outlier analysis
            overall_scores = df.mean(axis=1)
            if method == 'iqr':
                overall_outliers = self._detect_outliers_iqr(overall_scores)
            elif method == 'zscore':
                overall_outliers = self._detect_outliers_zscore(overall_scores)
            else:
                overall_outliers = self._detect_outliers_isolation_forest(overall_scores)
            
            outlier_results['overall'] = overall_outliers
            
            return {
                'method': method,
                'timestamp': datetime.now().isoformat(),
                'outliers_by_metric': outlier_results,
                'summary': {
                    'total_reports': len(df),
                    'outliers_found': sum(len(outliers['indices']) for outliers in outlier_results.values()),
                    'outlier_percentage': (sum(len(outliers['indices']) for outliers in outlier_results.values()) / (len(df) * len(df.columns))) * 100
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {e}")
            raise
    
    def _detect_outliers_iqr(self, scores: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        q1 = scores.quantile(0.25)
        q3 = scores.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = scores[(scores < lower_bound) | (scores > upper_bound)]
        
        return {
            'method': 'IQR',
            'count': len(outliers),
            'indices': outliers.index.tolist(),
            'values': outliers.tolist(),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'percentage': (len(outliers) / len(scores)) * 100
        }
    
    def _detect_outliers_zscore(self, scores: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(scores))
        outlier_mask = z_scores > threshold
        outliers = scores[outlier_mask]
        
        return {
            'method': 'Z-Score',
            'threshold': threshold,
            'count': len(outliers),
            'indices': outliers.index.tolist(),
            'values': outliers.tolist(),
            'z_scores': z_scores[outlier_mask].tolist(),
            'percentage': (len(outliers) / len(scores)) * 100
        }
    
    def _detect_outliers_isolation_forest(self, scores: pd.Series) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape for sklearn
            X = scores.values.reshape(-1, 1)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            
            # Get outliers (labeled as -1)
            outlier_mask = outlier_labels == -1
            outliers = scores[outlier_mask]
            
            return {
                'method': 'Isolation Forest',
                'count': len(outliers),
                'indices': outliers.index.tolist(),
                'values': outliers.tolist(),
                'percentage': (len(outliers) / len(scores)) * 100
            }
            
        except ImportError:
            self.logger.warning("sklearn not available, falling back to IQR method")
            return self._detect_outliers_iqr(scores)
    
    def analyze_quality_trends(self, 
                             results_over_time: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze quality trends over time.
        
        Args:
            results_over_time: List of result dictionaries with timestamps
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            if not results_over_time:
                raise ValueError("No results provided for trend analysis")
            
            # Extract time series data
            timestamps = []
            metrics_over_time = {}
            
            for result in results_over_time:
                if 'timestamp' not in result or 'metrics' not in result:
                    continue
                
                timestamps.append(result['timestamp'])
                
                for metric, score in result['metrics'].items():
                    if metric not in metrics_over_time:
                        metrics_over_time[metric] = []
                    metrics_over_time[metric].append(score)
            
            if not timestamps:
                raise ValueError("No valid timestamp data found")
            
            # Analyze trends for each metric
            trend_analysis = {}
            
            for metric, scores in metrics_over_time.items():
                if len(scores) < 2:
                    continue
                
                # Calculate trend statistics
                scores_array = np.array(scores)
                
                # Linear trend
                x = np.arange(len(scores))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores_array)
                
                # Moving average (if enough data points)
                if len(scores) >= 5:
                    moving_avg = pd.Series(scores).rolling(window=3).mean().tolist()
                else:
                    moving_avg = scores
                
                # Trend direction
                if slope > 0.01:
                    trend_direction = "Improving"
                elif slope < -0.01:
                    trend_direction = "Declining"
                else:
                    trend_direction = "Stable"
                
                trend_analysis[metric] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'trend_direction': trend_direction,
                    'trend_strength': abs(float(r_value)),
                    'moving_average': moving_avg,
                    'volatility': float(np.std(scores_array)),
                    'improvement_rate': float(slope * len(scores))  # Total improvement over period
                }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'analysis_period': {
                    'start': timestamps[0] if timestamps else None,
                    'end': timestamps[-1] if timestamps else None,
                    'data_points': len(timestamps)
                },
                'trends_by_metric': trend_analysis,
                'overall_trend': self._calculate_overall_trend(trend_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing quality trends: {e}")
            raise
    
    def _calculate_overall_trend(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall trend across all metrics."""
        if not trend_analysis:
            return {}
        
        slopes = [trend['slope'] for trend in trend_analysis.values()]
        r_squared_values = [trend['r_squared'] for trend in trend_analysis.values()]
        
        overall_slope = np.mean(slopes)
        overall_r_squared = np.mean(r_squared_values)
        
        if overall_slope > 0.01:
            overall_direction = "Improving"
        elif overall_slope < -0.01:
            overall_direction = "Declining"
        else:
            overall_direction = "Stable"
        
        return {
            'average_slope': float(overall_slope),
            'average_r_squared': float(overall_r_squared),
            'overall_direction': overall_direction,
            'consistency': float(1 - np.std(slopes))  # Higher = more consistent trends
        }
    
    def export_analysis_report(self, 
                             analysis_results: Dict[str, Any], 
                             filename: str = None) -> str:
        """
        Export comprehensive analysis report.
        
        Args:
            analysis_results: Results from various analysis methods
            filename: Optional custom filename
            
        Returns:
            Path to exported report
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"distribution_analysis_report_{timestamp}.json"
            
            report_path = self.output_dir / filename
            
            with open(report_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            self.logger.info(f"Analysis report exported to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis report: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample data for testing
    np.random.seed(42)
    sample_results = {}
    
    # Generate sample data with some outliers
    for i in range(50):
        report_id = f"report_{i+1}"
        
        # Normal scores with some variation
        bleu = np.random.normal(0.65, 0.1)
        rouge = np.random.normal(0.70, 0.08)
        meteor = np.random.normal(0.68, 0.09)
        bert = np.random.normal(0.72, 0.07)
        
        # Add some outliers
        if i in [5, 15, 25]:
            bleu *= 0.5  # Make some scores much lower
        if i in [10, 20, 30]:
            rouge *= 1.3  # Make some scores higher
        
        # Ensure scores are in valid range
        sample_results[report_id] = {
            'BLEU': max(0, min(1, bleu)),
            'ROUGE': max(0, min(1, rouge)),
            'METEOR': max(0, min(1, meteor)),
            'BERTScore': max(0, min(1, bert))
        }
    
    # Test the analyzer
    analyzer = DistributionAnalyzer()
    
    print("Analyzing score distribution...")
    distribution_analysis = analyzer.analyze_score_distribution(sample_results)
    print(f"Distribution analysis completed: {distribution_analysis['basic_statistics']}")
    
    print("Creating distribution plots...")
    analyzer.create_distribution_plots(sample_results)
    
    print("Detecting outliers...")
    outliers = analyzer.detect_outliers(sample_results, method='iqr')
    print(f"Outliers detected: {outliers['summary']}")
    
    print("Exporting analysis report...")
    report_path = analyzer.export_analysis_report({
        'distribution_analysis': distribution_analysis,
        'outlier_analysis': outliers
    })
    
    print("✅ DistributionAnalyzer testing completed successfully!") 