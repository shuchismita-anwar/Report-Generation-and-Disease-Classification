"""
Metric Visualization System for Medical Report Evaluation

This module provides comprehensive visualization capabilities for evaluation metrics,
including comparison charts, correlation analysis, and performance visualization.

Author: Medical Report Evaluation System
Date: 2025
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json

class MetricVisualizer:
    """
    Creates visualizations for metric comparison and analysis.
    
    Supports:
    - Bar charts for metric comparison
    - Scatter plots for correlation analysis
    - Correlation matrices
    - Distribution plots
    - Performance trend analysis
    """
    
    def __init__(self, output_dir: str = "visualizations", style: str = "seaborn-v0_8"):
        """
        Initialize the metric visualizer.
        
        Args:
            output_dir: Directory to save visualization files
            style: Matplotlib style to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')
            logging.warning(f"Style '{style}' not found, using default")
        
        # Configure seaborn
        sns.set_palette("husl")
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"MetricVisualizer initialized with output directory: {output_dir}")
    
    def create_metric_comparison_chart(self, 
                                     results: Dict[str, Dict[str, float]], 
                                     title: str = "Metric Comparison",
                                     save_path: Optional[str] = None) -> str:
        """
        Create bar chart comparing different metrics across reports.
        
        Args:
            results: Dictionary with report_id -> {metric_name: score}
            title: Chart title
            save_path: Optional custom save path
            
        Returns:
            Path to saved chart
        """
        try:
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(results).T
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create grouped bar chart
            df.plot(kind='bar', ax=ax, width=0.8)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Reports', fontsize=12)
            ax.set_ylabel('Metric Scores', fontsize=12)
            ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save chart
            if save_path is None:
                save_path = self.output_dir / f"metric_comparison_{title.lower().replace(' ', '_')}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Metric comparison chart saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Error creating metric comparison chart: {e}")
            raise
    
    def create_correlation_matrix(self, 
                                results: Dict[str, Dict[str, float]], 
                                title: str = "Metric Correlation Matrix",
                                save_path: Optional[str] = None) -> str:
        """
        Create correlation matrix heatmap for metrics.
        
        Args:
            results: Dictionary with report_id -> {metric_name: score}
            title: Chart title
            save_path: Optional custom save path
            
        Returns:
            Path to saved chart
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(results).T
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(corr_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       square=True,
                       fmt='.3f',
                       cbar_kws={'label': 'Correlation Coefficient'},
                       ax=ax)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save chart
            if save_path is None:
                save_path = self.output_dir / f"correlation_matrix_{title.lower().replace(' ', '_')}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Correlation matrix saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Error creating correlation matrix: {e}")
            raise
    
    def create_scatter_plot(self, 
                          results: Dict[str, Dict[str, float]], 
                          metric_x: str, 
                          metric_y: str,
                          title: Optional[str] = None,
                          save_path: Optional[str] = None) -> str:
        """
        Create scatter plot comparing two metrics.
        
        Args:
            results: Dictionary with report_id -> {metric_name: score}
            metric_x: Metric for x-axis
            metric_y: Metric for y-axis
            title: Chart title
            save_path: Optional custom save path
            
        Returns:
            Path to saved chart
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(results).T
            
            if metric_x not in df.columns or metric_y not in df.columns:
                raise ValueError(f"Metrics {metric_x} or {metric_y} not found in results")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create scatter plot
            scatter = ax.scatter(df[metric_x], df[metric_y], alpha=0.6, s=50)
            
            # Add trend line
            z = np.polyfit(df[metric_x], df[metric_y], 1)
            p = np.poly1d(z)
            ax.plot(df[metric_x], p(df[metric_x]), "r--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            correlation = df[metric_x].corr(df[metric_y])
            
            if title is None:
                title = f"{metric_x} vs {metric_y} (r={correlation:.3f})"
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel(metric_x, fontsize=12)
            ax.set_ylabel(metric_y, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            if save_path is None:
                save_path = self.output_dir / f"scatter_{metric_x}_vs_{metric_y}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Scatter plot saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Error creating scatter plot: {e}")
            raise
    
    def create_distribution_plot(self, 
                               results: Dict[str, Dict[str, float]], 
                               metric: str,
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> str:
        """
        Create distribution plot for a specific metric.
        
        Args:
            results: Dictionary with report_id -> {metric_name: score}
            metric: Metric to plot distribution for
            title: Chart title
            save_path: Optional custom save path
            
        Returns:
            Path to saved chart
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(results).T
            
            if metric not in df.columns:
                raise ValueError(f"Metric {metric} not found in results")
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram
            ax1.hist(df[metric], bins=20, alpha=0.7, edgecolor='black')
            ax1.set_title(f"{metric} Distribution", fontsize=14, fontweight='bold')
            ax1.set_xlabel(f"{metric} Score", fontsize=12)
            ax1.set_ylabel("Frequency", fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(df[metric])
            ax2.set_title(f"{metric} Box Plot", fontsize=14, fontweight='bold')
            ax2.set_ylabel(f"{metric} Score", fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"Mean: {df[metric].mean():.3f}\n"
            stats_text += f"Std: {df[metric].std():.3f}\n"
            stats_text += f"Min: {df[metric].min():.3f}\n"
            stats_text += f"Max: {df[metric].max():.3f}"
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            if title is None:
                title = f"{metric} Distribution Analysis"
            
            fig.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save chart
            if save_path is None:
                save_path = self.output_dir / f"distribution_{metric.lower().replace(' ', '_')}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Distribution plot saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Error creating distribution plot: {e}")
            raise
    
    def create_performance_trend(self, 
                               results_over_time: List[Dict[str, Any]], 
                               title: str = "Performance Trend Over Time",
                               save_path: Optional[str] = None) -> str:
        """
        Create trend analysis for performance over time.
        
        Args:
            results_over_time: List of result dictionaries with timestamps
            title: Chart title
            save_path: Optional custom save path
            
        Returns:
            Path to saved chart
        """
        try:
            # Extract data for plotting
            timestamps = []
            metrics_data = {}
            
            for result in results_over_time:
                if 'timestamp' in result and 'metrics' in result:
                    timestamps.append(result['timestamp'])
                    
                    for metric, score in result['metrics'].items():
                        if metric not in metrics_data:
                            metrics_data[metric] = []
                        metrics_data[metric].append(score)
            
            if not timestamps:
                raise ValueError("No timestamp data found in results")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot trend lines for each metric
            for metric, scores in metrics_data.items():
                ax.plot(timestamps, scores, marker='o', label=metric, linewidth=2)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Metric Scores', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save chart
            if save_path is None:
                save_path = self.output_dir / f"performance_trend_{title.lower().replace(' ', '_')}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Performance trend chart saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Error creating performance trend: {e}")
            raise
    
    def create_comprehensive_dashboard(self, 
                                     results: Dict[str, Dict[str, float]], 
                                     title: str = "Evaluation Dashboard",
                                     save_path: Optional[str] = None) -> str:
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Args:
            results: Dictionary with report_id -> {metric_name: score}
            title: Dashboard title
            save_path: Optional custom save path
            
        Returns:
            Path to saved dashboard
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(results).T
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 15))
            
            # 1. Metric comparison bar chart
            ax1 = plt.subplot(2, 3, 1)
            df.mean().plot(kind='bar', ax=ax1)
            ax1.set_title('Average Metric Scores', fontweight='bold')
            ax1.set_ylabel('Score')
            plt.xticks(rotation=45)
            
            # 2. Correlation matrix
            ax2 = plt.subplot(2, 3, 2)
            corr_matrix = df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2, fmt='.2f')
            ax2.set_title('Metric Correlations', fontweight='bold')
            
            # 3. Distribution of overall scores
            ax3 = plt.subplot(2, 3, 3)
            overall_scores = df.mean(axis=1)
            ax3.hist(overall_scores, bins=15, alpha=0.7, edgecolor='black')
            ax3.set_title('Overall Score Distribution', fontweight='bold')
            ax3.set_xlabel('Average Score')
            ax3.set_ylabel('Frequency')
            
            # 4. Box plots for all metrics
            ax4 = plt.subplot(2, 3, 4)
            df.boxplot(ax=ax4)
            ax4.set_title('Metric Score Distributions', fontweight='bold')
            ax4.set_ylabel('Score')
            plt.xticks(rotation=45)
            
            # 5. Scatter plot of two main metrics (if available)
            ax5 = plt.subplot(2, 3, 5)
            if len(df.columns) >= 2:
                metric1, metric2 = df.columns[0], df.columns[1]
                ax5.scatter(df[metric1], df[metric2], alpha=0.6)
                ax5.set_xlabel(metric1)
                ax5.set_ylabel(metric2)
                ax5.set_title(f'{metric1} vs {metric2}', fontweight='bold')
            else:
                ax5.text(0.5, 0.5, 'Need at least 2 metrics\nfor scatter plot', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Scatter Plot', fontweight='bold')
            
            # 6. Summary statistics table
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('tight')
            ax6.axis('off')
            
            # Create summary statistics
            summary_stats = df.describe().round(3)
            table_data = []
            for metric in summary_stats.columns:
                table_data.append([
                    metric,
                    f"{summary_stats.loc['mean', metric]:.3f}",
                    f"{summary_stats.loc['std', metric]:.3f}",
                    f"{summary_stats.loc['min', metric]:.3f}",
                    f"{summary_stats.loc['max', metric]:.3f}"
                ])
            
            table = ax6.table(cellText=table_data,
                            colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax6.set_title('Summary Statistics', fontweight='bold')
            
            # Main title
            fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save dashboard
            if save_path is None:
                save_path = self.output_dir / f"dashboard_{title.lower().replace(' ', '_')}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Comprehensive dashboard saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive dashboard: {e}")
            raise
    
    def get_visualization_summary(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Get summary statistics for visualization.
        
        Args:
            results: Dictionary with report_id -> {metric_name: score}
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            df = pd.DataFrame(results).T
            
            summary = {
                'total_reports': len(df),
                'total_metrics': len(df.columns),
                'metric_names': list(df.columns),
                'average_scores': df.mean().to_dict(),
                'score_ranges': {
                    metric: {'min': df[metric].min(), 'max': df[metric].max()}
                    for metric in df.columns
                },
                'correlations': df.corr().to_dict(),
                'overall_statistics': df.describe().to_dict()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating visualization summary: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample data for testing
    sample_results = {
        'report_1': {'BLEU': 0.65, 'ROUGE': 0.72, 'METEOR': 0.68, 'BERTScore': 0.75},
        'report_2': {'BLEU': 0.58, 'ROUGE': 0.69, 'METEOR': 0.62, 'BERTScore': 0.71},
        'report_3': {'BLEU': 0.71, 'ROUGE': 0.78, 'METEOR': 0.74, 'BERTScore': 0.82},
        'report_4': {'BLEU': 0.63, 'ROUGE': 0.70, 'METEOR': 0.66, 'BERTScore': 0.73},
        'report_5': {'BLEU': 0.69, 'ROUGE': 0.76, 'METEOR': 0.72, 'BERTScore': 0.79}
    }
    
    # Test the visualizer
    visualizer = MetricVisualizer()
    
    print("Creating metric comparison chart...")
    visualizer.create_metric_comparison_chart(sample_results)
    
    print("Creating correlation matrix...")
    visualizer.create_correlation_matrix(sample_results)
    
    print("Creating scatter plot...")
    visualizer.create_scatter_plot(sample_results, 'BLEU', 'ROUGE')
    
    print("Creating distribution plot...")
    visualizer.create_distribution_plot(sample_results, 'BERTScore')
    
    print("Creating comprehensive dashboard...")
    visualizer.create_comprehensive_dashboard(sample_results)
    
    print("Getting visualization summary...")
    summary = visualizer.get_visualization_summary(sample_results)
    print(f"Summary: {summary}")
    
    print("✅ MetricVisualizer testing completed successfully!") 