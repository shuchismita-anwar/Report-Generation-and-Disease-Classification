#!/usr/bin/env python3
"""
Gaze Attention Metrics Statistics Calculator
===========================================
Calculate mean and standard deviation for gaze attention validation metrics
across all 400 patients from the analysis JSON files.

Author: Medical Report Evaluation System
"""

import json
import os
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def load_gaze_metrics_from_analysis_files(analysis_dir: str) -> Dict[str, List[float]]:
    """
    Load gaze attention validation metrics from all analysis JSON files.
    
    Args:
        analysis_dir (str): Directory containing analysis_<DICOM_ID>.json files
        
    Returns:
        Dict[str, List[float]]: Dictionary with metric names as keys and lists of values
    """
    metrics_data = {
        'pearson_correlation': [],
        'pearson_p_value': [],
        'jensen_shannon_divergence': [],
        'mean_squared_error': [],
        'normalized_scanpath_saliency': [],
        'human_attention_entropy': [],
        'model_attention_entropy': []
    }
    
    analysis_files = []
    
    # Find all analysis JSON files
    for filename in os.listdir(analysis_dir):
        if filename.startswith('analysis_') and filename.endswith('.json'):
            analysis_files.append(filename)
    
    print(f"Found {len(analysis_files)} analysis files")
    
    processed_count = 0
    error_count = 0
    
    for filename in analysis_files:
        file_path = os.path.join(analysis_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract gaze attention validation metrics
            gaze_validation = data.get('gaze_attention_validation', {})
            correlation_metrics = gaze_validation.get('correlation_metrics', {})
            
            if correlation_metrics:
                # Extract each metric value
                for metric_name in metrics_data.keys():
                    if metric_name in correlation_metrics:
                        value = correlation_metrics[metric_name]
                        if isinstance(value, (int, float)) and not (value != value):  # Check for NaN
                            metrics_data[metric_name].append(value)
                
                processed_count += 1
            else:
                print(f"Warning: No gaze_attention_validation found in {filename}")
                error_count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            error_count += 1
    
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors encountered: {error_count} files")
    
    # Print data counts for verification
    for metric_name, values in metrics_data.items():
        print(f"{metric_name}: {len(values)} values")
    
    return metrics_data


def calculate_statistics(metrics_data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate mean and standard deviation for each metric.
    
    Args:
        metrics_data (Dict[str, List[float]]): Raw metric values
        
    Returns:
        Dict[str, Dict[str, float]]: Statistics for each metric
    """
    statistics_results = {}
    
    for metric_name, values in metrics_data.items():
        if values:  # Only calculate if we have data
            stats = {
                'count': len(values),
                'mean': statistics.mean(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values)
            }
            statistics_results[metric_name] = stats
        else:
            print(f"Warning: No data found for {metric_name}")
            statistics_results[metric_name] = {
                'count': 0,
                'mean': 0.0,
                'std_dev': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0
            }
    
    return statistics_results


def generate_markdown_report(statistics_results: Dict[str, Dict[str, float]], output_path: str) -> None:
    """Generate markdown report with the statistics."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    markdown_content = f"""# Gaze Attention Validation Metrics - Statistical Summary
**Generated**: {timestamp}

## 📊 Overall Statistics
*Values shown as Mean ± Standard Deviation across all patients*

"""
    
    # Add each metric with its statistics
    for metric_name, stats in statistics_results.items():
        metric_display = metric_name.replace('_', ' ').title()
        
        # Format metric name for better readability
        if metric_name == 'pearson_correlation':
            metric_display = 'Pearson Correlation'
        elif metric_name == 'pearson_p_value':
            metric_display = 'Pearson P-Value'
        elif metric_name == 'jensen_shannon_divergence':
            metric_display = 'Jensen-Shannon Divergence'
        elif metric_name == 'mean_squared_error':
            metric_display = 'Mean Squared Error'
        elif metric_name == 'normalized_scanpath_saliency':
            metric_display = 'Normalized Scanpath Saliency'
        elif metric_name == 'human_attention_entropy':
            metric_display = 'Human Attention Entropy'
        elif metric_name == 'model_attention_entropy':
            metric_display = 'Model Attention Entropy'
        
        markdown_content += f"""### {metric_display}
- **Mean ± Std Dev**: {stats['mean']:.6f} ± {stats['std_dev']:.6f}
- **Median**: {stats['median']:.6f}
- **Range**: {stats['min']:.6f} to {stats['max']:.6f}
- **Sample Count**: {stats['count']} patients

"""
    
    # Add interpretation section
    markdown_content += """## 🔍 Key Insights

### Correlation Analysis
- **Pearson Correlation**: Measures linear relationship between human and AI attention
  - Values closer to 1.0 indicate better alignment
  - Values closer to 0.0 indicate poor alignment

### Divergence Analysis  
- **Jensen-Shannon Divergence**: Measures difference between attention distributions
  - Values closer to 0.0 indicate better alignment
  - Values closer to 1.0 indicate poor alignment

### Attention Quality
- **Normalized Scanpath Saliency**: How well AI attention follows human scanpaths
- **Entropy Values**: Measure of attention distribution complexity
  - Higher entropy = more distributed attention
  - Lower entropy = more focused attention

"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Markdown report saved to: {output_path}")


def generate_json_report(statistics_results: Dict[str, Dict[str, float]], output_path: str) -> None:
    """Generate JSON report with the statistics."""
    
    json_output = {
        "metadata": {
            "generated_timestamp": datetime.now().isoformat(),
            "analysis_type": "gaze_attention_validation_statistics",
            "total_metrics": len(statistics_results),
            "description": "Statistical summary of gaze attention validation metrics across all patients"
        },
        "metrics_statistics": statistics_results,
        "summary": {
            "best_performing_metric": None,
            "most_consistent_metric": None,
            "highest_variability_metric": None
        }
    }
    
    # Find best/worst performing metrics
    if statistics_results:
        # For correlation metrics, higher is better
        correlation_metrics = ['pearson_correlation', 'normalized_scanpath_saliency']
        # For divergence/error metrics, lower is better  
        divergence_metrics = ['jensen_shannon_divergence', 'mean_squared_error']
        
        # Find most consistent (lowest std dev)
        lowest_std = min(stats['std_dev'] for stats in statistics_results.values() if stats['count'] > 0)
        most_consistent = next(metric for metric, stats in statistics_results.items() 
                             if stats['std_dev'] == lowest_std and stats['count'] > 0)
        
        # Find highest variability (highest std dev)
        highest_std = max(stats['std_dev'] for stats in statistics_results.values() if stats['count'] > 0)
        highest_variability = next(metric for metric, stats in statistics_results.items() 
                                 if stats['std_dev'] == highest_std and stats['count'] > 0)
        
        json_output["summary"]["most_consistent_metric"] = {
            "metric": most_consistent,
            "std_dev": lowest_std
        }
        json_output["summary"]["highest_variability_metric"] = {
            "metric": highest_variability,
            "std_dev": highest_std
        }
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"JSON report saved to: {output_path}")


def main():
    """Main execution function."""
    
    # Define paths
    analysis_dir = "./real_analysis_results"
    output_dir = "./real_analysis_results/mean-sd-output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    markdown_output_path = os.path.join(output_dir, f"gaze_metrics_statistics_{timestamp}.md")
    json_output_path = os.path.join(output_dir, f"gaze_metrics_statistics_{timestamp}.json")
    
    print("=== Gaze Attention Metrics Statistics Calculator ===")
    print(f"Analysis directory: {analysis_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load data from analysis files
    print("Loading gaze attention metrics from analysis files...")
    metrics_data = load_gaze_metrics_from_analysis_files(analysis_dir)
    
    # Calculate statistics
    print("\nCalculating statistics...")
    statistics_results = calculate_statistics(metrics_data)
    
    # Generate reports
    print("\nGenerating reports...")
    generate_markdown_report(statistics_results, markdown_output_path)
    generate_json_report(statistics_results, json_output_path)
    
    print("\n=== Summary ===")
    for metric_name, stats in statistics_results.items():
        if stats['count'] > 0:
            print(f"{metric_name}: {stats['mean']:.6f} ± {stats['std_dev']:.6f} (n={stats['count']})")
        else:
            print(f"{metric_name}: No data available")
    
    print(f"\nReports generated successfully!")
    print(f"Markdown: {markdown_output_path}")
    print(f"JSON: {json_output_path}")


if __name__ == "__main__":
    main()
