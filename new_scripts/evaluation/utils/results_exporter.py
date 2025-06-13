#!/usr/bin/env python3
"""
Results Exporter
===============
Utility for exporting evaluation results in multiple formats (JSON, CSV, TXT, HTML).

This module provides:
- Multi-format export support
- Configurable output options
- Data formatting and validation
- Summary report generation
- Batch export capabilities

Author: Medical Report Evaluation System
Version: 1.0
"""

import json
import csv
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import html


class ResultsExporter:
    """
    Exports evaluation results in various formats with configurable options.
    
    Supports JSON, CSV, TXT, and HTML export formats with customizable
    formatting and content inclusion options.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the results exporter.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.supported_formats = ['json', 'csv', 'txt', 'html']
        self.default_precision = 4
        
        self.logger.debug("Initialized ResultsExporter")
    
    def export_results(self, results: Dict[str, Any], output_path: str,
                      format: str = 'json', include_metadata: bool = True,
                      precision: int = 4, **kwargs) -> str:
        """
        Export evaluation results to specified format.
        
        Args:
            results (Dict[str, Any]): Evaluation results to export
            output_path (str): Output file path
            format (str): Export format ('json', 'csv', 'txt', 'html')
            include_metadata (bool): Whether to include metadata
            precision (int): Decimal precision for numeric values
            **kwargs: Additional format-specific options
            
        Returns:
            str: Path to exported file
            
        Raises:
            ValueError: If format is not supported
            IOError: If file cannot be written
        """
        if format.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.supported_formats}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Format-specific export
        if format.lower() == 'json':
            return self._export_json(results, output_path, include_metadata, precision, **kwargs)
        elif format.lower() == 'csv':
            return self._export_csv(results, output_path, include_metadata, precision, **kwargs)
        elif format.lower() == 'txt':
            return self._export_txt(results, output_path, include_metadata, precision, **kwargs)
        elif format.lower() == 'html':
            return self._export_html(results, output_path, include_metadata, precision, **kwargs)
    
    def export_batch_results(self, batch_results: Dict[str, Any], output_dir: str,
                           formats: List[str] = None, base_filename: str = None,
                           **kwargs) -> Dict[str, str]:
        """
        Export batch results in multiple formats.
        
        Args:
            batch_results (Dict[str, Any]): Batch evaluation results
            output_dir (str): Output directory
            formats (List[str]): List of formats to export
            base_filename (str): Base filename (without extension)
            **kwargs: Additional export options
            
        Returns:
            Dict[str, str]: Mapping of format to exported file path
        """
        if formats is None:
            formats = ['json', 'csv', 'txt']
        
        if base_filename is None:
            batch_id = batch_results.get('batch_id', 'batch_results')
            base_filename = f"{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        exported_files = {}
        
        for format in formats:
            try:
                output_path = os.path.join(output_dir, f"{base_filename}.{format}")
                exported_path = self.export_results(batch_results, output_path, format, **kwargs)
                exported_files[format] = exported_path
                self.logger.info(f"Exported {format.upper()} to {exported_path}")
            except Exception as e:
                self.logger.error(f"Error exporting {format}: {e}")
                exported_files[format] = None
        
        return exported_files
    
    def _export_json(self, results: Dict[str, Any], output_path: str,
                    include_metadata: bool, precision: int, **kwargs) -> str:
        """Export results as JSON."""
        # Format numeric values with specified precision
        formatted_results = self._format_numeric_values(results, precision)
        
        # Remove metadata if not requested
        if not include_metadata:
            formatted_results = self._remove_metadata(formatted_results)
        
        # JSON export options
        indent = kwargs.get('indent', 2)
        ensure_ascii = kwargs.get('ensure_ascii', False)
        sort_keys = kwargs.get('sort_keys', False)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_results, f, indent=indent, 
                         ensure_ascii=ensure_ascii, sort_keys=sort_keys)
            
            self.logger.debug(f"Exported JSON to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting JSON: {e}")
            raise IOError(f"Failed to export JSON: {e}")
    
    def _export_csv(self, results: Dict[str, Any], output_path: str,
                   include_metadata: bool, precision: int, **kwargs) -> str:
        """Export results as CSV."""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Determine if this is a single result or batch results
                if 'individual_results' in results:
                    # Batch results
                    self._write_batch_csv(writer, results, include_metadata, precision)
                else:
                    # Single result
                    self._write_single_csv(writer, results, include_metadata, precision)
            
            self.logger.debug(f"Exported CSV to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting CSV: {e}")
            raise IOError(f"Failed to export CSV: {e}")
    
    def _export_txt(self, results: Dict[str, Any], output_path: str,
                   include_metadata: bool, precision: int, **kwargs) -> str:
        """Export results as formatted text."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("Medical Report Evaluation Results\n")
                f.write("=" * 50 + "\n\n")
                
                # Write timestamp
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Determine result type and write accordingly
                if 'individual_results' in results:
                    self._write_batch_txt(f, results, include_metadata, precision)
                else:
                    self._write_single_txt(f, results, include_metadata, precision)
            
            self.logger.debug(f"Exported TXT to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting TXT: {e}")
            raise IOError(f"Failed to export TXT: {e}")
    
    def _export_html(self, results: Dict[str, Any], output_path: str,
                    include_metadata: bool, precision: int, **kwargs) -> str:
        """Export results as HTML."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write HTML header
                f.write(self._get_html_header())
                
                # Write content
                if 'individual_results' in results:
                    self._write_batch_html(f, results, include_metadata, precision)
                else:
                    self._write_single_html(f, results, include_metadata, precision)
                
                # Write HTML footer
                f.write(self._get_html_footer())
            
            self.logger.debug(f"Exported HTML to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting HTML: {e}")
            raise IOError(f"Failed to export HTML: {e}")
    
    def _write_batch_csv(self, writer, results: Dict[str, Any], 
                        include_metadata: bool, precision: int) -> None:
        """Write batch results to CSV."""
        # Write header
        header = ['image_id', 'overall_score', 'quality_level']
        
        # Add metric columns
        if results.get('individual_results'):
            first_result = results['individual_results'][0]
            if 'metrics' in first_result:
                for metric_name in first_result['metrics'].keys():
                    header.append(f'{metric_name}_score')
        
        if include_metadata:
            header.extend(['evaluation_time', 'timestamp'])
        
        writer.writerow(header)
        
        # Write individual results
        for result in results.get('individual_results', []):
            row = [
                result.get('image_id', ''),
                round(result.get('overall_score', 0.0), precision),
                result.get('quality_level', '')
            ]
            
            # Add metric scores
            if 'metrics' in result:
                for metric_name in first_result['metrics'].keys():
                    metric_data = result['metrics'].get(metric_name, {})
                    if isinstance(metric_data, dict) and 'error' not in metric_data:
                        # Extract primary score
                        primary_score = self._extract_primary_score_for_export(metric_name, metric_data)
                        row.append(round(primary_score, precision) if primary_score is not None else '')
                    else:
                        row.append('')
            
            if include_metadata:
                row.extend([
                    result.get('evaluation_time', ''),
                    result.get('timestamp', '')
                ])
            
            writer.writerow(row)
        
        # Write batch summary
        writer.writerow([])  # Empty row
        writer.writerow(['BATCH SUMMARY'])
        
        batch_summary = results.get('batch_summary', {})
        writer.writerow(['Total Evaluated', batch_summary.get('total_evaluated', 0)])
        writer.writerow(['Successful', batch_summary.get('successful', 0)])
        writer.writerow(['Failed', batch_summary.get('failed', 0)])
        writer.writerow(['Success Rate', f"{batch_summary.get('success_rate', 0.0):.2%}"])
        
        if 'average_scores' in batch_summary:
            avg_scores = batch_summary['average_scores']
            writer.writerow(['Average Overall Score', round(avg_scores.get('overall_score', 0.0), precision)])
    
    def _write_single_csv(self, writer, results: Dict[str, Any], 
                         include_metadata: bool, precision: int) -> None:
        """Write single result to CSV."""
        # Write basic info
        writer.writerow(['Field', 'Value'])
        writer.writerow(['Image ID', results.get('image_id', '')])
        writer.writerow(['Overall Score', round(results.get('overall_score', 0.0), precision)])
        writer.writerow(['Quality Level', results.get('quality_level', '')])
        
        # Write metrics
        writer.writerow([])  # Empty row
        writer.writerow(['METRICS'])
        writer.writerow(['Metric', 'Score', 'Details'])
        
        for metric_name, metric_data in results.get('metrics', {}).items():
            if isinstance(metric_data, dict) and 'error' not in metric_data:
                primary_score = self._extract_primary_score_for_export(metric_name, metric_data)
                score_str = f"{primary_score:.{precision}f}" if primary_score is not None else "N/A"
                details = str(metric_data)[:100] + "..." if len(str(metric_data)) > 100 else str(metric_data)
                writer.writerow([metric_name, score_str, details])
            else:
                writer.writerow([metric_name, "ERROR", str(metric_data)])
    
    def _write_batch_txt(self, f, results: Dict[str, Any], 
                        include_metadata: bool, precision: int) -> None:
        """Write batch results to text file."""
        # Batch summary
        batch_summary = results.get('batch_summary', {})
        f.write("BATCH SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Batch ID: {results.get('batch_id', 'N/A')}\n")
        f.write(f"Total Evaluated: {batch_summary.get('total_evaluated', 0)}\n")
        f.write(f"Successful: {batch_summary.get('successful', 0)}\n")
        f.write(f"Failed: {batch_summary.get('failed', 0)}\n")
        f.write(f"Success Rate: {batch_summary.get('success_rate', 0.0):.2%}\n")
        f.write(f"Processing Time: {batch_summary.get('processing_time', 0.0):.2f}s\n")
        
        if 'average_scores' in batch_summary:
            avg_scores = batch_summary['average_scores']
            f.write(f"Average Overall Score: {avg_scores.get('overall_score', 0.0):.{precision}f}\n")
        
        f.write("\n")
        
        # Aggregate metrics
        if 'aggregate_metrics' in results:
            agg_metrics = results['aggregate_metrics']
            f.write("AGGREGATE METRICS\n")
            f.write("-" * 20 + "\n")
            
            batch_stats = agg_metrics.get('batch_statistics', {})
            if batch_stats:
                f.write(f"Mean Score: {batch_stats.get('mean', 0.0):.{precision}f}\n")
                f.write(f"Median Score: {batch_stats.get('median', 0.0):.{precision}f}\n")
                f.write(f"Std Deviation: {batch_stats.get('std_dev', 0.0):.{precision}f}\n")
                f.write(f"Min Score: {batch_stats.get('min_score', 0.0):.{precision}f}\n")
                f.write(f"Max Score: {batch_stats.get('max_score', 0.0):.{precision}f}\n")
        
        f.write("\n")
        
        # Individual results summary
        f.write("INDIVIDUAL RESULTS\n")
        f.write("-" * 20 + "\n")
        
        for i, result in enumerate(results.get('individual_results', [])[:10]):  # Show first 10
            f.write(f"{i+1}. {result.get('image_id', 'N/A')}: ")
            f.write(f"Score={result.get('overall_score', 0.0):.{precision}f}, ")
            f.write(f"Quality={result.get('quality_level', 'N/A')}\n")
        
        if len(results.get('individual_results', [])) > 10:
            f.write(f"... and {len(results['individual_results']) - 10} more results\n")
    
    def _write_single_txt(self, f, results: Dict[str, Any], 
                         include_metadata: bool, precision: int) -> None:
        """Write single result to text file."""
        # Basic information
        f.write("EVALUATION RESULT\n")
        f.write("-" * 20 + "\n")
        f.write(f"Image ID: {results.get('image_id', 'N/A')}\n")
        f.write(f"Overall Score: {results.get('overall_score', 0.0):.{precision}f}\n")
        f.write(f"Quality Level: {results.get('quality_level', 'N/A')}\n")
        
        if include_metadata:
            f.write(f"Evaluation Time: {results.get('evaluation_time', 0.0):.3f}s\n")
            f.write(f"Timestamp: {results.get('timestamp', 'N/A')}\n")
        
        f.write("\n")
        
        # Metrics
        f.write("METRIC SCORES\n")
        f.write("-" * 20 + "\n")
        
        for metric_name, metric_data in results.get('metrics', {}).items():
            f.write(f"{metric_name.upper()}:\n")
            if isinstance(metric_data, dict) and 'error' not in metric_data:
                for key, value in metric_data.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.{precision}f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  ERROR: {metric_data}\n")
            f.write("\n")
    
    def _write_batch_html(self, f, results: Dict[str, Any], 
                         include_metadata: bool, precision: int) -> None:
        """Write batch results to HTML file."""
        f.write("<h1>Medical Report Evaluation - Batch Results</h1>\n")
        
        # Batch summary
        batch_summary = results.get('batch_summary', {})
        f.write("<h2>Batch Summary</h2>\n")
        f.write("<table class='summary-table'>\n")
        f.write(f"<tr><td>Batch ID</td><td>{html.escape(str(results.get('batch_id', 'N/A')))}</td></tr>\n")
        f.write(f"<tr><td>Total Evaluated</td><td>{batch_summary.get('total_evaluated', 0)}</td></tr>\n")
        f.write(f"<tr><td>Successful</td><td>{batch_summary.get('successful', 0)}</td></tr>\n")
        f.write(f"<tr><td>Failed</td><td>{batch_summary.get('failed', 0)}</td></tr>\n")
        f.write(f"<tr><td>Success Rate</td><td>{batch_summary.get('success_rate', 0.0):.2%}</td></tr>\n")
        f.write(f"<tr><td>Processing Time</td><td>{batch_summary.get('processing_time', 0.0):.2f}s</td></tr>\n")
        
        if 'average_scores' in batch_summary:
            avg_scores = batch_summary['average_scores']
            f.write(f"<tr><td>Average Overall Score</td><td>{avg_scores.get('overall_score', 0.0):.{precision}f}</td></tr>\n")
        
        f.write("</table>\n")
        
        # Individual results table
        f.write("<h2>Individual Results</h2>\n")
        f.write("<table class='results-table'>\n")
        f.write("<tr><th>Image ID</th><th>Overall Score</th><th>Quality Level</th><th>Evaluation Time</th></tr>\n")
        
        for result in results.get('individual_results', []):
            f.write("<tr>")
            f.write(f"<td>{html.escape(str(result.get('image_id', 'N/A')))}</td>")
            f.write(f"<td>{result.get('overall_score', 0.0):.{precision}f}</td>")
            f.write(f"<td>{html.escape(str(result.get('quality_level', 'N/A')))}</td>")
            f.write(f"<td>{result.get('evaluation_time', 0.0):.3f}s</td>")
            f.write("</tr>\n")
        
        f.write("</table>\n")
    
    def _write_single_html(self, f, results: Dict[str, Any], 
                          include_metadata: bool, precision: int) -> None:
        """Write single result to HTML file."""
        f.write("<h1>Medical Report Evaluation - Single Result</h1>\n")
        
        # Basic information
        f.write("<h2>Evaluation Summary</h2>\n")
        f.write("<table class='summary-table'>\n")
        f.write(f"<tr><td>Image ID</td><td>{html.escape(str(results.get('image_id', 'N/A')))}</td></tr>\n")
        f.write(f"<tr><td>Overall Score</td><td>{results.get('overall_score', 0.0):.{precision}f}</td></tr>\n")
        f.write(f"<tr><td>Quality Level</td><td>{html.escape(str(results.get('quality_level', 'N/A')))}</td></tr>\n")
        
        if include_metadata:
            f.write(f"<tr><td>Evaluation Time</td><td>{results.get('evaluation_time', 0.0):.3f}s</td></tr>\n")
            f.write(f"<tr><td>Timestamp</td><td>{html.escape(str(results.get('timestamp', 'N/A')))}</td></tr>\n")
        
        f.write("</table>\n")
        
        # Metrics
        f.write("<h2>Metric Scores</h2>\n")
        f.write("<table class='metrics-table'>\n")
        f.write("<tr><th>Metric</th><th>Primary Score</th><th>Details</th></tr>\n")
        
        for metric_name, metric_data in results.get('metrics', {}).items():
            f.write("<tr>")
            f.write(f"<td>{html.escape(metric_name.upper())}</td>")
            
            if isinstance(metric_data, dict) and 'error' not in metric_data:
                primary_score = self._extract_primary_score_for_export(metric_name, metric_data)
                score_str = f"{primary_score:.{precision}f}" if primary_score is not None else "N/A"
                f.write(f"<td>{score_str}</td>")
                
                # Format details
                details_html = "<ul>"
                for key, value in metric_data.items():
                    if isinstance(value, (int, float)):
                        details_html += f"<li>{html.escape(key)}: {value:.{precision}f}</li>"
                    else:
                        details_html += f"<li>{html.escape(key)}: {html.escape(str(value))}</li>"
                details_html += "</ul>"
                f.write(f"<td>{details_html}</td>")
            else:
                f.write(f"<td>ERROR</td>")
                f.write(f"<td>{html.escape(str(metric_data))}</td>")
            
            f.write("</tr>\n")
        
        f.write("</table>\n")
    
    def _get_html_header(self) -> str:
        """Get HTML document header with CSS styling."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Report Evaluation Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        h1, h2 { color: #2c3e50; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .summary-table { max-width: 600px; }
        .results-table, .metrics-table { width: 100%; }
        ul { margin: 0; padding-left: 20px; }
    </style>
</head>
<body>
"""
    
    def _get_html_footer(self) -> str:
        """Get HTML document footer."""
        return f"""
<hr>
<p><small>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Medical Report Evaluation System</small></p>
</body>
</html>
"""
    
    def _format_numeric_values(self, data: Any, precision: int) -> Any:
        """Recursively format numeric values with specified precision."""
        if isinstance(data, dict):
            return {k: self._format_numeric_values(v, precision) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._format_numeric_values(item, precision) for item in data]
        elif isinstance(data, float):
            return round(data, precision)
        else:
            return data
    
    def _remove_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove metadata fields from results."""
        metadata_fields = ['timestamp', 'session_id', 'evaluator_version', 'metadata']
        filtered_data = {k: v for k, v in data.items() if k not in metadata_fields}
        
        # Also remove metadata from individual results if present
        if 'individual_results' in filtered_data:
            filtered_individual = []
            for result in filtered_data['individual_results']:
                filtered_result = {k: v for k, v in result.items() if k not in metadata_fields}
                filtered_individual.append(filtered_result)
            filtered_data['individual_results'] = filtered_individual
        
        return filtered_data
    
    def _extract_primary_score_for_export(self, metric_name: str, metric_data: Dict) -> Optional[float]:
        """Extract primary score from metric data for export."""
        if "error" in metric_data:
            return None
        
        # Define primary score extraction logic for each metric
        if metric_name == "bleu":
            return metric_data.get("bleu_4", metric_data.get("bleu", None))
        elif metric_name == "rouge":
            return metric_data.get("rouge_l", metric_data.get("rouge_1", None))
        elif metric_name == "meteor":
            return metric_data.get("meteor", metric_data.get("score", None))
        elif metric_name == "bert_score":
            return metric_data.get("f1", metric_data.get("bert_score", None))
        elif metric_name == "cider":
            return metric_data.get("cider", metric_data.get("score", None))
        elif metric_name == "medical":
            return metric_data.get("overall_score", metric_data.get("medical_score", None))
        else:
            # Generic fallback
            for field in ["score", "f1", "overall", "primary"]:
                if field in metric_data:
                    return metric_data[field]
        
        return None
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats."""
        return self.supported_formats.copy()


def test_results_exporter():
    """Test the ResultsExporter functionality."""
    print("Testing ResultsExporter...")
    
    # Create exporter
    exporter = ResultsExporter()
    
    # Test data
    test_result = {
        "image_id": "test_001",
        "overall_score": 0.7234,
        "quality_level": "good",
        "metrics": {
            "bleu": {"bleu": 0.45, "bleu_1": 0.6, "bleu_4": 0.3},
            "rouge": {"rouge_1": 0.62, "rouge_l": 0.58},
            "meteor": {"meteor": 0.51}
        },
        "timestamp": "2025-06-13T10:30:00Z"
    }
    
    # Test JSON export
    json_path = "test_result.json"
    exporter.export_results(test_result, json_path, format='json')
    print(f"JSON exported to {json_path}")
    
    # Test CSV export
    csv_path = "test_result.csv"
    exporter.export_results(test_result, csv_path, format='csv')
    print(f"CSV exported to {csv_path}")
    
    # Test TXT export
    txt_path = "test_result.txt"
    exporter.export_results(test_result, txt_path, format='txt')
    print(f"TXT exported to {txt_path}")
    
    # Test HTML export
    html_path = "test_result.html"
    exporter.export_results(test_result, html_path, format='html')
    print(f"HTML exported to {html_path}")
    
    print("ResultsExporter test completed!")
    return True


if __name__ == "__main__":
    test_results_exporter() 