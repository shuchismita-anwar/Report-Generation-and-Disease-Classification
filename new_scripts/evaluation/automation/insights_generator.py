#!/usr/bin/env python3
"""
Automated Insights Generator
===========================
AI-powered insights and recommendations for MIMIC-Eye evaluation results.

Author: Medical Report Evaluation System
Version: 1.0
"""

import json
from typing import Dict, List
from datetime import datetime


class AutomatedInsightsGenerator:
    """Generate automated insights and recommendations."""
    
    def __init__(self):
        pass
    
    def generate_insights(self, results: Dict) -> Dict:
        """Generate comprehensive insights from evaluation results."""
        insights = {
            "performance_analysis": self._analyze_performance(results),
            "metric_analysis": self._analyze_metrics(results),
            "recommendations": self._generate_recommendations(results),
            "quality_patterns": self._detect_quality_patterns(results),
            "generated_at": datetime.now().isoformat()
        }
        
        return insights
    
    def _analyze_performance(self, results: Dict) -> Dict:
        """Analyze overall performance metrics."""
        batch_summary = results.get('batch_summary', {})
        overall_score = results.get('aggregate_metrics', {}).get('batch_overall_score', 0)
        
        performance_level = "excellent" if overall_score > 0.8 else \
                           "good" if overall_score > 0.6 else \
                           "moderate" if overall_score > 0.4 else \
                           "needs_improvement"
        
        return {
            "overall_score": overall_score,
            "performance_level": performance_level,
            "success_rate": batch_summary.get('success_rate', 0),
            "processing_efficiency": batch_summary.get('processing_time', 0),
            "key_strength": "High reliability" if batch_summary.get('success_rate', 0) > 0.95 else "Room for improvement"
        }
    
    def _analyze_metrics(self, results: Dict) -> Dict:
        """Analyze individual metric performance."""
        per_metric = results.get('aggregate_metrics', {}).get('batch_statistics', {}).get('per_metric', {})
        
        if not per_metric:
            return {"status": "no_metric_data"}
        
        # Find strongest and weakest metrics with std dev
        metric_scores = {name: data.get('mean', 0) for name, data in per_metric.items()}
        metric_stds = {name: data.get('std', 0) for name, data in per_metric.items()}
        strongest = max(metric_scores, key=metric_scores.get) if metric_scores else None
        weakest = min(metric_scores, key=metric_scores.get) if metric_scores else None
        
        return {
            "strongest_metric": strongest,
            "strongest_score": metric_scores.get(strongest, 0) if strongest else 0,
            "strongest_std": metric_stds.get(strongest, 0) if strongest else 0,
            "weakest_metric": weakest,
            "weakest_score": metric_scores.get(weakest, 0) if weakest else 0,
            "weakest_std": metric_stds.get(weakest, 0) if weakest else 0,
            "metric_count": len(metric_scores),
            "average_score": sum(metric_scores.values()) / len(metric_scores) if metric_scores else 0
        }
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        overall_score = results.get('aggregate_metrics', {}).get('batch_overall_score', 0)
        success_rate = results.get('batch_summary', {}).get('success_rate', 0)
        
        # Performance-based recommendations
        if overall_score < 0.3:
            recommendations.append("Consider reviewing LLM training data alignment with radiologist writing style")
            recommendations.append("Evaluate if the model is generating appropriate medical terminology")
        elif overall_score < 0.6:
            recommendations.append("Fine-tune model parameters to improve semantic similarity")
            recommendations.append("Consider additional training on medical report datasets")
        else:
            recommendations.append("Performance is strong - consider optimizing for specific metrics")
        
        # Reliability recommendations
        if success_rate < 0.9:
            recommendations.append("Investigate and resolve processing failures to improve reliability")
        
        # Metric-specific recommendations
        per_metric = results.get('aggregate_metrics', {}).get('batch_statistics', {}).get('per_metric', {})
        if per_metric:
            for metric, data in per_metric.items():
                score = data.get('mean', 0)
                if score < 0.3:
                    recommendations.append(f"Focus on improving {metric} scores - consider metric-specific training")
        
        return recommendations
    
    def _detect_quality_patterns(self, results: Dict) -> Dict:
        """Detect patterns in evaluation quality."""
        patterns = {
            "consistency": "high",  # Default assumption
            "trend": "stable",
            "outliers": False
        }
        
        # Analyze consistency
        per_metric = results.get('aggregate_metrics', {}).get('batch_statistics', {}).get('per_metric', {})
        if per_metric:
            std_devs = [data.get('std_dev', 0) for data in per_metric.values()]
            avg_std = sum(std_devs) / len(std_devs) if std_devs else 0
            
            if avg_std > 0.2:
                patterns["consistency"] = "low"
            elif avg_std > 0.1:
                patterns["consistency"] = "moderate"
        
        return patterns


def generate_insights(results: Dict) -> Dict:
    """Generate comprehensive insights for automation results."""
    generator = AutomatedInsightsGenerator()
    return generator.generate_insights(results)