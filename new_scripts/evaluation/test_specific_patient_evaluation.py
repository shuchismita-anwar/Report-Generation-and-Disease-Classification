#!/usr/bin/env python3
"""
Specific Patient Report Evaluation
==================================

This script evaluates the LLM-generated report for patient a5bf1678-df56650d-3d94165c-71ef2d00-509ad013
against the ground truth radiology report using all available metrics (BLEU, ROUGE, METEOR, CIDEr, BERTScore, Medical).

Patient Details:
- Patient ID: patient_13620301
- Study ID: 51583714
- DICOM ID: a5bf1678-df56650d-3d94165c-71ef2d00-509ad013

Author: Medical Report Evaluation System
Date: 2025-01-19
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add the evaluation system to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from medical_report_evaluator import MedicalReportEvaluator


def load_llm_generated_report() -> str:
    """
    Load the LLM-generated report from latest_fixed_analysis.txt
    
    Returns:
        str: The cleaned LLM-generated report text
    """
    llm_report_path = current_dir.parent / "latest_fixed_analysis.txt"
    
    if not llm_report_path.exists():
        raise FileNotFoundError(f"LLM report not found: {llm_report_path}")
    
    with open(llm_report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # The file appears to contain a complete medical report already
    # Clean it up by removing any extra headers and formatting
    lines = content.split('\n')
    report_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('🔍') and not line.startswith('--'):
            report_lines.append(line)
    
    # If no content found, return the entire content stripped
    result = '\n'.join(report_lines).strip()
    if not result:
        result = content.strip()
    
    return result


def load_ground_truth_report() -> str:
    """
    Load the ground truth radiology report from the MIMIC dataset
    
    Returns:
        str: The cleaned ground truth report text
    """
    # Path to the ground truth report
    gt_report_path = current_dir.parent.parent / "mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0" / "mimic-eye" / "patient_13620301" / "CXR-DICOM" / "s51583714.txt"
    
    if not gt_report_path.exists():
        raise FileNotFoundError(f"Ground truth report not found: {gt_report_path}")
    
    with open(gt_report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Clean the ground truth report by removing headers and extra whitespace
    lines = content.split('\n')
    report_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('FINAL REPORT'):
            report_lines.append(line)
    
    return '\n'.join(report_lines).strip()


def format_detailed_results(evaluation_result: dict) -> str:
    """
    Format the evaluation results into a detailed, readable report
    
    Args:
        evaluation_result (dict): Evaluation results from the evaluator
        
    Returns:
        str: Formatted detailed report
    """
    results = evaluation_result
    
    # Extract key metrics
    overall_score = results.get('overall_score', 0)
    # quality_level = results.get('quality_level', 'unknown')
    metrics = results.get('metrics', {})
    
    # Build detailed report
    report = f"""
# 🏥 DETAILED PATIENT EVALUATION REPORT
**Patient ID**: patient_13620301 (a5bf1678-df56650d-3d94165c-71ef2d00-509ad013)
**Study ID**: 51583714
**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 OVERALL PERFORMANCE
- **Overall Score**: {overall_score:.4f}
- **Processing Time**: {results.get('evaluation_time', 0):.3f} seconds

## 📊 DETAILED METRIC BREAKDOWN

"""
    
    # Add individual metric details
    for metric_name, metric_data in metrics.items():
        report += f"### {metric_name.upper()} Metric\n"
        
        if isinstance(metric_data, dict):
            # Extract primary score based on metric type
            if metric_name == "bleu":
                primary_score = metric_data.get('bleu', 0)
                report += f"- **Primary Score**: {primary_score:.4f}\n"
                if 'bleu_1' in metric_data:
                    report += f"- **BLEU-1**: {metric_data['bleu_1']:.4f}\n"
                if 'bleu_2' in metric_data:
                    report += f"- **BLEU-2**: {metric_data['bleu_2']:.4f}\n"
                if 'bleu_3' in metric_data:
                    report += f"- **BLEU-3**: {metric_data['bleu_3']:.4f}\n"
                if 'bleu_4' in metric_data:
                    report += f"- **BLEU-4**: {metric_data['bleu_4']:.4f}\n"
                    
            elif metric_name == "rouge":
                rouge_f1 = metric_data.get('rouge_avg_f1', 0)
                report += f"- **Average F1**: {rouge_f1:.4f}\n"
                if 'rouge1_f1' in metric_data:
                    report += f"- **ROUGE-1 F1**: {metric_data['rouge1_f1']:.4f}\n"
                if 'rouge2_f1' in metric_data:
                    report += f"- **ROUGE-2 F1**: {metric_data['rouge2_f1']:.4f}\n"
                if 'rougeL_f1' in metric_data:
                    report += f"- **ROUGE-L F1**: {metric_data['rougeL_f1']:.4f}\n"
                    
            elif metric_name == "meteor":
                meteor_score = metric_data.get('meteor', 0)
                report += f"- **METEOR Score**: {meteor_score:.4f}\n"
                
            elif metric_name == "cider":
                cider_score = metric_data.get('cider', 0)
                report += f"- **CIDEr Score**: {cider_score:.4f}\n"
                
            elif metric_name == "bert_score":
                if 'f1' in metric_data:
                    report += f"- **F1 Score**: {metric_data['f1']:.4f}\n"
                if 'precision' in metric_data:
                    report += f"- **Precision**: {metric_data['precision']:.4f}\n"
                if 'recall' in metric_data:
                    report += f"- **Recall**: {metric_data['recall']:.4f}\n"
                    
            elif metric_name == "medical":
                medical_score = metric_data.get('medical_score', 0)
                report += f"- **Medical Score**: {medical_score:.4f}\n"
                if 'anatomy_score' in metric_data:
                    report += f"- **Anatomy Score**: {metric_data['anatomy_score']:.4f}\n"
                if 'pathology_score' in metric_data:
                    report += f"- **Pathology Score**: {metric_data['pathology_score']:.4f}\n"
                if 'terminology_score' in metric_data:
                    report += f"- **Terminology Score**: {metric_data['terminology_score']:.4f}\n"
        else:
            report += f"- **Score**: {metric_data:.4f}\n"
        
        report += "\n"
    
    # Add interpretation
    report += f"""
## 🔍 INTERPRETATION & INSIGHTS

### Overall Assessment
"""
    
    if overall_score >= 0.8:
        report += "- **Excellent Performance**: The LLM-generated report shows high similarity to the ground truth.\n"
    elif overall_score >= 0.6:
        report += "- **Good Performance**: The LLM-generated report captures most key elements from the ground truth.\n"
    elif overall_score >= 0.4:
        report += "- **Moderate Performance**: The LLM-generated report has some alignment but needs improvement.\n"
    else:
        report += "- **Poor Performance**: The LLM-generated report shows limited similarity to the ground truth.\n"
    
    # Metric-specific insights
    report += "\n### Metric-Specific Insights\n"
    
    if 'bleu' in metrics:
        bleu_score = metrics['bleu'].get('bleu', 0) if isinstance(metrics['bleu'], dict) else metrics['bleu']
        if bleu_score < 0.1:
            report += "- **BLEU**: Very low word overlap suggests different vocabulary or phrasing.\n"
        elif bleu_score < 0.3:
            report += "- **BLEU**: Moderate word overlap indicates some shared terminology.\n"
        else:
            report += "- **BLEU**: Good word overlap shows similar vocabulary usage.\n"
    
    if 'rouge' in metrics:
        rouge_score = metrics['rouge'].get('rouge_avg_f1', 0) if isinstance(metrics['rouge'], dict) else metrics['rouge']
        if rouge_score < 0.2:
            report += "- **ROUGE**: Low recall suggests missing key information from ground truth.\n"
        elif rouge_score < 0.5:
            report += "- **ROUGE**: Moderate recall indicates partial coverage of ground truth content.\n"
        else:
            report += "- **ROUGE**: Good recall shows comprehensive coverage of ground truth.\n"
    
    if 'medical' in metrics:
        medical_score = metrics['medical'].get('medical_score', 0) if isinstance(metrics['medical'], dict) else metrics['medical']
        if medical_score >= 0.7:
            report += "- **Medical**: Strong medical concept alignment indicates clinically relevant content.\n"
        elif medical_score >= 0.5:
            report += "- **Medical**: Moderate medical concept alignment shows some clinical accuracy.\n"
        else:
            report += "- **Medical**: Low medical concept alignment suggests need for better clinical terminology.\n"
    
    return report



def main():
    """Main execution function."""
    
    print("🚀 Starting Specific Patient Evaluation...")
    print("=" * 80)
    print(f"📋 Patient: a5bf1678-df56650d-3d94165c-71ef2d00-509ad013")
    print(f"🏥 Study: 51583714 (patient_13620301)")
    print("=" * 80)
    
    try:
        # Step 1: Load both reports
        print("\n📖 Step 1: Loading reports...")
        
        print("   Loading LLM-generated report...")
        llm_report = load_llm_generated_report()
        print(f"   ✅ LLM report loaded ({len(llm_report)} characters)")
        
        print("   Loading ground truth report...")
        gt_report = load_ground_truth_report()
        print(f"   ✅ Ground truth loaded ({len(gt_report)} characters)")
        
        # Step 2: Initialize evaluator
        print("\n🔧 Step 2: Initializing evaluation system...")
        evaluator = MedicalReportEvaluator()
        print("   ✅ Evaluator initialized with all metrics")
        
        # Step 3: Run evaluation
        print("\n⚡ Step 3: Running comprehensive evaluation...")
        print("   • Calculating BLEU scores...")
        print("   • Calculating ROUGE scores...")
        print("   • Calculating METEOR scores...")
        print("   • Calculating CIDEr scores...")
        print("   • Calculating BERTScore...")
        print("   • Calculating Medical concept scores...")
        
        results = evaluator.evaluate_single(
            generated_report=llm_report,
            ground_truth_report=gt_report,
            image_id="a5bf1678-df56650d-3d94165c-71ef2d00-509ad013",
            metadata={
                "patient_id": "patient_13620301",
                "study_id": "51583714",
                "evaluation_type": "specific_patient_test"
            }
        )
        
        # Step 4: Generate detailed report
        print("\n📊 Step 4: Generating detailed results...")
        
        detailed_report = format_detailed_results(results)
        print(detailed_report)
        
        # Results displayed above - no saving required
        
        # Summary output
        print(f"\n🎯 QUICK SUMMARY:")
        # print(f"   Overall Score: {results.get('overall_score', 0):.4f}")
        # print(f"   Quality Level: {results.get('quality_level', 'unknown').upper()}")
        
        # Extract key metric scores for quick view
        metrics = results.get('metrics', {})
        if 'bleu' in metrics:
            bleu_score = metrics['bleu'].get('bleu', 0) if isinstance(metrics['bleu'], dict) else metrics['bleu']
            print(f"   BLEU Score: {bleu_score:.4f}")
        
        if 'rouge' in metrics:
            rouge_score = metrics['rouge'].get('rouge_avg_f1', 0) if isinstance(metrics['rouge'], dict) else metrics['rouge']
            print(f"   ROUGE F1: {rouge_score:.4f}")
        
        if 'meteor' in metrics:
            meteor_score = metrics['meteor'].get('meteor', 0) if isinstance(metrics['meteor'], dict) else metrics['meteor']
            print(f"   METEOR Score: {meteor_score:.4f}")
        
        if 'medical' in metrics:
            medical_score = metrics['medical'].get('medical_score', 0) if isinstance(metrics['medical'], dict) else metrics['medical']
            print(f"   Medical Score: {medical_score:.4f}")
        
        print(f"\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        print("\nPlease ensure:")
        print("1. Virtual environment is activated")
        print("2. Both report files exist and are readable")
        print("3. All required dependencies are installed")
        
        import traceback
        traceback.print_exc()
        
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 