#!/usr/bin/env python3
"""
Fixed Random X-ray Analysis Script
==================================

This script randomly selects a chest X-ray image from the MIMIC dataset
and generates a comprehensive medical report using the AI pipeline.
FIXED: Excludes image data from LLM request to avoid oversized prompts.

Author: AI Medical Report Generator
Date: 2025-01-19
"""

import os
import random
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from new_scripts.medical_report_generator import MedicalReportGenerator


def format_medical_report(analysis_result):
    """
    Format the medical report for readable console output.
    
    Args:
        analysis_result (dict): The comprehensive analysis result from the generator
        
    Returns:
        str: Formatted medical report text
    """
    if not analysis_result or 'medical_report' not in analysis_result:
        return "❌ No medical report generated"
    
    report_data = analysis_result['medical_report']
    
    # Clinical Findings - Just the core medical report
    formatted_report = "🔍 CLINICAL FINDINGS:\n"
    formatted_report += "-" * 50 + "\n"
    
    # Extract the actual report text
    report_text = report_data.get('report_text', 'No report text available')
    
    # Format the report text with proper indentation
    lines = report_text.split('\n')
    for line in lines:
        if line.strip():
            formatted_report += f"   {line.strip()}\n"
    
    return formatted_report


def get_random_xray_image():
    """
    Randomly select a chest X-ray image from the MIMIC dataset.
    
    Returns:
        str: Path to the randomly selected image
    """
    # Path to the image directory
    img_dir = project_root / "data_dump" / "output" / "img_png"
    
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    # Get all PNG files
    png_files = list(img_dir.glob("*.png"))
    
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {img_dir}")
    
    # Randomly select one
    selected_image = random.choice(png_files)
    
    print(f"🎲 Randomly selected image: {selected_image.name}")
    print(f"📁 Full path: {selected_image}")
    print(f"📊 Total available images: {len(png_files)}")
    
    return str(selected_image)


def main():
    """Main execution function."""
    
    print("🚀 Starting Fixed Random X-ray Analysis...")
    print("=" * 80)
    
    try:
        # Step 1: Select random image
        print("\n🎯 Step 1: Selecting random X-ray image...")
        image_path = get_random_xray_image()
        
        # Step 2: Initialize the generator
        print("\n🔧 Step 2: Initializing Medical Report Generator...")
        print("   Loading models and dependencies...")
        generator = MedicalReportGenerator()
        print("   ✅ Generator initialized successfully!")
        
        # Step 3: Generate comprehensive analysis
        print("\n🔬 Step 3: Analyzing chest X-ray...")
        print("   • Running multimodal classification...")
        print("   • Extracting clinical keywords...")
        print("   • Generating attention analysis...")
        print("   • Creating medical report with LLM...")
        print("   🔧 FIX: Excluding image from LLM request to avoid oversized prompts")
        
        # FIXED: Set include_image_in_llm=False to avoid oversized prompts
        result = generator.generate_comprehensive_analysis(
            image_path=image_path,
            template="standard",
            include_image_in_llm=False  # 🔧 This is the key fix!
        )
        
        # Step 4: Format and display results
        print("\n✅ Step 4: Analysis complete! Formatting results...")
        print("\n")
        
        formatted_report = format_medical_report(result)
        print(formatted_report)
        
        # Optional: Save to file
        output_file = project_root / "new_scripts" / "latest_fixed_analysis.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_report)
        
        print(f"💾 Report also saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("\nPlease ensure:")
        print("1. Virtual environment is activated")
        print("2. LLM server is running at http://127.0.0.1:1234")
        print("3. All model files are available")
        print("4. CUDA GPU is available (recommended)")
        
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 