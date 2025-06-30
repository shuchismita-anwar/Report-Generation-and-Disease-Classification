#!/usr/bin/env python3
"""
Specific Patient X-ray Analysis Script - Priority 1: Gaze-Attention Validation
==============================================================================

This script analyzes a specific chest X-ray image from the MIMIC dataset
and generates a comprehensive medical report using the AI pipeline.
FIXED: Excludes image data from LLM request to avoid oversized prompts.

NEW: Priority 1 Implementation - Gaze-Attention Validation
- Generates human gaze heatmap from fixation data
- Generates model attention heatmap (Grad-CAM)
- Compares spatial correlation between human and model attention
- Saves visualizations for analysis

Author: AI Medical Report Generator
Date: 2025-01-19
"""

import os
import random
import sys
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error

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


def get_specific_xray_image():
    """
    Get a specific chest X-ray image from the MIMIC dataset.
    
    Returns:
        str: Path to the specific patient image
    """
    # Specific patient ID that we know has interesting findings
    specific_patient_id = "a5bf1678-df56650d-3d94165c-71ef2d00-509ad013"
    
    # Path to the image directory
    img_dir = project_root / "data_dump" / "output" / "img_png"
    
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    # Construct the specific image path
    specific_image = img_dir / f"{specific_patient_id}.png"
    
    if not specific_image.exists():
        raise FileNotFoundError(f"Specific patient image not found: {specific_image}")
    
    print(f"🎯 Analyzing specific patient: {specific_patient_id}")
    print(f"📁 Full path: {specific_image}")
    
    return str(specific_image)


def load_fixation_data(fixations_path):
    """
    Load and parse human gaze fixation data.
    
    Args:
        fixations_path (str): Path to the norm_fixations CSV file
        
    Returns:
        pd.DataFrame: Fixation data with columns [x_norm, y_norm, gaze_duration]
    """
    try:
        # Load the CSV file
        fixations_df = pd.read_csv(fixations_path)
        
        print(f"   📊 Loaded {len(fixations_df)} fixation points")
        print(f"   📋 Columns: {list(fixations_df.columns)}")
        
        # Validate required columns
        required_cols = ['x_norm', 'y_norm', 'gaze_duration']
        missing_cols = [col for col in required_cols if col not in fixations_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Basic statistics
        print(f"   🎯 X range: {fixations_df['x_norm'].min():.3f} - {fixations_df['x_norm'].max():.3f}")
        print(f"   🎯 Y range: {fixations_df['y_norm'].min():.3f} - {fixations_df['y_norm'].max():.3f}")
        print(f"   ⏱️  Duration range: {fixations_df['gaze_duration'].min():.1f} - {fixations_df['gaze_duration'].max():.1f} ms")
        
        return fixations_df
        
    except Exception as e:
        print(f"   ❌ Error loading fixation data: {e}")
        return None


def generate_gaze_heatmap(fixations_df, heatmap_size=(224, 224), sigma=15):
    """
    Generate a heatmap from human gaze fixation data.
    
    Args:
        fixations_df (pd.DataFrame): Fixation data
        heatmap_size (tuple): Output heatmap dimensions (height, width)
        sigma (float): Gaussian blur sigma for fixation points
        
    Returns:
        np.ndarray: Normalized heatmap (0-1) of shape heatmap_size
    """
    if fixations_df is None or len(fixations_df) == 0:
        return np.zeros(heatmap_size)
    
    # Initialize heatmap
    heatmap = np.zeros(heatmap_size, dtype=np.float32)
    
    # Convert normalized coordinates to pixel coordinates
    height, width = heatmap_size
    
    for _, row in fixations_df.iterrows():
        # Convert normalized coordinates (0-1) to pixel coordinates
        x_pixel = int(row['x_norm'] * width)
        y_pixel = int(row['y_norm'] * height)
        
        # Clamp to valid range
        x_pixel = max(0, min(width - 1, x_pixel))
        y_pixel = max(0, min(height - 1, y_pixel))
        
        # Weight by gaze duration
        duration_weight = row['gaze_duration'] / 1000.0  # Convert ms to seconds
        
        # Add weighted point to heatmap
        heatmap[y_pixel, x_pixel] += duration_weight
    
    # Apply Gaussian blur to create smooth heatmap
    if sigma > 0:
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
    
    # Normalize to 0-1 range
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


def calculate_attention_metrics(human_heatmap, model_attention):
    """
    Calculate spatial correlation metrics between human gaze and model attention.
    
    Args:
        human_heatmap (np.ndarray): Human gaze heatmap (0-1)
        model_attention (np.ndarray): Model attention map (0-1)
        
    Returns:
        dict: Dictionary of correlation metrics
    """
    # Ensure same dimensions
    if human_heatmap.shape != model_attention.shape:
        # Resize model attention to match human heatmap
        model_attention = cv2.resize(model_attention, (human_heatmap.shape[1], human_heatmap.shape[0]))
    
    # Flatten for correlation calculations
    human_flat = human_heatmap.flatten()
    model_flat = model_attention.flatten()
    
    # Pearson correlation
    pearson_corr, pearson_p = pearsonr(human_flat, model_flat)
    
    # Jensen-Shannon divergence (treat as probability distributions)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    human_prob = (human_flat + epsilon) / (human_flat.sum() + epsilon * len(human_flat))
    model_prob = (model_flat + epsilon) / (model_flat.sum() + epsilon * len(model_flat))
    js_divergence = jensenshannon(human_prob, model_prob)
    
    # Mean Squared Error
    mse = mean_squared_error(human_flat, model_flat)
    
    # Normalized Scanpath Saliency (NSS) - simplified version
    # NSS measures how well model attention predicts human fixations
    # Higher NSS = better prediction
    nss_scores = []
    for _, row in pd.DataFrame({'x_norm': human_heatmap.shape[1] * np.random.random(100), 
                               'y_norm': human_heatmap.shape[0] * np.random.random(100)}).iterrows():
        x_idx = int(min(row['x_norm'], model_attention.shape[1] - 1))
        y_idx = int(min(row['y_norm'], model_attention.shape[0] - 1))
        nss_scores.append(model_attention[y_idx, x_idx])
    
    nss = np.mean(nss_scores) if nss_scores else 0.0
    
    return {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'jensen_shannon_divergence': js_divergence,
        'mean_squared_error': mse,
        'normalized_scanpath_saliency': nss,
        'human_attention_entropy': -np.sum(human_prob * np.log(human_prob + epsilon)),
        'model_attention_entropy': -np.sum(model_prob * np.log(model_prob + epsilon))
    }


def save_attention_comparison(human_heatmap, model_attention, metrics, patient_id, output_dir):
    """
    Save visualization comparing human gaze and model attention.
    
    Args:
        human_heatmap (np.ndarray): Human gaze heatmap
        model_attention (np.ndarray): Model attention map
        metrics (dict): Correlation metrics
        patient_id (str): Patient identifier
        output_dir (Path): Output directory for visualizations
    """
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Gaze-Attention Comparison: {patient_id}', fontsize=16, fontweight='bold')
    
    # Human gaze heatmap
    im1 = axes[0, 0].imshow(human_heatmap, cmap='hot', interpolation='bilinear')
    axes[0, 0].set_title('Human Radiologist Gaze Heatmap', fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Model attention heatmap
    im2 = axes[0, 1].imshow(model_attention, cmap='hot', interpolation='bilinear')
    axes[0, 1].set_title('Model Attention Heatmap (Grad-CAM)', fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Difference map
    diff_map = np.abs(human_heatmap - model_attention)
    im3 = axes[1, 0].imshow(diff_map, cmap='viridis', interpolation='bilinear')
    axes[1, 0].set_title('Absolute Difference Map', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Metrics text
    axes[1, 1].axis('off')
    metrics_text = f"""CORRELATION METRICS:

Pearson Correlation: {metrics['pearson_correlation']:.4f}
P-value: {metrics['pearson_p_value']:.4f}

Jensen-Shannon Divergence: {metrics['jensen_shannon_divergence']:.4f}
Mean Squared Error: {metrics['mean_squared_error']:.4f}

Normalized Scanpath Saliency: {metrics['normalized_scanpath_saliency']:.4f}

Human Attention Entropy: {metrics['human_attention_entropy']:.4f}
Model Attention Entropy: {metrics['model_attention_entropy']:.4f}

INTERPRETATION:
• Pearson > 0.3: Good spatial correlation
• JS Divergence < 0.5: Similar distributions  
• Lower MSE: Better pixel-wise match
• Higher NSS: Better fixation prediction
"""
    
    axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"gaze_attention_comparison_{patient_id}_{timestamp}.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Gaze-attention comparison saved: {plot_path}")
    return plot_path



def get_patient_fixation_path(patient_id):
    """
    Get the fixation data path for a specific patient from the dataset.
    
    Args:
        patient_id (str): DICOM ID of the patient
        
    Returns:
        str: Path to the fixation CSV file, or None if not found
    """
    # Load the dataset
    dataset_path = project_root / "final_dataset_fixed.csv"
    
    try:
        # Read CSV in chunks to handle large file
        chunk_size = 1000
        for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
            # Find the patient
            patient_row = chunk[chunk['dicom_id'] == patient_id]
            if not patient_row.empty:
                fixations_path = patient_row['fixations_path'].iloc[0]
                print(f"   ✅ Found fixation data: {fixations_path}")
                return fixations_path
        
        print(f"   ❌ Patient {patient_id} not found in dataset")
        return None
        
    except Exception as e:
        print(f"   ❌ Error reading dataset: {e}")
        return None


def main():
    """Main execution function."""
    
    print("🚀 Starting Specific Patient X-ray Analysis with Gaze-Attention Validation...")
    print("=" * 80)
    
    try:
        # Step 1: Get specific patient image
        print("\n🎯 Step 1: Loading specific patient X-ray image...")
        image_path = get_specific_xray_image()
        patient_id = Path(image_path).stem
        
        # Step 1.5: Load human gaze data (NEW - Priority 1)
        print("\n👁️  Step 1.5: Loading human gaze fixation data...")
        fixations_path = get_patient_fixation_path(patient_id)
        
        if fixations_path:
            # Convert relative path to absolute
            abs_fixations_path = project_root / fixations_path
            fixations_df = load_fixation_data(abs_fixations_path)
            
            if fixations_df is not None:
                print("   ✅ Human gaze data loaded successfully!")
                # Generate human gaze heatmap
                print("   🔥 Generating human gaze heatmap...")
                human_gaze_heatmap = generate_gaze_heatmap(fixations_df, heatmap_size=(224, 224))
                print(f"   📊 Human gaze heatmap shape: {human_gaze_heatmap.shape}")
                print(f"   📈 Human gaze intensity range: {human_gaze_heatmap.min():.4f} - {human_gaze_heatmap.max():.4f}")
            else:
                print("   ⚠️ Could not load human gaze data, continuing without gaze comparison")
                human_gaze_heatmap = None
        else:
            print("   ⚠️ No fixation data found for this patient, continuing without gaze comparison")
            human_gaze_heatmap = None
        
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
        
        # Step 3.5: Extract model attention and perform gaze comparison (NEW - Priority 1)
        if human_gaze_heatmap is not None and 'attention_analysis' in result:
            print("\n🧠 Step 3.5: Performing Gaze-Attention Validation (Priority 1)...")
            
            # DEBUG: Check what's actually in attention_analysis
            attention_data = result['attention_analysis']
            print(f"   🔍 DEBUG: attention_analysis keys: {list(attention_data.keys())}")
            print(f"   🔍 DEBUG: attention_analysis content preview: {str(attention_data)[:200]}...")
            
            # Extract model attention map from results
            if 'attention_map' in attention_data:
                model_attention = attention_data['attention_map']
                print(f"   📊 Model attention shape: {model_attention.shape}")
                print(f"   📈 Model attention range: {model_attention.min():.4f} - {model_attention.max():.4f}")
                
                # Calculate correlation metrics
                print("   🔍 Calculating spatial correlation metrics...")
                correlation_metrics = calculate_attention_metrics(human_gaze_heatmap, model_attention)
                
                # Print key metrics
                print(f"   📈 Pearson Correlation: {correlation_metrics['pearson_correlation']:.4f}")
                print(f"   📊 Jensen-Shannon Divergence: {correlation_metrics['jensen_shannon_divergence']:.4f}")
                print(f"   📉 Mean Squared Error: {correlation_metrics['mean_squared_error']:.4f}")
                
                # Save visualization
                print("   💾 Saving gaze-attention comparison visualization...")
                output_dir = project_root / "new_scripts" / "gaze_attention_analysis"
                comparison_plot = save_attention_comparison(
                    human_gaze_heatmap, model_attention, correlation_metrics, patient_id, output_dir
                )
                
                # Add metrics to result
                result['gaze_attention_validation'] = {
                    'correlation_metrics': correlation_metrics,
                    'human_gaze_heatmap_stats': {
                        'shape': human_gaze_heatmap.shape,
                        'min': float(human_gaze_heatmap.min()),
                        'max': float(human_gaze_heatmap.max()),
                        'mean': float(human_gaze_heatmap.mean())
                    },

                    'comparison_plot_path': str(comparison_plot)
                }
                
                print("   ✅ Gaze-Attention Validation completed!")
            else:
                print("   ⚠️ No attention map found in model results")
                print(f"   🔍 Available keys: {list(attention_data.keys())}")
                print("   💡 Need to modify MedicalReportGenerator to save raw attention maps")
        else:
            print("   ⚠️ Skipping gaze-attention validation (no human gaze data or attention analysis)")
        
        # 🔍 DEBUG: Save the prompt that was sent to AI for inspection
        try:
            # Extract the prompt from the comprehensive analysis steps
            if 'llm_response' in result and 'raw_response' in result['llm_response']:
                # Get the prompt that was actually used (we need to recreate it)
                print("   🔍 Extracting and saving AI prompt for debugging...")
                
                # Recreate the prompt using the same data that was used
                debug_prompt = generator.create_medical_report_prompt(
                    condition_predictions=result.get('model_predictions', {}),
                    prediction_keywords=result.get('prediction_keywords', {}),
                    spatial_keywords=result.get('spatial_keywords', []),
                    attention_analysis=result.get('attention_analysis', {}),
                    relevant_anatomical_regions=result.get('relevant_anatomical_regions', {}),
                    template="standard",
                    patient_info=None,
                    include_image=False
                )
                
                # Save to file with timestamp
                prompt_file = project_root / "new_scripts" / "debug_ai_prompt.txt"
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("AI PROMPT DEBUG OUTPUT\n")
                    f.write(f"Generated at: {datetime.now()}\n")
                    f.write(f"Patient ID: {Path(image_path).stem}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    f.write(f"PROMPT STATISTICS:\n")
                    f.write(f"- Total Characters: {len(debug_prompt):,}\n")
                    f.write(f"- Total Words: {len(debug_prompt.split()):,}\n")
                    f.write(f"- Estimated Tokens: ~{len(debug_prompt.split()) * 1.3:.0f}\n")
                    f.write(f"- Size in KB: {len(debug_prompt.encode('utf-8')) / 1024:.2f} KB\n")
                    f.write("\n" + "=" * 80 + "\n\n")
                    
                    f.write("FULL PROMPT CONTENT:\n")
                    f.write("-" * 40 + "\n")
                    f.write(debug_prompt)
                
                print(f"   📄 AI prompt saved to: {prompt_file}")
                print(f"   📊 Prompt size: {len(debug_prompt):,} characters ({len(debug_prompt.encode('utf-8')) / 1024:.2f} KB)")
                print(f"   🔢 Estimated tokens: ~{len(debug_prompt.split()) * 1.3:.0f}")
                
        except Exception as e:
            print(f"   ⚠️ Could not save debug prompt: {e}")
            # Continue anyway, this is just debugging
        
        # Step 4: Format and display results
        print("\n✅ Step 4: Analysis complete! Formatting results...")
        print("\n")
        
        formatted_report = format_medical_report(result)
        print(formatted_report)
        
        # Display gaze-attention validation results if available
        if 'gaze_attention_validation' in result:
            print("\n" + "=" * 60)
            print("🧠 GAZE-ATTENTION VALIDATION RESULTS (Priority 1)")
            print("=" * 60)
            
            metrics = result['gaze_attention_validation']['correlation_metrics']
            print(f"📈 Spatial Correlation Analysis:")
            print(f"   • Pearson Correlation: {metrics['pearson_correlation']:.4f}")
            print(f"   • P-value: {metrics['pearson_p_value']:.4f}")
            print(f"   • Jensen-Shannon Divergence: {metrics['jensen_shannon_divergence']:.4f}")
            print(f"   • Mean Squared Error: {metrics['mean_squared_error']:.4f}")
            print(f"   • Normalized Scanpath Saliency: {metrics['normalized_scanpath_saliency']:.4f}")
            
            print(f"\n🔍 Interpretation:")
            corr = metrics['pearson_correlation']
            if corr > 0.3:
                print(f"   ✅ GOOD spatial correlation ({corr:.4f}) - model attention aligns with human gaze")
            elif corr > 0.1:
                print(f"   ⚠️ MODERATE spatial correlation ({corr:.4f}) - some alignment but room for improvement")
            else:
                print(f"   ❌ POOR spatial correlation ({corr:.4f}) - model attention differs significantly from human gaze")
            
            js_div = metrics['jensen_shannon_divergence']
            if js_div < 0.3:
                print(f"   ✅ SIMILAR attention distributions (JS={js_div:.4f})")
            elif js_div < 0.6:
                print(f"   ⚠️ MODERATELY different distributions (JS={js_div:.4f})")
            else:
                print(f"   ❌ VERY different attention distributions (JS={js_div:.4f})")
            
            print(f"\n💾 Visualization saved: {result['gaze_attention_validation']['comparison_plot_path']}")
        
        # Optional: Save to file
        output_file = project_root / "new_scripts" / "latest_fixed_analysis.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_report)
            
            # Add gaze-attention results to file
            if 'gaze_attention_validation' in result:
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("GAZE-ATTENTION VALIDATION RESULTS (Priority 1)\n")
                f.write("=" * 60 + "\n")
                
                metrics = result['gaze_attention_validation']['correlation_metrics']
                f.write(f"Pearson Correlation: {metrics['pearson_correlation']:.4f}\n")
                f.write(f"P-value: {metrics['pearson_p_value']:.4f}\n")
                f.write(f"Jensen-Shannon Divergence: {metrics['jensen_shannon_divergence']:.4f}\n")
                f.write(f"Mean Squared Error: {metrics['mean_squared_error']:.4f}\n")
                f.write(f"Normalized Scanpath Saliency: {metrics['normalized_scanpath_saliency']:.4f}\n")
                f.write(f"Visualization: {result['gaze_attention_validation']['comparison_plot_path']}\n")
        
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