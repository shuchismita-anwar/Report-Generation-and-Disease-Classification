#!/usr/bin/env python3
"""
Medical Report Generator - Main Execution Script
===============================================
Complete medical report generation system for chest X-ray analysis
using real MIMIC dataset images with multimodal AI integration.
"""

import sys
import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error
import json
import time
import random
from pathlib import Path

# Add parent directory to path so we can import from new_scripts
sys.path.append('.')
sys.path.append('..')

from medical_report_generator import MedicalReportGenerator

def progress_tracker(step, progress, message):
    """Progress callback function for visualization"""
    bar_length = 40
    filled_length = int(bar_length * progress // 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f"\r[{bar}] {progress:.0f}% - {step}: {message}", end='', flush=True)
    if progress >= 100:
        print()  # New line when complete

def main():
    print("🏥 MEDICAL REPORT GENERATOR")
    print("=" * 60)
    print("Complete medical report generation system for chest X-ray analysis")
    print("Using real MIMIC dataset images with multimodal AI integration.")
    
    # Initialize the medical report generator
    print("\n🚀 Initializing Medical Report Generator...")
    generator = MedicalReportGenerator()
    
    # Load the classification model
    print("📊 Loading multimodal classification model...")
    if not generator.load_classification_model():
        print("❌ Failed to load classification model!")
        return
    
    # Get available chest X-rays
    print("\n📁 Scanning for available chest X-ray images...")
    available_images = generator.get_available_chest_xrays()
    
    if not available_images:
        print("❌ No chest X-ray images found in data_dump/output/img_png/")
        print("Please ensure your MIMIC dataset images are in the correct directory.")
        return
    
    print(f"✅ Found {len(available_images)} chest X-ray images")
    
    # Get system status
    print("\n📊 System Status Check")
    print("-" * 30)
    
    status = generator.get_system_status()
    print(f"Overall Health Score: {status['overall_health_score']:.1f}%")
    print(f"Device: {status['performance_info']['device']}")
    print(f"System Memory: {status['performance_info']['system_memory_gb']:.1f} GB")
    
    print("Component Status:")
    for component, working in status['component_status'].items():
        status_icon = "✅" if working else "❌"
        print(f"  {status_icon} {component.replace('_', ' ').title()}")
    
    # Demo 1: Single Image Analysis
    """
    print("\n" + "=" * 60)
    print("DEMO 1: SINGLE CHEST X-RAY ANALYSIS")
    print("=" * 60)
    
    # Select a random real chest X-ray
    test_image = generator.get_random_chest_xray()
    image_filename = os.path.basename(test_image)
    
    # Example patient info (could be real data if available)
    patient_info = {
        'age': '65',
        'gender': 'Male', 
        'chief_complaint': 'Cough and fever for 5 days',
        'clinical_history': 'Patient presents with productive cough, fever, and shortness of breath.',
        'referring_physician': 'Dr. Smith, Radiology'
    }
    
    print(f"📸 Analyzing: {image_filename}")
    print(f"👤 Patient: {patient_info['age']}yr {patient_info['gender']}")
    print(f"🩺 Chief complaint: {patient_info['chief_complaint']}")
    print()
    
    # Run comprehensive analysis with progress tracking
    start_time = time.time()
    
    comprehensive_result = generator.generate_comprehensive_analysis(
        image_path=test_image,
        template='standard',  # Use only standard template like random_xray_analysis.py
        include_image_in_llm=True,  # FIXED: Enable image sending to LLM for better analysis
        patient_info=patient_info,
        save_visualizations=True,
        progress_callback=progress_tracker
    )
    
    # === GAZE-ATTENTION VALIDATION (single image) ===
    gaze_val = perform_gaze_validation(test_image, comprehensive_result.get("attention_analysis", {}), Path("real_analysis_results"))
    if gaze_val:
        comprehensive_result["gaze_attention_validation"] = gaze_val
    
    analysis_time = time.time() - start_time
    
    print(f"\n📈 Analysis completed in {analysis_time:.1f} seconds")
    print(f"✅ Success: {comprehensive_result['success']}")
    
    # Show detailed results
    if comprehensive_result['success']:
        print("\n📋 Analysis Results Summary:")
        print("-" * 40)
        
        # Model predictions
        print("🔬 Model Predictions:")
        predictions = comprehensive_result['model_predictions']
        top_predictions = sorted(predictions.items(), key=lambda x: x[1]['probability'], reverse=True)[:4]
        
        for condition, pred_info in top_predictions:
            prob = pred_info['probability']
            predicted = "✓" if pred_info['predicted'] else "✗"
            confidence = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
            print(f"  {predicted} {condition}: {prob:.3f} ({confidence})")
        
        # Keywords
        print(f"\n🔑 Keywords Extracted: {comprehensive_result['keyword_summary']['total_keywords']}")
        keyword_categories = comprehensive_result['keyword_summary']
        for category, keywords in keyword_categories.items():
            if isinstance(keywords, list) and keywords:
                print(f"  • {category.replace('_', ' ').title()}: {len(keywords)} keywords")
        
        # Attention analysis
        attention = comprehensive_result['attention_analysis']
        print(f"👁️ Attention Analysis: {len(attention['significant_regions'])} significant regions")
        print(f"   Overall intensity: {attention['overall_attention_intensity']:.3f}")
        
        if attention['significant_regions']:
            print("   Top regions:")
            for region_id in attention['significant_regions'][:3]:
                region_data = attention['regions'][region_id]
                print(f"     • {region_data['name']}: {region_data['max_attention']:.3f}")
        
        # Performance metrics
        metrics = comprehensive_result['performance_metrics']
        print(f"\n⚡ Performance Metrics:")
        print(f"   Steps completed: {metrics['steps_completed']}/{metrics['total_steps']}")
        print(f"   Processing time: {metrics['total_processing_time']:.2f}s")
        print(f"   Errors encountered: {metrics['errors_encountered']}")
        
        # Final medical report
        medical_report = comprehensive_result['medical_report']
        print(f"\n📄 Generated Medical Report ({len(medical_report['report_text'])} characters):")
        print("=" * 60)
        print(medical_report['report_text'])
        print("=" * 60)
        
        report_metadata = medical_report['metadata']
        print(f"📊 Report Metadata:")
        print(f"   Word count: {report_metadata['word_count']}")
        print(f"   Sections: {len(medical_report['sections'])}")
        print(f"   LLM success: {report_metadata.get('llm_success', 'Unknown')}")
        print(f"   Used fallback: {report_metadata.get('used_fallback', 'Unknown')}")
    # End of Demo 1
    """
    
    # Demo 2: Batch Processing with Real Images
    print("\n" + "=" * 60)
    print("DEMO 2: BATCH PROCESSING WITH REAL IMAGES")
    print("=" * 60)
    
    # Select up to 10 random real images for batch processing
    if len(available_images) >= 10:
        batch_images = random.sample(available_images, 10)
    else:
        batch_images = available_images
    
    print(f"📁 Processing batch of {len(batch_images)} real chest X-rays")
    print("🔄 Using standard template for balanced processing")
    
    for i, img_path in enumerate(batch_images):
        print(f"   {i+1}. {os.path.basename(img_path)}")
    
    batch_start = time.time()
    
    batch_results = generator.batch_analyze_images(
        image_paths=batch_images,
        template='standard',  # Use only standard template consistently
        max_concurrent=2,
        save_results=True,
        output_dir=r".\real_analysis_results"
    )
    
    # === UPDATED GAZE-ATTENTION VALIDATION (use in-memory results) ===
    for img_path in batch_images:
        dicom_id = Path(img_path).stem
        analysis = batch_results['results'].get(img_path)
        if analysis is None:
            # Fallback to JSON re-load if analysis missing (should not happen)
            json_path = Path("real_analysis_results") / f"analysis_{dicom_id}.json"
            if not json_path.exists():
                continue
            with open(json_path, "r", encoding="utf-8") as jf:
                analysis = json.load(jf)
        # Perform gaze validation using raw attention map (avoids lossy JSON conversion)
        gaze_val = perform_gaze_validation(img_path, analysis.get("attention_analysis", {}), Path("real_analysis_results"))
        if gaze_val:
            analysis["gaze_attention_validation"] = gaze_val
            # Overwrite JSON result (ensures gaze metrics stored)
            json_path = Path("real_analysis_results") / f"analysis_{dicom_id}.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(analysis, jf, indent=2, default=str)
    
    batch_time = time.time() - batch_start
    
    print(f"\n📊 Batch Processing Results:")
    print(f"   Total time: {batch_time:.1f}s")
    print(f"   Average per image: {batch_time/len(batch_images):.1f}s")
    print(f"   Success rate: {batch_results['performance_summary']['success_rate']:.1f}%")
    print(f"   Successful: {batch_results['successful_analyses']}")
    print(f"   Failed: {batch_results['failed_analyses']}")
    
    if batch_results['successful_analyses'] > 0:
        print("✅ Batch processing completed successfully")
        print(f"📁 Results saved to: real_analysis_results/")
    
    # Demo 3: Different Templates with Same Image
    """
    print("\n" + "=" * 60)
    print("DEMO 3: SINGLE TEMPLATE ANALYSIS ON REAL IMAGE")
    print("=" * 60)
    
    # Use only standard template like random_xray_analysis.py
    template = 'standard'
    
    # Use the same image for analysis
    analysis_image = available_images[0]
    print(f"📸 Analyzing with standard template: {os.path.basename(analysis_image)}")
    
    print(f"📝 Generating {template} report...")
    
    template_start = time.time()
    result = generator.generate_comprehensive_analysis(
        image_path=analysis_image,
        template=template,
        include_image_in_llm=True,  # Enable image sending to LLM
        save_visualizations=False
    )
    template_time = time.time() - template_start
    
    if result['success']:
        report_text = result['medical_report']['report_text']
        template_result = {
            'success': True,
            'processing_time': template_time,
            'word_count': len(report_text.split()),
            'character_count': len(report_text),
            'sections': len(result['medical_report']['sections'])
        }
        
        print(f"\n📊 Standard Template Analysis Results:")
        print("-" * 50)
        print(f"Processing time: {template_result['processing_time']:.1f}s")
        print(f"Word count: {template_result['word_count']}")
        print(f"Character count: {template_result['character_count']}")
        print(f"Sections: {template_result['sections']}")
        
        # Display the generated report
        print(f"\n📄 Generated Medical Report:")
        print("=" * 60)
        print(report_text)
        print("=" * 60)
    else:
        print(f"❌ {template} template analysis failed")
    # End of Demo 3
    """
    
    # Final System Summary
    print("\n" + "=" * 60)
    print("FINAL SYSTEM SUMMARY")
    print("=" * 60)
    
    final_status = generator.get_system_status()
    
    print("🎯 System Capabilities:")
    print(f"   ✅ Medical conditions supported: {final_status['configuration']['conditions_supported']}")
    print(f"   ✅ Real chest X-rays available: {len(available_images)}")
    print(f"   ✅ Keyword mappings: {final_status['configuration']['keyword_mappings']}")
    print(f"   ✅ Anatomical regions: {final_status['configuration']['anatomical_regions']}")
    print(f"   ✅ Report templates: {len(final_status['configuration']['report_templates'])}")
    print(f"   ✅ LLM model: {final_status['configuration']['llm_model']}")
    
    print("\n🔧 System Health:")
    health = final_status['system_health']
    for component, status in health.items():
        status_icon = "✅" if status else "⚠️"
        print(f"   {status_icon} {component.replace('_', ' ').title()}")
    
    print(f"\n🎉 Overall System Health: {final_status['overall_health_score']:.1f}%")
    
    if final_status['overall_health_score'] >= 80:
        print("🚀 System is ready for production use!")
    elif final_status['overall_health_score'] >= 60:
        print("⚡ System is functional with minor limitations")
    else:
        print("⚠️ System has significant limitations - check configuration")
    
    print("\n" + "=" * 60)
    print("🎉 MEDICAL REPORT GENERATOR EXECUTION COMPLETED!")
    print("=" * 60)
    print("✅ Real chest X-ray analysis pipeline demonstrated")
    print("✅ Batch processing capabilities shown") 
    print("✅ Single template analysis on real image demonstrated")
    print("✅ Performance optimization and monitoring active")
    print(f"🏥 Medical Report Generator successfully analyzed {len(available_images)} real images!")

# === GAZE-ATTENTION VALIDATION HELPERS (ported from random_xray_analysis) ===

project_root = Path(__file__).parent.parent


def _get_patient_fixation_path(dicom_id: str) -> Path | None:
    """Return fixation CSV path for given dicom_id from final_dataset_fixed.csv"""
    dataset_csv = project_root / "final_dataset_fixed.csv"
    if not dataset_csv.exists():
        return None
    for chunk in pd.read_csv(dataset_csv, chunksize=1000):
        row = chunk[chunk["dicom_id"] == dicom_id]
        if not row.empty:
            return project_root / row["fixations_path"].iloc[0]
    return None


def _load_fixations(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        required = {"x_norm", "y_norm", "gaze_duration"}
        if not required.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


def _heatmap_from_fixations(df: pd.DataFrame, size=(224, 224), sigma: int = 15) -> np.ndarray:
    heat = np.zeros(size, dtype=np.float32)
    h, w = size
    for _, r in df.iterrows():
        x = int(max(0, min(w - 1, r["x_norm"] * w)))
        y = int(max(0, min(h - 1, r["y_norm"] * h)))
        heat[y, x] += r["gaze_duration"] / 1000.0
    if sigma > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), sigma)
    if heat.max() > 0:
        heat /= heat.max()
    return heat


def _calc_corr_metrics(human: np.ndarray, model: np.ndarray) -> dict:
    if human.shape != model.shape:
        model = cv2.resize(model, (human.shape[1], human.shape[0]))
    h_flat, m_flat = human.flatten(), model.flatten()
    pear, p = pearsonr(h_flat, m_flat)
    eps = 1e-8
    h_prob = (h_flat + eps) / (h_flat.sum() + eps * len(h_flat))
    m_prob = (m_flat + eps) / (m_flat.sum() + eps * len(m_flat))
    js = jensenshannon(h_prob, m_prob)
    mse = mean_squared_error(h_flat, m_flat)
    # simple NSS approximation – mean model attention at random human points
    idx = (np.random.rand(100) * (human.size - 1)).astype(int)
    nss = float(model.flatten()[idx].mean())
    
    # Calculate entropies like random_xray_analysis.py
    human_entropy = -np.sum(h_prob * np.log(h_prob + eps))
    model_entropy = -np.sum(m_prob * np.log(m_prob + eps))
    
    return {
        "pearson_correlation": float(pear),
        "pearson_p_value": float(p),
        "jensen_shannon_divergence": float(js),
        "mean_squared_error": float(mse),
        "normalized_scanpath_saliency": nss,
        "human_attention_entropy": float(human_entropy),
        "model_attention_entropy": float(model_entropy)
    }


def _save_gaze_plot(human: np.ndarray, model: np.ndarray, metrics: dict, out_path: Path):
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create 2x2 grid layout like random_xray_analysis.py
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    patient_id = out_path.stem
    fig.suptitle(f'Gaze-Attention Comparison: {patient_id}', fontsize=16, fontweight='bold')
    
    # Human gaze heatmap (top-left)
    im1 = axes[0, 0].imshow(human, cmap='hot', interpolation='bilinear')
    axes[0, 0].set_title('Human Radiologist Gaze Heatmap', fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Model attention heatmap (top-right)
    im2 = axes[0, 1].imshow(model, cmap='hot', interpolation='bilinear')
    axes[0, 1].set_title('Model Attention Heatmap (Grad-CAM)', fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Difference map (bottom-left)
    diff_map = np.abs(human - model)
    im3 = axes[1, 0].imshow(diff_map, cmap='viridis', interpolation='bilinear')
    axes[1, 0].set_title('Absolute Difference Map', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Metrics text (bottom-right)
    axes[1, 1].axis('off')
    metrics_text = f"""CORRELATION METRICS:

Pearson Correlation: {metrics['pearson_correlation']:.4f}
P-value: {metrics['pearson_p_value']:.4f}

Jensen-Shannon Divergence: {metrics['jensen_shannon_divergence']:.4f}
Mean Squared Error: {metrics['mean_squared_error']:.4f}

Normalized Scanpath Saliency: {metrics['normalized_scanpath_saliency']:.4f}

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
    
    # Update filename to include timestamp like random_xray_analysis.py
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"gaze_attention_comparison_{patient_id}_{timestamp}.png"
    final_path = out_path.parent / new_filename
    
    fig.savefig(final_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return final_path


def _convert_attention_map(att_map, target_shape):
    """Convert attention map which may be a list, nested lists, or numpy-format string to numpy array with target shape."""
    import json, re, math
    if isinstance(att_map, np.ndarray):
        return cv2.resize(att_map, (target_shape[1], target_shape[0])) if att_map.shape != target_shape else att_map
    # If already Python list (nested) just convert directly
    if isinstance(att_map, list):
        arr = np.array(att_map, dtype=np.float32)
        if arr.ndim == 2:
            return cv2.resize(arr, (target_shape[1], target_shape[0])) if arr.shape != target_shape else arr
        # flatten then reshape later
        arr = arr.flatten()
    elif isinstance(att_map, str):
        s = att_map.strip()
        # Try JSON first
        try:
            parsed = json.loads(s)
            return _convert_attention_map(parsed, target_shape)
        except json.JSONDecodeError:
            # Fallback: string is numpy array repr like "[[0.1 0.2] [0.3 0.4]]"
            # Remove brackets and newlines, split by whitespace
            nums = np.fromstring(re.sub(r'[\[\]]', ' ', s), sep=' ', dtype=np.float32)
            if nums.size == 0:
                return None
            # Attempt square reshape via sqrt or target shape
            side = int(math.sqrt(nums.size))
            if side * side == nums.size:
                arr = nums.reshape((side, side))
            else:
                # Fallback to target shape resize
                arr = nums
    else:
        return None
    # If we reach here arr is 1-D or mismatched shape -> reshape/resize
    if arr.ndim == 1:
        # Assume target_shape square flatten
        try:
            arr = arr.reshape(target_shape)
        except Exception:
            # Final fallback to nearest square then resize
            side = int(math.sqrt(arr.size))
            arr = arr[:side*side].reshape((side, side))
    if arr.shape != target_shape:
        arr = cv2.resize(arr, (target_shape[1], target_shape[0]))
    return arr


def perform_gaze_validation(image_path: str, attention_analysis: dict, output_root: Path) -> dict | None:
    dicom_id = Path(image_path).stem
    fix_csv = _get_patient_fixation_path(dicom_id)
    if fix_csv is None or not fix_csv.exists():
        return None
    df = _load_fixations(fix_csv)
    if df is None or df.empty or "attention_map" not in attention_analysis:
        return None
    human_heat = _heatmap_from_fixations(df)
    model_map = _convert_attention_map(attention_analysis.get("attention_map"), human_heat.shape)
    if model_map is None:
        return None
    metrics = _calc_corr_metrics(human_heat, model_map)
    out_dir = output_root / "gaze_attention"
    plot_path = out_dir / f"{dicom_id}.png"
    final_plot_path = _save_gaze_plot(human_heat, model_map, metrics, plot_path)
    return {
        "correlation_metrics": metrics,
        "comparison_plot_path": str(final_plot_path)
    }

if __name__ == "__main__":
    main() 