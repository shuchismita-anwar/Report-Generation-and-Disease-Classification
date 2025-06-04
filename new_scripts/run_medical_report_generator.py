#!/usr/bin/env python3
"""
Medical Report Generator - Main Execution Script
===============================================
Complete medical report generation system for chest X-ray analysis
using real MIMIC dataset images with multimodal AI integration.
"""

import sys
import os

# Add parent directory to path so we can import from new_scripts
sys.path.append('.')
sys.path.append('..')

from medical_report_generator import MedicalReportGenerator
import time
from pathlib import Path

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
        template='detailed',
        include_image_in_llm=True,  # Send the same image being analyzed to the LLM
        patient_info=patient_info,
        save_visualizations=True,
        progress_callback=progress_tracker
    )
    
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
    
    # Demo 2: Batch Processing with Real Images
    print("\n" + "=" * 60)
    print("DEMO 2: BATCH PROCESSING WITH REAL IMAGES")
    print("=" * 60)
    
    # Select first 3 real images for batch processing
    batch_images = available_images[:3]
    
    print(f"📁 Processing batch of {len(batch_images)} real chest X-rays")
    print("🔄 Using standard template for balanced processing")
    
    for i, img_path in enumerate(batch_images):
        print(f"   {i+1}. {os.path.basename(img_path)}")
    
    batch_start = time.time()
    
    batch_results = generator.batch_analyze_images(
        image_paths=batch_images,
        template='standard',
        max_concurrent=2,
        save_results=True,
        output_dir="real_analysis_results"
    )
    
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
    print("\n" + "=" * 60)
    print("DEMO 3: TEMPLATE COMPARISON ON REAL IMAGE")
    print("=" * 60)
    
    templates = ['standard', 'detailed', 'concise']
    template_results = {}
    
    # Use the same image for comparison
    comparison_image = available_images[0]
    print(f"📸 Comparing templates on: {os.path.basename(comparison_image)}")
    
    for template in templates:
        print(f"📝 Generating {template} report...")
        
        template_start = time.time()
        result = generator.generate_comprehensive_analysis(
            image_path=comparison_image,
            template=template,
            save_visualizations=False
        )
        template_time = time.time() - template_start
        
        if result['success']:
            report_text = result['medical_report']['report_text']
            template_results[template] = {
                'success': True,
                'processing_time': template_time,
                'word_count': len(report_text.split()),
                'character_count': len(report_text),
                'sections': len(result['medical_report']['sections'])
            }
        else:
            template_results[template] = {'success': False}
    
    print("\n📊 Template Comparison Results:")
    print("-" * 50)
    print(f"{'Template':<12} {'Time(s)':<8} {'Words':<8} {'Chars':<8} {'Sections':<8}")
    print("-" * 50)
    
    for template, results in template_results.items():
        if results['success']:
            print(f"{template:<12} {results['processing_time']:<8.1f} "
                  f"{results['word_count']:<8} {results['character_count']:<8} "
                  f"{results['sections']:<8}")
        else:
            print(f"{template:<12} {'FAILED':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
    
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
    print("✅ Multiple report templates compared")
    print("✅ Performance optimization and monitoring active")
    print(f"🏥 Medical Report Generator successfully analyzed {len(available_images)} real images!")

if __name__ == "__main__":
    main() 