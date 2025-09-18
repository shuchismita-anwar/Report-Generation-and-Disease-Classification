#!/usr/bin/env python3
"""
Test script for enhanced multi-condition reporting
"""

import sys
import os
from medical_report_generator import MedicalReportGenerator

def test_enhanced_analysis():
    """Test the enhanced multi-condition analysis"""
    
    # Target the specific image that had multiple significant findings
    image_path = '../data_dump/output/img_png/a5bf1678-df56650d-3d94165c-71ef2d00-509ad013.png'
    
    print("🔬 Testing Enhanced Multi-Condition Analysis")
    print("=" * 60)
    
    try:
        generator = MedicalReportGenerator()
        print("✅ Generator initialized successfully")
        
        # Run comprehensive analysis
        print(f"📊 Analyzing: {os.path.basename(image_path)}")
        result = generator.generate_comprehensive_analysis(
            image_path=image_path, 
            template='standard'
        )
        
        if result['success']:
            print("\n🎯 MODEL PREDICTIONS:")
            print("-" * 40)
            
            # Show all predictions sorted by confidence
            predictions = result['model_predictions']
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1]['probability'], reverse=True)
            
            for condition, pred in sorted_preds:
                prob = pred['probability']
                if prob >= 0.60:
                    status = "✅ SIGNIFICANT (≥60%)"
                elif prob >= 0.30:
                    status = "⚪ Moderate (30-60%)"
                else:
                    status = "⚫ Low (<30%)"
                print(f"  {condition:20}: {prob:6.1%} {status}")
            
            print(f"\n📝 ENHANCED REPORT:")
            print("-" * 40)
            print(result['medical_report']['report_text'])
            
            print(f"\n📊 ANALYSIS SUMMARY:")
            print("-" * 40)
            print(f"Report length: {len(result['medical_report']['report_text'])} characters")
            print(f"Sections found: {len(result['medical_report']['sections'])}")
            print(f"Total processing time: {result['performance_metrics']['total_processing_time']:.1f}s")
            print(f"Keywords extracted: {result['keyword_summary']['total_keywords']}")
            print(f"Relevant anatomical regions: {len(result['relevant_anatomical_regions'])}")
            
            # Save report
            with open('latest_enhanced_analysis.txt', 'w') as f:
                f.write("🔍 ENHANCED CLINICAL FINDINGS:\n")
                f.write("-" * 50 + "\n")
                f.write(result['medical_report']['report_text'])
            
            print(f"\n💾 Report saved to: latest_enhanced_analysis.txt")
            return True
            
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_analysis()
    sys.exit(0 if success else 1) 