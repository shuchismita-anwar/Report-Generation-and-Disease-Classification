#!/usr/bin/env python3
"""
Test Report Loading
==================
Test script to verify that radiology reports (.txt files) are being loaded correctly.
"""

import os
import sys
sys.path.append('automation')

from data_discovery import load_ground_truth_report, _clean_radiology_report

def test_report_loading():
    """Test loading of radiology reports."""
    print("🧪 Testing radiology report loading...")
    
    # Test file paths from the examples
    test_reports = [
        "../../mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye/patient_10003400/CXR-DICOM/s56466802.txt",
        "../../mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye/patient_10018423/CXR-DICOM/s58815716.txt",
        "../../mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye/patient_10028159/CXR-DICOM/s56342713.txt"
    ]
    
    for i, report_path in enumerate(test_reports, 1):
        print(f"\n📄 Test {i}: Loading {os.path.basename(report_path)}")
        
        try:
            if os.path.exists(report_path):
                result = load_ground_truth_report(report_path)
                
                if result:
                    print(f"✅ Successfully loaded report")
                    print(f"   File type: {result.get('file_type', 'unknown')}")
                    print(f"   Report length: {len(result['report_text'])} characters")
                    print(f"   Raw length: {len(result.get('raw_text', ''))} characters")
                    
                    # Show first few lines of cleaned report
                    lines = result['report_text'].split('\n')[:5]
                    print(f"   First few lines:")
                    for line in lines:
                        if line.strip():
                            print(f"     {line.strip()}")
                else:
                    print(f"❌ Failed to load report")
            else:
                print(f"⚠️  File not found: {report_path}")
                
        except Exception as e:
            print(f"❌ Error loading report: {e}")
    
    print(f"\n🧪 Testing report cleaning function...")
    
    # Test cleaning function with sample content
    sample_content = """ WET READ: ___ ___ 9:05 AM
  No acute cardiopulmonary process.
 ______________________________________________________________________________
                                 FINAL REPORT
 EXAMINATION:   CHEST (PA AND LAT)
 
 INDICATION:  ___M with CP bowel question pneumonia.
 
 TECHNIQUE:  Chest PA and lateral
 
 COMPARISON:  None available for comparison.
 
 FINDINGS: 
 
 Low lung volumes are noted.  The cardiomediastinal/hilar contours are
 unremarkable.  There is no pleural effusion or pneumothorax.
 
 IMPRESSION: 
 
 No acute cardiopulmonary process."""
    
    cleaned = _clean_radiology_report(sample_content)
    print(f"✅ Cleaned sample report ({len(cleaned)} chars):")
    print(f"   {cleaned[:150]}..." if len(cleaned) > 150 else cleaned)

if __name__ == "__main__":
    test_report_loading() 