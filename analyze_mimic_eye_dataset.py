import pandas as pd
import os
import json
from collections import Counter
import numpy as np

def analyze_mimic_eye_dataset():
    """Analyze the MIMIC-Eye dataset and generate comprehensive statistics."""
    
    print("=== MIMIC-Eye Dataset Analysis ===\n")
    
    # Read the main dataset file
    df = pd.read_csv('final_dataset.csv')
    
    # Basic statistics
    print("1. DATASET OVERVIEW")
    print("-" * 50)
    print(f"Total records: {len(df):,}")
    print(f"Unique patients: {df['patient_folder'].nunique():,}")
    print(f"Records with REFLACX data: {df['ifReflacx'].sum():,}")
    print(f"Records with eye gaze data: {df['ifEyegaze'].sum():,}")
    print(f"Records with both REFLACX and eye gaze: {(df['ifReflacx'] & df['ifEyegaze']).sum():,}")
    print(f"Records with images: {df['image_path'].notna().sum():,}")
    print(f"Records with reports: {df['report_path'].notna().sum():,}")
    
    # Data modality breakdown
    print("\n2. DATA MODALITY BREAKDOWN")
    print("-" * 50)
    print(f"Image-only records: {((~df['ifReflacx']) & (~df['ifEyegaze']) & df['image_path'].notna()).sum():,}")
    print(f"Image + REFLACX records: {(df['ifReflacx'] & (~df['ifEyegaze']) & df['image_path'].notna()).sum():,}")
    image_gaze_count = ((~df['ifReflacx']) & df['ifEyegaze'] & df['image_path'].notna()).sum()
    print(f"Image + Eye Gaze records: {image_gaze_count:,}")
    multimodal_count = (df['ifReflacx'] & df['ifEyegaze'] & df['image_path'].notna()).sum()
    print(f"Full multimodal records (Image + REFLACX + Eye Gaze): {multimodal_count:,}")
    
    # Analyze REFLACX data
    print("\n3. REFLACX DATA ANALYSIS")
    print("-" * 50)
    reflacx_df = df[df['ifReflacx'] == True]
    print(f"Total REFLACX records: {len(reflacx_df):,}")
    print(f"Unique REFLACX patients: {reflacx_df['patient_folder'].nunique():,}")
    print(f"REFLACX records with eye gaze: {reflacx_df['ifEyegaze'].sum():,}")
    
    # Analyze eye gaze data
    print("\n4. EYE GAZE DATA ANALYSIS")
    print("-" * 50)
    gaze_df = df[df['ifEyegaze'] == True]
    print(f"Total eye gaze records: {len(gaze_df):,}")
    print(f"Unique eye gaze patients: {gaze_df['patient_folder'].nunique():,}")
    print(f"Eye gaze records with REFLACX: {gaze_df['ifReflacx'].sum():,}")
    
    # Analyze image data
    print("\n5. IMAGE DATA ANALYSIS")
    print("-" * 50)
    image_df = df[df['image_path'].notna()]
    print(f"Total image records: {len(image_df):,}")
    print(f"Unique image patients: {image_df['patient_folder'].nunique():,}")
    
    # Analyze report data
    print("\n6. REPORT DATA ANALYSIS")
    print("-" * 50)
    report_df = df[df['report_path'].notna()]
    print(f"Total report records: {len(report_df):,}")
    print(f"Unique report patients: {report_df['patient_folder'].nunique():,}")
    
    # Count files by type
    print("\n7. FILE TYPE ANALYSIS")
    print("-" * 50)
    
    # Count CSV files (gaze data)
    csv_count = 0
    jpg_count = 0
    json_count = 0
    
    for root, dirs, files in os.walk('mimic-eye'):
        for file in files:
            if file.endswith('.csv'):
                csv_count += 1
            elif file.endswith('.jpg'):
                jpg_count += 1
            elif file.endswith('.json'):
                json_count += 1
    
    print(f"Total CSV files (gaze data): {csv_count:,}")
    print(f"Total JPG files (images): {jpg_count:,}")
    print(f"Total JSON files (transcripts): {json_count:,}")
    
    # Analyze spreadsheets
    print("\n8. SPREADSHEET ANALYSIS")
    print("-" * 50)
    
    try:
        cxr_meta = pd.read_csv('mimic-eye/spreadsheets/cxr_meta.csv')
        print(f"CXR metadata records: {len(cxr_meta):,}")
        print(f"Unique CXR studies: {cxr_meta['study_id'].nunique():,}")
    except:
        print("CXR metadata file not accessible")
    
    try:
        reflacx_clinical = pd.read_csv('mimic-eye/spreadsheets/reflacx_clinical_eye.csv')
        print(f"REFLACX clinical records: {len(reflacx_clinical):,}")
    except:
        print("REFLACX clinical file not accessible")
    
    # Summary table
    print("\n9. SUMMARY TABLE")
    print("-" * 50)
    print("| Metric | Count |")
    print("|--------|-------|")
    print(f"| Total Patients | {df['patient_folder'].nunique():,} |")
    print(f"| Total Records | {len(df):,} |")
    print(f"| Image Records | {df['image_path'].notna().sum():,} |")
    print(f"| Report Records | {df['report_path'].notna().sum():,} |")
    print(f"| REFLACX Records | {df['ifReflacx'].sum():,} |")
    print(f"| Eye Gaze Records | {df['ifEyegaze'].sum():,} |")
    print(f"| Full Multimodal Records | {(df['ifReflacx'] & df['ifEyegaze'] & df['image_path'].notna()).sum():,} |")
    print(f"| JPG Images | {jpg_count:,} |")
    print(f"| CSV Files | {csv_count:,} |")
    print(f"| JSON Files | {json_count:,} |")

if __name__ == "__main__":
    analyze_mimic_eye_dataset() 