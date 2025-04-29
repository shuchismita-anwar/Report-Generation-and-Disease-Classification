import os
import pandas as pd
import re
import glob
from data.paths import MIMIC_EYE_PATH
from tqdm import tqdm
import time

# Set the path to MIMIC-EYE dataset
mimic_eye_path = MIMIC_EYE_PATH

print(f"[INFO] mimic_eye_path: {mimic_eye_path}")
print(f"[INFO] Listing patient directories in: {mimic_eye_path}")

# Find all patient directories
patient_pattern = re.compile(r'^patient_(\d{8})$')
patient_dirs = []

start_time = time.time()
patient_items = os.listdir(mimic_eye_path)
print(f"[INFO] Found {len(patient_items)} items in mimic_eye_path")
with tqdm(total=len(patient_items), desc='Finding patient directories') as pbar:
    for idx, item in enumerate(patient_items):
        item_path = os.path.join(mimic_eye_path, item)
        match = patient_pattern.match(item)
        if os.path.isdir(item_path) and match:
            patient_dirs.append((item_path, match.group(1)))
            print(f"[DEBUG] Found patient dir: {item_path} (id={match.group(1)}) at index {idx}")
        pbar.update(1)
print(f"[INFO] Total patient directories found: {len(patient_dirs)}")

# Initialize lists to store paths
image_paths = []
report_paths = []
patient_ids = []
transcript_paths = []

# Process each patient directory
start_time = time.time()
with tqdm(total=len(patient_dirs), desc='Processing patient directories') as pbar:
    for patient_dir, patient_id in patient_dirs:
        patient_folder = os.path.basename(patient_dir)
        
        # Determine transcript type (REFLACX or EyeGaze)
        reflacx_folder = os.path.join(patient_dir, 'REFLACX')
        eyegaze_folder = os.path.join(patient_dir, 'EyeGaze')
        if os.path.exists(reflacx_folder):
            transcript_type = 'REFLACX'
        elif os.path.exists(eyegaze_folder):
            transcript_type = 'EyeGaze'
        else:
            transcript_type = None
        
        # Load all "id"s from metadata.csv for REFLACX patients
        reflacx_ids = []
        if transcript_type == 'REFLACX':
            metadata_path = os.path.join(reflacx_folder, 'metadata.csv')
            if os.path.exists(metadata_path):
                metadata_df = pd.read_csv(metadata_path)
                try:
                    reflacx_ids = metadata_df['id'].tolist()  # Get all "id" values
                except KeyError:
                    print(f"Warning: metadata.csv in {patient_dir} does not have an 'id' column.")
            else:
                print(f"Warning: metadata.csv not found in {reflacx_folder}")
        
        # Look for report files in CXR-DICOM folder
        cxr_dicom_folder = os.path.join(patient_dir, 'CXR-DICOM')
        if os.path.exists(cxr_dicom_folder):
            report_files = [f for f in os.listdir(cxr_dicom_folder) if f.endswith('.txt')]
            
            # Look for image files in CXR-JPG folder
            cxr_jpg_folder = os.path.join(patient_dir, 'CXR-JPG')
            if os.path.exists(cxr_jpg_folder):
                # Find all study folders (they start with 's')
                study_folders = [f for f in os.listdir(cxr_jpg_folder) if os.path.isdir(os.path.join(cxr_jpg_folder, f)) and f.startswith('s')]
                
                # Process each report file and find matching study folder
                for report_file in report_files:
                    report_path = os.path.join(cxr_dicom_folder, report_file)
                    study_id = report_file.replace('.txt', '')  # e.g., 's12345678'
                    
                    # Find the matching study folder
                    matching_study_folder = next((folder for folder in study_folders if folder == study_id), None)
                    
                    if matching_study_folder:
                        study_path = os.path.join(cxr_jpg_folder, matching_study_folder)
                        # Find all jpg files in the study folder
                        jpg_files = [f for f in os.listdir(study_path) if f.endswith('.jpg')]
                        
                        # Process each image in the study
                        for jpg_file in jpg_files:
                            image_path = os.path.join(study_path, jpg_file)
                            
                            # Set transcript paths based on transcript type
                            if transcript_type == 'REFLACX' and reflacx_ids:
                                # For each image, create a row for each REFLACX "id" found in metadata.csv
                                for p_id in reflacx_ids:
                                    transcript_path = os.path.join(reflacx_folder, 'main_data', p_id, 'transcription.txt')
                                    # Validate transcript path; set to None if it doesn’t exist
                                    if not os.path.exists(transcript_path):
                                        transcript_path = None
                                    # Append data to lists
                                    image_paths.append(image_path)
                                    report_paths.append(report_path)
                                    patient_ids.append(patient_id)
                                    transcript_paths.append(transcript_path)
                            elif transcript_type == 'EyeGaze':
                                # Extract DICOM ID for EyeGaze (unchanged)
                                dicom_id = os.path.splitext(jpg_file)[0]
                                transcript_path = os.path.join(eyegaze_folder, 'audio_segmentation_transcripts', dicom_id, 'transcript.json')
                                # Validate transcript path; set to None if it doesn’t exist
                                if not os.path.exists(transcript_path):
                                    transcript_path = None
                                # Append data to lists
                                image_paths.append(image_path)
                                report_paths.append(report_path)
                                patient_ids.append(patient_id)
                                transcript_paths.append(transcript_path)
                            else:
                                # No transcripts available
                                image_paths.append(image_path)
                                report_paths.append(report_path)
                                patient_ids.append(patient_id)
                                transcript_paths.append(None)
        pbar.update(1)

# Create dataframe
crosstrain_df = pd.DataFrame({
    'id': patient_ids,
    'report_path': report_paths,
    'image_path': image_paths,
    'transcript_path': transcript_paths
})

# Drop rows with no transcript_path
crosstrain_df = crosstrain_df.dropna(subset=['transcript_path'])

# ---- Add conditions column from condition_labels.csv ----
try:
    # Load condition_labels.csv
    cond_df = pd.read_csv('condition_labels.csv')
    # Ensure both id columns are string type for merge
    crosstrain_df['id'] = crosstrain_df['id'].astype(str)
    cond_df['id'] = cond_df['id'].astype(str)
    if 'id' in cond_df.columns and 'conditions' in cond_df.columns:
        # Merge on 'id' to add 'conditions' column to crosstrain_df
        crosstrain_df = pd.merge(
            crosstrain_df,
            cond_df[['id', 'conditions']],
            on='id',
            how='left'
        )
    else:
        print("Warning: condition_labels.csv does not have required columns ['id', 'conditions'].")
except Exception as e:
    print(f"Error loading or merging condition_labels.csv: {e}")
# ---- End conditions column addition ----

# Save to CSV (with conditions column)
output_path = os.path.join("./", "trans-data.csv")
crosstrain_df.to_csv(output_path, index=False)