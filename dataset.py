import os
import pandas as pd
import re
import glob
from data.paths import MIMIC_EYE_PATH
from utils import print as print_f

# Set the path to MIMIC-EYE dataset
mimic_eye_path = MIMIC_EYE_PATH

# Print the MIMIC_EYE_PATH
print_f.print_title("MIMIC-EYE Path")
print(f"MIMIC-EYE Path: {mimic_eye_path}")

# Find all patient directories
print_f.print_title("Finding patient directories")
patient_pattern = re.compile(r'^patient_(\d{8})$')
patient_dirs = []

for item in os.listdir(mimic_eye_path):
    item_path = os.path.join(mimic_eye_path, item)
    match = patient_pattern.match(item)
    if os.path.isdir(item_path) and match:
        patient_dirs.append((item_path, match.group(1)))

print(f"Found {len(patient_dirs)} patient directories")

# Initialize lists to store paths
image_paths = []
report_paths = []
patient_ids = []

# Process each patient directory
print_f.print_title("Processing patient directories")
for patient_dir, patient_id in patient_dirs:
    patient_folder = os.path.basename(patient_dir)
    print(f"Processing patient {patient_folder} with ID {patient_id}")
    
    # Look for report files in CXR-DICOM folder
    cxr_dicom_folder = os.path.join(patient_dir, 'CXR-DICOM')
    if os.path.exists(cxr_dicom_folder):
        report_files = [f for f in os.listdir(cxr_dicom_folder) if f.endswith('.txt')]
        
        # Look for image files in CXR-JPG folder
        cxr_jpg_folder = os.path.join(patient_dir, 'CXR-JPG')
        if os.path.exists(cxr_jpg_folder):
            # Find all study folders (they start with 's')
            study_folders = []
            for item in os.listdir(cxr_jpg_folder):
                if os.path.isdir(os.path.join(cxr_jpg_folder, item)) and item.startswith('s'):
                    study_folders.append(item)
            
            # Process each report file and find matching study folder
            for report_file in report_files:
                report_path = os.path.join(cxr_dicom_folder, report_file)
                study_id = report_file.replace('.txt', '')
                
                # Find the matching study folder
                matching_study_folder = None
                for folder in study_folders:
                    if folder == study_id:
                        matching_study_folder = folder
                        break
                
                if matching_study_folder:
                    study_path = os.path.join(cxr_jpg_folder, matching_study_folder)
                    # Find all jpg files in the study folder
                    jpg_files = [f for f in os.listdir(study_path) if f.endswith('.jpg')]
                    
                    # Add each image path with the corresponding report path and patient ID
                    for jpg_file in jpg_files:
                        image_path = os.path.join(study_path, jpg_file)
                        image_paths.append(image_path)
                        report_paths.append(report_path)
                        patient_ids.append(patient_id)

# Create dataframe
print_f.print_title("Creating dataframe")
crosstrain_df = pd.DataFrame({
    'id': patient_ids,
    'report_path': report_paths,
    'image_path': image_paths
})

print(f"Found {len(crosstrain_df)} pairs of report and image paths")

# Save to CSV in the root folder
output_path = os.path.join("./", "crosstrain.csv")
crosstrain_df.to_csv(output_path, index=False)

# Print some basic statistics
print_f.print_title("Dataset Statistics")
print(f"Total number of unique patients: {len(crosstrain_df['id'].unique())}")
print(f"Total number of instances: {len(crosstrain_df)}")
print(f"Dataset saved to: {output_path}")

# Print sample of the final dataframe
print_f.print_title("Sample of final dataframe")
print(crosstrain_df.head())