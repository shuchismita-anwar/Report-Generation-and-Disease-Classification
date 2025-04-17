import os
import pandas as pd
import glob

# List of conditions to check
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]

def process_patient_folder(patient_folder, trans_df):
    """Process a single patient folder and return (patient_id, study_id, conditions)"""
    # Extract patient ID from folder name
    patient_id = patient_folder.split('patient_')[-1]
    
    # Path to cxr_chexpert.csv
    chexpert_path = os.path.join(patient_folder, 'CXR-JPG', 'cxr_chexpert.csv')
    
    if not os.path.exists(chexpert_path):
        print(f"No cxr_chexpert.csv found for patient {patient_id}")
        return None
    
    # Find study folder (sXXXXXXXX)
    study_folders = glob.glob(os.path.join(patient_folder, 'CXR-JPG', 's[0-9]*'))
    if not study_folders:
        print(f"No study folder found for patient {patient_id}")
        return None
    
    # Extract study ID from folder name
    study_id = os.path.basename(study_folders[0])[1:]  # Remove 's' prefix
    
    try:
        # Read the CheXpert CSV
        df = pd.read_csv(chexpert_path)
        
        # Find matching study
        study_row = df[df['study_id'] == int(study_id)]
        if study_row.empty:
            print(f"No matching study_id {study_id} found for patient {patient_id}")
            return None
            
        # Get conditions that are 1.0
        present_conditions = []
        for condition in CONDITIONS:
            if study_row[condition].iloc[0] == 1.0:
                present_conditions.append(condition)
        
        # Get transcript_path from trans-data.csv
        transcript_path = trans_df[trans_df['transcript_path'].str.contains(f'patient_{patient_id}')]['transcript_path'].iloc[0]
                
        return {
            'id': patient_id,
            'transcript_path': transcript_path,
            'conditions': '|'.join(present_conditions) if present_conditions else 'No Finding'
        }
        
    except Exception as e:
        print(f"Error processing patient {patient_id}: {str(e)}")
        return None

def main():
    base_dir = 'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0\mimic-eye'
    
    # Load transcript data
    trans_df = pd.read_csv('trans-data.csv')
    
    # Get all patient folders
    patient_folders = glob.glob(os.path.join(base_dir, 'patient_*'))
    
    # Process each patient
    results = []
    for folder in patient_folders:
        result = process_patient_folder(folder, trans_df)
        if result:
            results.append(result)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv('condition_labels.csv', index=False)
    print(f"Processed {len(results)} patients successfully")

if __name__ == "__main__":
    main()