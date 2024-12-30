import os
import json
import pandas as pd

from data.paths import TabularDataPaths, MIMIC_EYE_PATH
from utils import print as print_f
from utils.preprocessing import get_split_list, checkMissingValue
from utils.init import reproducibility
from data.constants import REFLACX_ALL_LABEL_COLS, REFLACX_REPETITIVE_ALL_LABEL_COLS

def check_and_print_file(file_path, description):
    """Check if a file exists and print its contents"""
    print(f"\nChecking {description}:")
    print(f"Path: {file_path}")
    
    if os.path.exists(file_path):
        print("File exists!")
        # try:
        #     if file_path.endswith('.json'):
        #         with open(file_path, 'r') as f:
        #             content = json.load(f)
        #             print("Content (JSON):", json.dumps(content, indent=2))
        #     elif file_path.endswith('.csv'):
        #         df = pd.read_csv(file_path)
        #         print("Content (CSV head):\n", df.head())
        #     elif file_path.endswith('.txt'):
        #         with open(file_path, 'r') as f:
        #             content = f.read()
        #             print("Content (TXT):", content)
        #     else:
        #         print("File exists but not reading content (might be binary file)")
        # except Exception as e:
        #     print(f"Error reading file: {str(e)}")
    else:
        print("File does not exist!")

def main():
    # Set XAMI_MIMIC_PATH to the MIMIC-Eye dataset directory
    XAMI_MIMIC_PATH = os.path.join(os.getcwd(), "mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0", "mimic-eye")
    print(f"Using MIMIC-Eye path: {XAMI_MIMIC_PATH}")

    # Check each path
    paths_to_check = [
        {
            "path": os.path.join(XAMI_MIMIC_PATH, "patient_18522520", "CXR-JPG", "s59022382", 
                                "887ce638-036e6a6b-15d37e27-30515374-f78b68e9.jpg"),
            "desc": "image_path"
        },
        {
            "path": os.path.join(XAMI_MIMIC_PATH, "patient_18111516", "REFLACX", "main_data", 
                                "P102R765317", "anomaly_location_ellipses.csv"),
            "desc": "bbox_path"
        },
        {
            "path": os.path.join(XAMI_MIMIC_PATH, "patient_19346228", "REFLACX", "main_data", 
                                "P117R704578", "fixations.csv"),
            "desc": "fixation_path"
        },
        # {
        #     "path": "physionet.org/files/mimic-cxr/2.0.0/files/p15/p15276846/s57959166/85c363f0-f551672b-c20e32d9-0d7c5522-cc076373.dcm",
        #     "desc": "image path from physionet"
        # },
        {
            "path": os.path.join("mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0",
                                "mimic-eye", "patient_18522520", "CXR-DICOM", "s59022382.txt"),
            "desc": "report_path"
        },
        {
            "path": os.path.join("mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0",
                                "mimic-eye", "patient_14883411", "REFLACX", "main_data", 
                                "P211R205436", "timestamps_transcription.csv"),
            "desc": "reflacx_transcript_path"
        },
        {
            "path": os.path.join("mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0",
                                "mimic-eye", "patient_19346228", "EyeGaze", "audio_segmentation_transcripts",
                                "30ea0dcf-3173854a-fd22f3c5-dea153b8-aff23105", "transcript.json"),
            "desc": "eyegaze_transcript_path"
        }
    ]

    # Check each path
    for path_info in paths_to_check:
        check_and_print_file(path_info["path"], path_info["desc"])

if __name__ == "__main__":
    main()