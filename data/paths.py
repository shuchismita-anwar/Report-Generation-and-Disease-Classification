import os
import pandas as pd
from enum import Enum
from typing import Union

'''
Data Paths
'''

# MIMIC_EYE_PATH = "/Users/jrhs/Desktop/mimic-eye"
MIMIC_EYE_PATH = "mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0\mimic-eye"
PHYSIONET_PATH = "E:\\physionet.org"
SPREADSHEET_FOLDER = "spreadsheets"


class PhysioFileReader():

    def __init__(self, physio_path: str,) -> None:
        self.physio_path = physio_path
        self.files_path = os.path.join(physio_path, "files")


class ReflacxFiles(Enum):
    Metadata= "metadata_phase"

    BoundingEllipse = "anomaly_location_ellipses.csv"
    Fixations = "fixations.csv"
    ChestBoundingBox = "chest_bounding_box.csv"
    TimestampsTranscription = "timestamps_transcription.csv"
    Transcription = "transcription.txt" 

class ReflacxReader():
    
    def __init__(self, files_path: str, version="1.0.0") -> None:

        self.version = version
        self.folder_name = "reflacx-xray-localization"
        self.folder_path = os.path.join(files_path, self.folder_name, self.version)

        '''
        Each file name here
        '''
        self.metadata_file_name = "metadata_phase"
        self.bounding_ellipses  = "anomaly_location_ellipses.csv"
        self.fixations = "fixations.csv"
        self.chest_bounding_box = "chest_bounding_box.csv"
        self.folders = {
            "main": "main_data",
            "gaze": "gaze_data"
        }

    def get_metadata(self,):
        return  pd.concat([ pd.read_csv( os.path.join(self.get_studies_path()  ,f"{self.metadata_file_name}_{i}")) for i in [1, 2, 3]], axis=1)
    
    def get_studies_path(self,):
        return os.path.join(self.folder_path, self.folders['main'])

    def get_study_data(self, reflacx_id: str, file: ReflacxFiles) -> Union[pd.DataFrame, str]:

        study_folder_path = os.path.join(self.get_studies_path(), reflacx_id) 
        f_path = os.path.join(study_folder_path, file.value)

        if str(f_path.endswith(".csv")):
            data= pd.read_csv( f_path)

        elif str(f_path.endswith(".txt")):
            with open(os.path.join()) as f:
                data = f.read(f_path)
        else:
            raise NotImplementedError(f"{str(file)} is not a format implemented. If you're trying to access metadata, using `get_metadata` instead.")

        return data


class MIMIC_CXR_JPG_Reader():
    def __init__(self, files_path: str, version="2.0.0") -> None:

        self.version = version
        self.folder_name = "mimic-cxr-jpg"
        self.folder_path = os.path.join(files_path, self.folder_name, self.version)


        '''
        Each file name here
        '''
        self.metadata_file_name = "mimic-cxr-2.0.0-metadata.csv.gz"

        self.bounding_ellipses  = "anomaly_location_ellipses.csv"
        self.fixations = "fixations.csv"
        self.chest_bounding_box = "chest_bounding_box.csv"
        self.folders = {
            "files": "files"
        }

    def get_meta_data(self,):
        return pd.read_csv(os.path.join(self.folder_path, self.metadata_file_name))

# class MIMIC_ED():

# class MIMIC_IV():

# class EyeGaze():


class TabularDataPaths():
    
    class SpreadSheet():

        def get_sreadsheet(mimic_folder_path, path):
            return os.path.join(mimic_folder_path, path)

        root_path = "spreadsheets"
        cxr_meta = os.path.join(root_path, "cxr_meta.csv")
        cxr_meta_with_stay_id_only = os.path.join(
            root_path, "cxr_meta_with_stay_id_only.csv")

        class CXR_JPG():
            root_path = os.path.join("spreadsheets", "CXR-JPG")
            cxr_chexpert = os.path.join(root_path, "cxr_chexpert.csv")
            cxr_negbio = os.path.join(root_path, "cxr_negbio.csv")
            cxr_split = os.path.join(root_path, "cxr_split.csv")

        class EyeGaze():
            root_path = os.path.join("spreadsheets", "EyeGaze")
            bounding_boxes = os.path.join(root_path, "bounding_boxes.csv")
            fixations = os.path.join(root_path, "fixations.csv")
            master_sheet_with_updated_stayId = os.path.join(
                root_path, "master_sheet_with_updated_stayId.csv")

        class REFLACX():
            root_path = os.path.join("spreadsheets", "REFLACX")
            metadata = os.path.join(root_path, "metadata.csv")

    class PatientDataPaths():

        def get_patient_path(mimic_folder_path, patient_id, path):
            return os.path.join(mimic_folder_path, f"patient_{patient_id}", path)

        class Core():
            root_path = "Core"
            admissions = os.path.join(root_path, "admissions.csv")
            patients = os.path.join(root_path, "patients.csv")
            transfers = os.path.join(root_path, "transfers.csv")

        class CXR_DICOM():
            root_path = "CXR-DICOM"

        class CXR_JPG():
            root_path = "CXR-JPG"
            cxr_chexpert = os.path.join(root_path, "cxr_chexpert.csv")
            cxr_meta = os.path.join(root_path, "cxr_meta.csv")
            cxr_negbio = os.path.join(root_path, "cxr_negbio.csv")
            cxr_split = os.path.join(root_path, "cxr_split.csv")

        class ED():
            root_path = "ED"
            diagnosis = os.path.join(root_path, "diagnosis.csv")
            edstays = os.path.join(root_path, "edstays.csv")
            medrecon = os.path.join(root_path, "medrecon.csv")
            pyxis = os.path.join(root_path, "pyxis.csv")
            triage = os.path.join(root_path, "triage.csv")

        class REFLACX():

            root_path = "REFLACX"
            metadata = os.path.join(root_path, "metadata.csv")

            class REFLACXStudy(Enum):
                anomaly_location_ellipses = "anomaly_location_ellipses.csv"
                chest_bounding_box = "chest_bounding_box.csv"
                fixations = "fixations.csv"
                timestamps_transcription = "timestamps_transcription.csv"
                transcription = "transcription.csv"

                def get_reflacx_path(mimic_folder_path, patient_id, reflacx_id, path):
                    return os.path.join(mimic_folder_path, f"patient_{patient_id}", "REFLACX", reflacx_id, path)




