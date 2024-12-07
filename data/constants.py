import numpy as np

from typing import Dict, List

"""
MIMIC
"""
DEFAULT_MIMIC_CLINICAL_NUM_COLS: List[str] = [
    "age",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    # "pain",
    "acuity",
]

DEFAULT_MIMIC_CLINICAL_CAT_COLS: List[str] = ["gender"]

"""
REFLACX Dataset
"""
REFLACX_ALL_LABEL_COLS = [
    "Airway wall thickening",
    "Atelectasis",
    "Consolidation",
    "Enlarged cardiac silhouette",
    "Fibrosis",
    "Fracture",
    "Groundglass opacity",
    "Pneumothorax",
    "Pulmonary edema",
    "Wide mediastinum",
    "Abnormal mediastinal contour",
    "Acute fracture",
    "Enlarged hilum",
    "Quality issue",
    "Support devices",
    "Hiatal hernia",
    "High lung volume / emphysema",
    "Interstitial lung disease",
    "Lung nodule or mass",
    "Pleural abnormality",
    # "report_path",
    # "reflacx_transcript_path",
    # "eyegaze_transcript_path",
]

REFLACX_REPETITIVE_ALL_LABEL_COLS = [
    "Airway wall thickening",
    "Atelectasis",
    "Consolidation",
    "Emphysema",
    "Enlarged cardiac silhouette",
    "Fibrosis",
    "Fracture",
    "Groundglass opacity",
    "Mass",
    "Nodule",
    "Other",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary edema",
    "Wide mediastinum",
    "Abnormal mediastinal contour",
    "Acute fracture",
    "Enlarged hilum",
    "Hiatal hernia",
    "High lung volume / emphysema",
    "Interstitial lung disease",
    "Lung nodule or mass",
    "Pleural abnormality",
]

DEFAULT_REFLACX_LABEL_COLS: List[str] = [
    # "Fibrosis",
    # "Quality issue",
    # "Wide mediastinum",
    # "Fracture",
    # "Airway wall thickening",

    ######################
    # "Hiatal hernia",
    # "Acute fracture",
    # "Interstitial lung disease",
    # "Enlarged hilum",
    # "Abnormal mediastinal contour",
    # "High lung volume / emphysema",
    # "Pneumothorax",
    # "Lung nodule or mass",
    # "Groundglass opacity",
    ######################
    "Pulmonary edema",
    "Enlarged cardiac silhouette",
    "Consolidation",
    "Atelectasis",
    "Pleural abnormality",
    # "Support devices",
]

# [
#     "Enlarged cardiac silhouette",
#     "Atelectasis",
#     "Pleural abnormality",
#     "Consolidation",
#     "Pulmonary edema",
#     #  'Groundglass opacity', #6th disease.
# ]

# DEFAULT_REFLACX_ALL_DISEASES: List[str] = [
#     "Airway wall thickening",
#     "Atelectasis",
#     "Consolidation",
#     "Enlarged cardiac silhouette",
#     "Fibrosis",
#     "Groundglass opacity",
#     "Pneumothorax",
#     "Pulmonary edema",
#     "Wide mediastinum",
#     "Abnormal mediastinal contour",
#     "Acute fracture",
#     "Enlarged hilum",
#     "Hiatal hernia",
#     "High lung volume / emphysema",
#     "Interstitial lung disease",
#     "Lung nodule or mass",
#     "Pleural abnormality",
# ]

DEFAULT_REFLACX_REPETITIVE_LABEL_MAP: Dict[str, List[str]] = {
    "Airway wall thickening": ["Airway wall thickening"],
    "Atelectasis": ["Atelectasis"],
    "Consolidation": ["Consolidation"],
    "Enlarged cardiac silhouette": ["Enlarged cardiac silhouette"],
    "Fibrosis": ["Fibrosis"],
    "Groundglass opacity": ["Groundglass opacity"],
    "Pneumothorax": ["Pneumothorax"],
    "Pulmonary edema": ["Pulmonary edema"],
    "Quality issue": ["Quality issue"],
    "Support devices": ["Support devices"],
    "Wide mediastinum": ["Wide mediastinum"],
    "Abnormal mediastinal contour": ["Abnormal mediastinal contour"],
    "Acute fracture": ["Acute fracture"],
    "Enlarged hilum": ["Enlarged hilum"],
    "Hiatal hernia": ["Hiatal hernia"],
    "High lung volume / emphysema": ["High lung volume / emphysema", "Emphysema",],
    "Interstitial lung disease": ["Interstitial lung disease"],
    "Lung nodule or mass": ["Lung nodule or mass", "Mass", "Nodule"],
    "Pleural abnormality": [
        "Pleural abnormality",
        "Pleural thickening",
        "Pleural effusion",
    ],
}

DEFAULT_REFLACX_BOX_COORD_COLS: List[str] = ["xmin", "ymin", "xmax", "ymax"]
DEFAULT_REFLACX_BOX_FIX_COLS: List[str] = DEFAULT_REFLACX_BOX_COORD_COLS + ["certainty"]
DEFAULT_REFLACX_PATH_COLS: List[str] = [
    "image_path",
    "fixation_path",
    "bbox_path",
]


"""
IOU values during evaluation.
"""


FULL_IOU_THRS = np.array(
    [
        0.00,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    ]
)

IOU_THRS_5_TO_95 = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

