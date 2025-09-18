# Dataset Setup Instructions

## Overview

This document provides instructions for setting up the required datasets for the Medical Report Generation and Disease Classification system.

## Dataset Setup Sequence

Follow these steps in the exact order specified:

### 1. Download Official MIMIC-Eye Dataset

**Location:** Root folder

```
.\mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0\mimic-eye
```

**Description:** Main dataset containing chest X-rays with eye-gaze data, bounding boxes, and transcripts.

Download the official MIMIC-Eye dataset and place it in the root folder of this codebase.

### 2. Generate Initial Dataset

**Run:** `datasetcode.ipynb`

Execute the Jupyter notebook `datasetcode.ipynb` located in the root folder to generate the initial dataset file `final_dataset.csv`.

### 3. Create Final Dataset

**Run:** `dataset_final.py`

Execute the script `dataset_final.py` located in the root folder to process `final_dataset.csv` and generate `final_dataset_fixed.csv`.

### 4. Generate Dataset Splits

**Run:** `dataset_splitting.py`

Execute the script `dataset_splitting.py` located in the root folder to create the dataset splits containing:
- `train.csv`
- `test.csv` 
- `val.csv`

These will be placed in the `dataset_splits/` directory.

### 5. Generate Preprocessed Data

**Run:** `scripts/preprocess.py`

Execute the preprocessing script to generate the required data structures:

**Location:** `.\data_dump\output\` (root folder)

**Contents:**
- `bbox_mask/` - Bounding box mask data
- `fix_seq/` - Fixation sequence data
- `img_png/` - Processed PNG images

**Also generates:** `.\fixstats.npz` - Fixation statistics file

### 6. Ground Truth Reports

**Location:** `.\cleaned_reports\` (root folder)

**Note:** The `cleaned_reports/` directory is already provided in this codebase and contains ground truth medical reports for evaluation purposes.

## Final Directory Structure

After completing all steps, your directory structure should look like:

```
Root/
├── dataset_splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── data_dump/
│   └── output/
│       ├── bbox_mask/
│       ├── fix_seq/
│       └── img_png/
├── cleaned_reports/
├── final_dataset_fixed.csv
├── fixstats.npz
├── datasetcode.ipynb
├── dataset_final.py
├── dataset_splitting.py
└── scripts/
    └── preprocess.py
```
