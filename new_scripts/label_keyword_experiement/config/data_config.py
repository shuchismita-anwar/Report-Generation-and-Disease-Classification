"""
Data Configuration for Hybrid RadBERT + MedCLIP Label & Keyword Prediction
==========================================================================

This file contains all data processing related configuration parameters.
"""

import os

# Dataset Paths Configuration
DATASET_PATHS = {
    "train_csv": "../../dataset_splits/train.csv",
    "val_csv": "../../dataset_splits/val.csv", 
    "test_csv": "../../dataset_splits/test.csv",
    "fixstats_npz": "../../fixstats.npz",
}

# Data Directories Configuration
DATA_DIRS = {
    "img_dir": os.path.join("..", "..", "data_dump", "output", "img_png"),
    "bbox_dir": os.path.join("..", "..", "data_dump", "output", "bbox_mask"),
    "fixseq_dir": os.path.join("..", "..", "data_dump", "output", "fix_seq"),
}

# Image Processing Configuration
IMAGE_CONFIG = {
    "image_size": 224,
    "resize_size": 256,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "interpolation": "bilinear",
    "format": "RGB",
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    "train_augmentations": {
        "random_resized_crop": {
            "size": 224,
            "scale": (0.85, 1.0),
            "ratio": (0.9, 1.1),
        },
        "random_horizontal_flip": {"p": 0.5},
        "random_rotation": {"degrees": 5},
        "color_jitter": {
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.0,
        },
        "random_affine": {
            "degrees": 0,
            "translate": (0.05, 0.05),
        },
        "random_erasing": {
            "p": 0.1,
            "scale": (0.02, 0.1),
        },
    },
    "val_test_augmentations": {
        "resize": {"size": 224},
        "center_crop": {"size": 224},
    },
}

# Bbox/Mask Processing Configuration
BBOX_CONFIG = {
    "mask_size": 224,
    "interpolation": "nearest",
    "format": "L",  # Grayscale
    "threshold": 0.5,  # For binary masks
}

# Fixation Sequence Configuration
FIXATION_CONFIG = {
    "max_sequence_length": 128,
    "feature_dim": 4,  # [x, y, duration, saccade_amplitude]
    "normalization_method": "z_score",  # Options: "z_score", "min_max", "none"
    "padding_value": 0.0,
    "clamp_range": (-10.0, 10.0),
    "handle_nan": "zero",  # Options: "zero", "mean", "drop"
}

# Transcript Processing Configuration
TRANSCRIPT_CONFIG = {
    "max_length": 512,
    "encoding": "utf-8",
    "preprocessing": {
        "lowercase": True,
        "remove_special_chars": False,
        "remove_numbers": False,
        "remove_stopwords": False,
    },
    "tokenization": {
        "add_special_tokens": True,
        "padding": "max_length",
        "truncation": True,
        "return_attention_mask": True,
    },
}

# Keyword Extraction Configuration
KEYWORD_EXTRACTION_CONFIG = {
    "extraction_methods": ["regex", "spacy", "radbert"],  # Will start with basic methods
    "medical_patterns": [
        r'\b(?:atelectasis|collapse|volume loss)\b',
        r'\b(?:cardiomegaly|enlarged heart|cardiac enlargement)\b',
        r'\b(?:edema|fluid|swelling)\b',
        r'\b(?:opacity|consolidation|infiltrate)\b',
        r'\b(?:effusion|fluid collection)\b',
        r'\b(?:pneumonia|infection|inflammatory)\b',
        r'\b(?:device|tube|catheter|pacemaker)\b',
    ],
    "min_keyword_length": 3,
    "max_keyword_length": 50,
    "filter_stopwords": True,
    "case_sensitive": False,
}

# Vocabulary Building Configuration
VOCABULARY_CONFIG = {
    "min_frequency": 3,
    "max_vocab_size": 5000,
    "special_tokens": ["<PAD>", "<UNK>", "<START>", "<END>"],
    "include_medical_terms": True,
    "vocabulary_file": "output/vocabulary/medical_vocab.json",
    "word2idx_file": "output/vocabulary/word2idx.json",
    "idx2word_file": "output/vocabulary/idx2word.json",
}

# Dataset Filtering Configuration
FILTERING_CONFIG = {
    "require_bbox": True,
    "require_fixations": True,
    "require_transcript": True,
    "min_transcript_length": 10,
    "max_transcript_length": 2000,
    "valid_conditions_only": True,
    "remove_duplicates": True,
}

# Data Loading Configuration
DATALOADER_CONFIG = {
    "batch_size": 32,
    "num_workers": 2,
    "pin_memory": True,
    "drop_last": True,
    "shuffle_train": True,
    "shuffle_val": False,
    "shuffle_test": False,
    "persistent_workers": True,
    "prefetch_factor": 2,
}

# Memory Optimization Configuration
MEMORY_CONFIG = {
    "low_memory_mode": False,  # Will be set based on system detection
    "low_memory_batch_size": 8,
    "low_memory_num_workers": 0,
    "gradient_accumulation_steps": 6,
    "max_memory_usage_gb": 8.0,
    "enable_memory_monitoring": True,
}

# Data Quality Configuration
QUALITY_CONFIG = {
    "check_file_existence": True,
    "validate_image_format": True,
    "validate_mask_format": True,
    "validate_fixation_format": True,
    "validate_transcript_format": True,
    "log_missing_files": True,
    "skip_corrupted_samples": True,
}

# Output Configuration
OUTPUT_CONFIG = {
    "save_processed_data": True,
    "processed_data_dir": "output/processed_data",
    "save_statistics": True,
    "statistics_file": "output/data_statistics.json",
    "save_vocabulary": True,
    "save_keyword_mappings": True,
}

# Region Availability Configuration
REGION_AVAILABILITY_CONFIG = {
    "eyegaze_regions": 17,  # EyeGaze data has all 17 regions
    "reflacx_regions": "variable",  # REFLACX regions are predicted by YOLO
    "imputation_strategy": "zero_mask",  # How to handle missing regions
    "region_confidence_threshold": 0.5,  # For YOLO predictions
    "log_region_statistics": True,
}

# Export all configurations
__all__ = [
    'DATASET_PATHS',
    'DATA_DIRS',
    'IMAGE_CONFIG',
    'AUGMENTATION_CONFIG',
    'BBOX_CONFIG',
    'FIXATION_CONFIG',
    'TRANSCRIPT_CONFIG',
    'KEYWORD_EXTRACTION_CONFIG',
    'VOCABULARY_CONFIG',
    'FILTERING_CONFIG',
    'DATALOADER_CONFIG',
    'MEMORY_CONFIG',
    'QUALITY_CONFIG',
    'OUTPUT_CONFIG',
    'REGION_AVAILABILITY_CONFIG'
] 