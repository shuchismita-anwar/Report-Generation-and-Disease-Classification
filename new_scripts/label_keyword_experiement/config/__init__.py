"""
Configuration Package for Hybrid RadBERT + MedCLIP Label & Keyword Prediction
==============================================================================

This package contains all configuration files for the experiment.
"""

# Import all configuration modules
from . import model_config
from . import data_config  
from . import training_config

# Make key configurations easily accessible
from .model_config import (
    VISION_MODEL, TEXT_MODEL, RADBERT_MODEL, MEDCLIP_MODEL,
    DISEASE_CONFIG, KEYWORD_CONFIG, FUSION_CONFIG, REGION_CONFIG
)

from .data_config import (
    DATASET_PATHS, DATA_DIRS, IMAGE_CONFIG, AUGMENTATION_CONFIG,
    VOCABULARY_CONFIG, DATALOADER_CONFIG
)

from .training_config import (
    TRAINING_CONFIG, BATCH_CONFIG, OPTIMIZER_CONFIG, SCHEDULER_CONFIG,
    LOSS_CONFIG, EARLY_STOPPING_CONFIG
)

__version__ = "0.1.0"
__all__ = [
    'model_config', 'data_config', 'training_config',
    'VISION_MODEL', 'TEXT_MODEL', 'RADBERT_MODEL', 'MEDCLIP_MODEL',
    'DISEASE_CONFIG', 'KEYWORD_CONFIG', 'FUSION_CONFIG', 'REGION_CONFIG',
    'DATASET_PATHS', 'DATA_DIRS', 'IMAGE_CONFIG', 'AUGMENTATION_CONFIG',
    'VOCABULARY_CONFIG', 'DATALOADER_CONFIG',
    'TRAINING_CONFIG', 'BATCH_CONFIG', 'OPTIMIZER_CONFIG', 'SCHEDULER_CONFIG',
    'LOSS_CONFIG', 'EARLY_STOPPING_CONFIG'
] 