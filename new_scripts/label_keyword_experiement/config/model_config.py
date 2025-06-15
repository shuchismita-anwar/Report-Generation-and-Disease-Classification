"""
Model Configuration for Hybrid RadBERT + MedCLIP Label & Keyword Prediction
===========================================================================

This file contains all model-related configuration parameters.
"""

# Vision Model Configuration
VISION_MODEL = {
    "model_name": "google/vit-base-patch16-224-in21k",
    "image_size": 224,
    "patch_size": 16,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "dropout": 0.1,
    "attention_dropout": 0.1,
}

# Text Model Configuration
TEXT_MODEL = {
    "model_name": "emilyalsentzer/Bio_ClinicalBERT",
    "hidden_size": 768,
    "max_length": 512,
    "dropout": 0.1,
}

# RadBERT Configuration (for keyword extraction)
RADBERT_MODEL = {
    "model_name": "emilyalsentzer/Bio_ClinicalBERT",  # Placeholder - will be updated with actual RadBERT
    "max_length": 512,
    "batch_size": 8,
    "extraction_method": "entity_extraction",
}

# MedCLIP Configuration (for contrastive learning)
MEDCLIP_MODEL = {
    "model_name": "flaviagiammarino/pubmed-clip-vit-base-patch32",  # Placeholder - will be updated with actual MedCLIP
    "temperature": 0.07,
    "embedding_dim": 512,
    "projection_dim": 256,
}

# Disease Classification Configuration
DISEASE_CONFIG = {
    "num_conditions": 8,
    "conditions": [
        'Atelectasis', 'Cardiomegaly', 'Edema',
        'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pneumonia', 'Support Devices'
    ],
    "loss_type": "focal",  # Options: "focal", "asymmetric", "bce"
    "focal_alpha": 1.10,
    "focal_gamma": 2.15,
    "label_smoothing": 0.1,
}

# Keyword Prediction Configuration
KEYWORD_CONFIG = {
    "vocab_size": 5000,  # Initial size - will be determined from data
    "max_keywords_per_sample": 50,
    "min_keyword_frequency": 3,
    "global_keyword_head_dim": 768,
    "region_keyword_head_dim": 768,
    "num_regions": 17,
    "keyword_loss_weight": 0.3,
    "region_keyword_loss_weight": 0.2,
}

# Multi-Modal Fusion Configuration
FUSION_CONFIG = {
    "fusion_method": "cross_attention",  # Options: "concatenation", "cross_attention", "bilinear"
    "cross_attention_heads": 8,
    "cross_attention_dropout": 0.1,
    "fusion_hidden_dim": 768,
    "contrastive_temperature": 0.1,
    "contrastive_weight": 0.1,
}

# Region Handling Configuration
REGION_CONFIG = {
    "num_regions": 17,
    "region_names": [
        "cardiac silhouette", "left clavicle", "left costophrenic angle",
        "left hilar structures", "left lower lung zone", "left lung",
        "left mid lung zone", "left upper lung zone", "right clavicle",
        "right costophrenic angle", "right hilar structures", "right lower lung zone",
        "right lung", "right mid lung zone", "right upper lung zone",
        "trachea", "upper mediastinum"
    ],
    "region_imputation_method": "zero_mask",  # Options: "zero_mask", "average_pooling", "learned_embedding"
    "region_attention_threshold": 0.1,
}

# Model Architecture Configuration
ARCHITECTURE_CONFIG = {
    "use_bbox_features": True,
    "use_fixation_features": True,
    "use_transcript_features": True,
    "use_keyword_prediction": True,
    "use_region_keyword_prediction": True,
    "use_contrastive_learning": True,
    "gradient_checkpointing": False,  # Will be enabled for memory optimization
}

# Loss Function Configuration
LOSS_CONFIG = {
    "disease_loss_weight": 1.0,
    "global_keyword_loss_weight": 0.3,
    "region_keyword_loss_weight": 0.2,
    "contrastive_loss_weight": 0.1,
    "total_loss_scaling": 1.0,
}

# Model Initialization Configuration
INIT_CONFIG = {
    "use_pretrained_weights": True,
    "chexpert_model_path": "../output/vit_chexpert_training_output/model_1.chexpert_finetune_only_feat_image/model.safetensors",
    "freeze_vision_backbone": False,
    "freeze_text_backbone": False,
    "init_method": "xavier_uniform",  # Options: "xavier_uniform", "kaiming_normal", "normal"
}

# Unified Model Configuration (for convenience)
MODEL_CONFIG = {
    'num_diseases': DISEASE_CONFIG['num_conditions'],
    'vocab_size': KEYWORD_CONFIG['vocab_size'],
    'num_regions': REGION_CONFIG['num_regions'],
    'hidden_dim': VISION_MODEL['hidden_size'],
    'dropout': VISION_MODEL['dropout'],
    'use_keywords': ARCHITECTURE_CONFIG['use_keyword_prediction'],
    'use_regions': ARCHITECTURE_CONFIG['use_region_keyword_prediction'],
    'disease_loss_weight': LOSS_CONFIG['disease_loss_weight'],
    'keyword_loss_weight': LOSS_CONFIG['global_keyword_loss_weight'],
    'region_loss_weight': LOSS_CONFIG['region_keyword_loss_weight'],
}

# Export all configurations
__all__ = [
    'VISION_MODEL',
    'TEXT_MODEL', 
    'RADBERT_MODEL',
    'MEDCLIP_MODEL',
    'DISEASE_CONFIG',
    'KEYWORD_CONFIG',
    'FUSION_CONFIG',
    'REGION_CONFIG',
    'ARCHITECTURE_CONFIG',
    'LOSS_CONFIG',
    'INIT_CONFIG',
    'MODEL_CONFIG'
] 