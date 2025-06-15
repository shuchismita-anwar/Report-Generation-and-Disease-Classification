"""
Training Configuration for Hybrid RadBERT + MedCLIP Label & Keyword Prediction
==============================================================================

This file contains all training-related configuration parameters.
"""

import os

# Basic Training Configuration
BASIC_TRAINING_CONFIG = {
    "num_epochs": 40,
    "learning_rate": 6e-6,
    "weight_decay": 8e-4,
    "warmup_steps": 175,
    "max_grad_norm": 1.0,
    "seed": 42,
}

# Batch Configuration
BATCH_CONFIG = {
    "batch_size": 32,
    "gradient_accumulation_steps": 6,
    "effective_batch_size": 192,  # batch_size * gradient_accumulation_steps
    "eval_batch_size": 16,
    "dataloader_num_workers": 2,
    "pin_memory": True,
}

# Memory Optimization Configuration
MEMORY_CONFIG = {
    "low_memory_batch_size": 8,
    "low_memory_grad_accum": 12,
    "low_memory_num_workers": 0,
    "gradient_checkpointing": False,  # Will be enabled if needed
    "fp16_training": True,  # Mixed precision training
    "dataloader_pin_memory": True,
    "empty_cache_steps": 100,  # Clear cache every N steps
}

# Optimizer Configuration
OPTIMIZER_CONFIG = {
    "optimizer_type": "adamw",  # Options: "adamw", "adam", "sgd"
    "learning_rate": 6e-6,
    "weight_decay": 8e-4,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "amsgrad": False,
}

# Learning Rate Scheduler Configuration
SCHEDULER_CONFIG = {
    "scheduler_type": "cosine_annealing",  # Options: "cosine_annealing", "linear", "polynomial", "constant"
    "warmup_steps": 175,
    "warmup_ratio": 0.1,
    "min_lr": 1e-7,
    "cosine_restarts": False,
    "polynomial_power": 1.0,
}

# Loss Function Configuration
LOSS_CONFIG = {
    "disease_loss_type": "focal",  # Options: "focal", "asymmetric", "bce", "weighted_bce"
    "focal_alpha": 1.10,
    "focal_gamma": 2.15,
    "label_smoothing": 0.1,
    "pos_weight_cap": 16.0,
    "pos_weight_smoothing": 0.07,
    
    # Multi-task loss weights
    "disease_loss_weight": 1.0,
    "global_keyword_loss_weight": 0.3,
    "region_keyword_loss_weight": 0.2,
    "contrastive_loss_weight": 0.1,
}

# Early Stopping Configuration
EARLY_STOPPING_CONFIG = {
    "patience": 10,
    "min_delta": 0.001,
    "monitor_metric": "val_f1",  # Options: "val_f1", "val_loss", "val_precision", "val_recall"
    "mode": "max",  # "max" for metrics to maximize, "min" for metrics to minimize
    "restore_best_weights": True,
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "eval_strategy": "steps",  # Options: "steps", "epoch", "no"
    "eval_steps": 100,
    "eval_accumulation_steps": None,
    "eval_delay": 0,
    "include_inputs_for_metrics": False,
    "metric_for_best_model": "eval_f1",
    "greater_is_better": True,
}

# Logging Configuration
LOGGING_CONFIG = {
    "logging_strategy": "steps",  # Options: "steps", "epoch", "no"
    "logging_steps": 50,
    "log_level": "info",  # Options: "debug", "info", "warning", "error"
    "log_on_each_node": True,
    "logging_nan_inf_filter": True,
    "report_to": ["tensorboard"],  # Options: ["tensorboard", "wandb", "mlflow", "none"]
}

# Checkpointing Configuration
CHECKPOINT_CONFIG = {
    "save_strategy": "steps",  # Options: "steps", "epoch", "no"
    "save_steps": 200,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "save_safetensors": True,
    "save_on_each_node": False,
    "resume_from_checkpoint": None,  # Will be set automatically if checkpoint exists
}

# Output Configuration
OUTPUT_CONFIG = {
    "output_dir": "output/training_mimic_on_chexpert_optimized",
    "overwrite_output_dir": True,
    "logging_dir": "output/logs",
    "run_name": "hybrid_radbert_medclip_training",
    "experiment_name": "label_keyword_prediction",
}

# Validation Configuration
VALIDATION_CONFIG = {
    "validation_split": 0.2,
    "stratify_split": True,
    "validation_metrics": [
        "accuracy", "precision", "recall", "f1", 
        "auc", "average_precision", "keyword_precision_at_k"
    ],
    "compute_metrics_each_epoch": True,
    "save_predictions": True,
    "prediction_output_dir": "output/predictions",
}

# Multi-Task Training Configuration
MULTITASK_CONFIG = {
    "task_weights": {
        "disease_classification": 1.0,
        "global_keyword_prediction": 0.3,
        "region_keyword_prediction": 0.2,
        "contrastive_learning": 0.1,
    },
    "task_sampling_strategy": "proportional",  # Options: "proportional", "uniform", "temperature"
    "temperature": 1.0,  # For temperature-based sampling
    "gradient_surgery": False,  # Advanced multi-task optimization
}

# Threshold Optimization Configuration
THRESHOLD_CONFIG = {
    "optimize_thresholds": True,
    "threshold_search_range": (0.38, 0.78),
    "threshold_search_steps": 20,
    "threshold_metric": "f1",  # Metric to optimize thresholds for
    "per_class_thresholds": True,
    "threshold_optimization_split": "validation",
}

# Advanced Training Configuration
ADVANCED_CONFIG = {
    "use_gradient_checkpointing": False,
    "dataloader_persistent_workers": True,
    "dataloader_prefetch_factor": 2,
    "remove_unused_columns": False,
    "label_names": ["labels"],
    "prediction_loss_only": False,
    "ignore_data_skip": False,
    "fsdp": "",  # Fully Sharded Data Parallel (for large models)
    "fsdp_config": {},
}

# Debugging Configuration
DEBUG_CONFIG = {
    "debug_mode": False,
    "overfit_batches": 0,  # Set to > 0 to overfit on small subset for debugging
    "fast_dev_run": False,  # Quick run for debugging
    "profile_memory": False,
    "detect_anomaly": False,  # PyTorch autograd anomaly detection
    "log_memory_usage": True,
}

# Reproducibility Configuration
REPRODUCIBILITY_CONFIG = {
    "seed": 42,
    "deterministic_training": True,
    "benchmark_cudnn": False,  # Set to True for performance, False for reproducibility
    "use_deterministic_algorithms": False,  # May slow down training
}

# Hardware Configuration
HARDWARE_CONFIG = {
    "device": "auto",  # Options: "auto", "cuda", "cpu"
    "mixed_precision": "fp16",  # Options: "no", "fp16", "bf16"
    "dataloader_num_workers": 2,
    "dataloader_pin_memory": True,
    "torch_compile": False,  # PyTorch 2.0 compilation (experimental)
}

# Unified Training Configuration (for convenience)
TRAINING_CONFIG = {
    'num_epochs': BASIC_TRAINING_CONFIG['num_epochs'],
    'learning_rate': BASIC_TRAINING_CONFIG['learning_rate'],
    'weight_decay': BASIC_TRAINING_CONFIG['weight_decay'],
    'warmup_steps': BASIC_TRAINING_CONFIG['warmup_steps'],
    'max_grad_norm': BASIC_TRAINING_CONFIG['max_grad_norm'],
    'seed': BASIC_TRAINING_CONFIG['seed'],
    'batch_size': BATCH_CONFIG['batch_size'],
    'gradient_accumulation_steps': BATCH_CONFIG['gradient_accumulation_steps'],
    'eval_steps': EVALUATION_CONFIG['eval_steps'],
    'save_steps': CHECKPOINT_CONFIG['save_steps'],
    'low_memory': MEMORY_CONFIG['gradient_checkpointing'],
}

# Export all configurations
__all__ = [
    'TRAINING_CONFIG',
    'BATCH_CONFIG',
    'MEMORY_CONFIG',
    'OPTIMIZER_CONFIG',
    'SCHEDULER_CONFIG',
    'LOSS_CONFIG',
    'EARLY_STOPPING_CONFIG',
    'EVALUATION_CONFIG',
    'LOGGING_CONFIG',
    'CHECKPOINT_CONFIG',
    'OUTPUT_CONFIG',
    'VALIDATION_CONFIG',
    'MULTITASK_CONFIG',
    'THRESHOLD_CONFIG',
    'ADVANCED_CONFIG',
    'DEBUG_CONFIG',
    'REPRODUCIBILITY_CONFIG',
    'HARDWARE_CONFIG'
] 