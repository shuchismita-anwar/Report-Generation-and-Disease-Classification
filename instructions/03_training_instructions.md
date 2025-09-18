# Training Instructions

## Overview

This document provides instructions for training all models in the medical report generation system.

## Prerequisites

1. Activate virtual environment: `.\venv\Scripts\activate`
2. Ensure dataset setup is complete (see `01_dataset_setup.md`)
3. Verify all required data is in `.\data_dump\output\`

## Training Scripts

### 1. Enhanced Gaze Models

#### 1.1 Experiment: Train+Val as Training Data

**Script:** `.\main\0.((experiment)((train+val)_as_train_data)full+enhanced_gaze)_training_mimic_on_chexpert_optimized.py`

**Hyperparameters:**

- Learning Rate: `6e-6`
- Batch Size: `8` (low memory) / `32` (normal)
- Epochs: `40`
- LR Scheduler: `cosine`

**Description:** Experimental configuration using combined train+validation data for training.

#### 1.2 Standard Enhanced Gaze

**Script:** `.\main\0.(full+enhanced_gaze)_training_mimic_on_chexpert_optimized.py`

**Hyperparameters:**

- Learning Rate: `6e-6`
- Batch Size: `8` (low memory) / `32` (normal)
- Epochs: `40`
- LR Scheduler: `cosine`

**Description:** Full multimodal model with enhanced gaze attention mechanisms.

### 2. Ablation Study Models

#### 2.1 Full Model (Baseline)

**Script:** `.\main\1.(full)_training_mimic_on_chexpert_optimized.py`

**Hyperparameters:**

- Learning Rate: `6e-6`
- Batch Size: `8` (low memory) / `32` (normal)
- Epochs: `40`
- LR Scheduler: `cosine`

**Description:** Complete multimodal model with all features (image + gaze + transcript + bbox).

#### 2.2 Fixation Removed

**Script:** `.\main\2.(fixation_removed)_training_mimic_on_chexpert_optimized.py`

**Hyperparameters:**

- Learning Rate: `6e-6`
- Batch Size: `8` (low memory) / `32` (normal)
- Epochs: `40`
- LR Scheduler: `cosine`

**Description:** Model without fixation sequence data.

#### 2.3 Transcript Removed

**Script:** `.\main\3.(transcript_removed)_training_mimic_on_chexpert_optimized.py`

**Hyperparameters:**

- Learning Rate: `6e-6`
- Batch Size: `8` (low memory) / `32` (normal)
- Epochs: `40`
- LR Scheduler: `cosine`

**Description:** Model without transcript/text data.

#### 2.4 Bounding Box Removed

**Script:** `.\main\4.(bbox_removed)_training_mimic_on_chexpert_optimized.py`

**Hyperparameters:**

- Learning Rate: `6e-6`
- Batch Size: `8` (low memory) / `32` (normal)
- Epochs: `40`
- LR Scheduler: `cosine`

**Description:** Model without bounding box data.

### 3. Baseline Models

#### 3.1 Original MIMIC Training

**Script:** `.\main\training_mimic_on_chexpert.py`

**Hyperparameters:**

- Learning Rate: `5e-6`
- Batch Size: `8` (low memory) / `32` (normal)
- Epochs: `35`
- LR Scheduler: `cosine`

**Description:** Original multimodal training configuration.

#### 3.2 Vision Transformer Only

**Script:** `.\main\vit_chexpert_training.py`

**Hyperparameters:**

- Learning Rate: `5e-5`
- Batch Size: `128`
- Epochs: `20`

**Description:** Pure Vision Transformer model without multimodal features.

## Output Locations

**Model Outputs:** `.\main\output\`

The output folder contains subfolders for each training run:

- `0.((train+val)full+enhanced_gaze)_training_mimic_on_chexpert_optimized/`
- `0.(full+enhanced_gaze)_training_mimic_on_chexpert_optimized/`
- `1.(full)_training_mimic_on_chexpert_optimized/`
- `2.(fixation_removed)_training_mimic_on_chexpert_optimized/`
- `3.(transcript_removed)_training_mimic_on_chexpert_optimized/`
- `4.(bbox_removed)_training_mimic_on_chexpert_optimized/`
- `training_mimic_on_chexpert/`
- `vit_chexpert_training_output/`

**Note:** The output folder is automatically created during training.

## Training Process

1. **Activate Environment**

   ```powershell
   .\venv\Scripts\activate
   ```

2. **Run Training Script**

   ```powershell
   python .\main\[script_name].py
   ```

3. **Monitor Progress**
   - Training logs are saved in respective output folders
   - Models are checkpointed during training
   - Final models are saved in the output directories

## Memory Considerations

- **Low Memory Systems:** Batch size automatically reduces to 8
- **Normal Systems:** Batch size of 32 for multimodal models, 128 for ViT-only
- **GPU Memory:** Scripts include automatic memory optimization

## Model Architecture Details

### Multimodal Components

- **Vision:** ViT-base-patch16-224-in21k
- **Text:** Bio_ClinicalBERT
- **Gaze Processing:** GRU-based fixation encoding
- **Attention:** Enhanced spatial attention mechanisms

### Training Features

- **Loss-based Early Stopping:** Patience=20 with overfitting detection
- **Cosine Learning Rate Scheduling**
- **Mixed Precision Training**
- **Gradient Accumulation** for memory efficiency

## Troubleshooting

- **CUDA Out of Memory:** Reduce batch size or enable gradient checkpointing
- **Data Loading Issues:** Verify all required files in `.\data_dump\output\`
- **Model Save Errors:** Ensure sufficient disk space in output directory
