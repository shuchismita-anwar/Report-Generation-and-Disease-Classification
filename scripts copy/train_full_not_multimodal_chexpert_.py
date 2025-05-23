import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# import timm # Removed timm
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, EvalPrediction, PreTrainedModel, EarlyStoppingCallback
import safetensors
from datasets import load_dataset # Import Hugging Face datasets
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score
)
from tabulate import tabulate
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('finetune_chexpert_image_only_raddino') # Updated logger name to be more specific

############################
# 1) Fixed label list
############################
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Edema',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pneumonia', 'Support Devices'
]
cond2idx = {c: i for i, c in enumerate(CONDITIONS)}

#####################
# === CONSTANTS === #
#####################
IMG_DIR      = os.path.join("output", "img_png") # Keep for reference, but dataset loader handles images now

HF_DATASET_NAME = "danjacobellis/chexpert"

# Path to the *multi-modal* model checkpoint containing the pre-trained image encoder
# MULTI_MODAL_MODEL_PATH = r"output_full_with_report_exp_raddino\\model_full_rad_dino_no_report\\model.safetensors" # <-- REMOVED/Commented out

# Output directory for the CheXpert fine-tuned image-only model
OUTPUT_DIR   = "output_image_only_chexpert_finetuned_raddino_from_hf" # <-- UPDATED Path
EXP_NAME     = "image_only_classifier_chexpert_finetuned_raddino_from_hf" # <-- UPDATED Name

OUTPUT_METRICS_FILE = os.path.join(OUTPUT_DIR, "finetuned_image_only_metrics.txt")

BATCH_SIZE   = 64
GRAD_ACCUM   = 4
LR           = 2e-5 # <-- UPDATED LR for full fine-tuning
WD           = 1e-4
EPOCHS       = 1 # Number of epochs for fine-tuning
EARLY_STOPPING_PATIENCE = 5
WARMUP_STEPS = 100

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Ensure consistency with the trained ViT
# VIT_MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
IMG_PROJ_DIM = 768 # Dimension of the image projection layer output

# === NEW: Specify the ViT model to use for THIS fine-tuning run ===
# FINETUNE_VIT_MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k" # Default to original, change this to try others
# Example: FINETUNE_VIT_MODEL_NAME = "vit_large_patch16_224.augreg_in21k_ft_in1k"
# Example: FINETUNE_VIT_MODEL_NAME = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"
FINETUNE_VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k" # Using a HF model name
# FINETUNE_VIT_MODEL_NAME = 'google/vit-base-patch16-224'
# FINETUNE_VIT_MODEL_NAME = "microsoft/rad-dino"

STRATIFY_COLUMN_NAME = "Consolidation" # Column to use for stratification

#########################
# === DATASET DEF ===   # (Adapted for Hugging Face Dataset)
#########################
class CXRImageDataset(Dataset):
    def __init__(self, hf_dataset: Any, split_name: str = "train"):
        logger.info(f"Initializing CXRImageDataset for {split_name} with {len(hf_dataset)} samples from Hugging Face dataset.")
        self.hf_dataset = hf_dataset
        # Filtering and cond_vec creation is now expected to be done in a preprocessing step before this class is instantiated.

        # Use non-random transforms for evaluation (same as original test transform)
        self.img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3), # Ensure this matches ViT pretraining if applicable
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        # --- Image ---
        try:
            # image field from Hugging Face dataset is typically a PIL Image object
            img = item['image'].convert("RGB") 
            img = self.img_tf(img)
        except Exception as e:
            logger.error(f"Error processing image for item {idx}: {e}")
            img = torch.zeros(3, 224, 224) # Return a blank image on error

        # --- Condition Vector (expected to be preprocessed and named 'cond_vec') ---
        try:
            # Ensure labels are long for CrossEntropyLoss
            labels = torch.tensor(item["cond_vec"], dtype=torch.long)
        except KeyError:
            logger.error(f"'cond_vec' not found in dataset item {idx}. Ensure preprocessing was applied.")
            # Return a zero vector for labels if not found, matching num_conditions
            labels = torch.zeros(len(CONDITIONS), dtype=torch.long) # Use long
        except Exception as e:
            logger.error(f"Error processing labels for item {idx}: {e}")
            labels = torch.zeros(len(CONDITIONS), dtype=torch.long) # Use long

        return {
            "img": img,
            "labels": labels
        }

#########################
# === MODEL DEF ===     # (Simplified Image-Only Classifier)
#########################
class ImageOnlyClassifier(nn.Module):
    def __init__(self, num_conditions: int, vit_model_name: str):
        super().__init__()
        logger.info(f"Initializing ImageOnlyClassifier for {num_conditions} conditions using HF ViT: {vit_model_name} (4-class output).")
        logger.info(f"Model will be fine-tuned end-to-end starting from pre-trained {vit_model_name} weights.") # <-- Added log

        # --- Image encoder (Hugging Face ViT) ---
        try:
            self.vit = AutoModel.from_pretrained(vit_model_name)
            img_feature_dim = self.vit.config.hidden_size
            logger.info(f"Loaded HF ViT {vit_model_name}. Feature dimension: {img_feature_dim}")
        except Exception as e:
            logger.error(f"Failed to load HF ViT {vit_model_name}: {e}")
            logger.warning("Attempting to load fallback HF ViT: google/vit-base-patch16-224-in21k")
            try:
                self.vit = AutoModel.from_pretrained("")
                img_feature_dim = self.vit.config.hidden_size
            except Exception as fallback_e:
                logger.critical(f"FATAL: Failed to load fallback HF image model: {fallback_e}")
                raise fallback_e

        # Projection layer (will be randomly initialized and fine-tuned)
        self.img_proj = nn.Linear(img_feature_dim, IMG_PROJ_DIM)

        # Classifier Head: Outputs 4 logits for each of the num_conditions
        # Total output features = num_conditions * 4
        # (will be randomly initialized and fine-tuned)
        self.classifier = nn.Linear(IMG_PROJ_DIM, num_conditions * 4)

    def forward(self, img, labels=None):
        # --- Image encoding (Hugging Face ViT) ---
        # The HF ViT model expects 'pixel_values' as input kwarg
        img_outputs = self.vit(pixel_values=img)
        # Use pooler_output if available, otherwise CLS token's last_hidden_state
        if hasattr(img_outputs, 'pooler_output') and img_outputs.pooler_output is not None:
            img_feat = img_outputs.pooler_output
        else:
            img_feat = img_outputs.last_hidden_state[:, 0, :] # CLS token

        img_proj = self.img_proj(img_feat) # [B, IMG_PROJ_DIM]

        # --- Classification ---
        # logits will be of shape [B, num_conditions * 4]
        logits = self.classifier(img_proj)
        
        loss = None
        if labels is not None:
            # Reshape for CrossEntropyLoss:
            # Logits: [B, num_conditions * 4] -> [B * num_conditions, 4]
            # Labels: [B, num_conditions] (containing class indices 0-3) -> [B * num_conditions]
            num_classes_per_condition = 4
            loss_fct = nn.CrossEntropyLoss()
            # Ensure labels are torch.long (will be handled in dataset)
            reshaped_logits = logits.view(-1, num_classes_per_condition)
            reshaped_labels = labels.view(-1)
            loss = loss_fct(reshaped_logits, reshaped_labels)

        # Hugging Face Trainer expects a dict or a ModelOutput object.
        # If loss is calculated, it must be returned.
        if loss is not None:
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits} # For inference if no labels are provided

def compute_metrics(eval_pred: EvalPrediction):
    predictions_dict = eval_pred.predictions
    labels_raw = eval_pred.label_ids
    num_classes_per_condition = 4 # 4 classes (0, 1, 2, 3) per condition

    if isinstance(predictions_dict, dict):
        logits = predictions_dict['logits'] # Shape: [B, num_conditions * 4]
    else:
        logger.warning("compute_metrics received predictions not in expected dict format.")
        logits = predictions_dict 

    # Ensure numpy arrays
    if isinstance(logits, torch.Tensor): logits = logits.cpu().numpy()
    if isinstance(labels_raw, torch.Tensor): labels = labels_raw.cpu().numpy() # Shape: [B, num_conditions]
    else: labels = np.array(labels_raw)

    # Reshape for multi-class evaluation (each condition treated as an independent multi-class task)
    # Logits: [B, num_conditions * 4] -> [B * num_conditions, 4]
    # Labels: [B, num_conditions] -> [B * num_conditions]
    logits_reshaped = logits.reshape(-1, num_classes_per_condition)
    labels_reshaped = labels.reshape(-1)

    # Get predictions (predicted class index 0-3 for each condition instance)
    preds_reshaped = np.argmax(logits_reshaped, axis=1)

    # Filter out any potential placeholder/invalid labels (e.g., if -1 was used for missing)
    # Assuming valid labels are strictly 0, 1, 2, 3
    valid_indices = (labels_reshaped >= 0) & (labels_reshaped < num_classes_per_condition)
    if not np.all(valid_indices):
        logger.warning(f"Found invalid labels outside range [0, {num_classes_per_condition-1}]. Filtering them out for metrics.")
        labels_filtered = labels_reshaped[valid_indices]
        preds_filtered = preds_reshaped[valid_indices]
        if len(labels_filtered) == 0:
             logger.error("No valid labels found after filtering in compute_metrics.")
             return {"accuracy": 0.0, "f1_weighted": 0.0, "recall_weighted": 0.0, "precision_weighted": 0.0}
    else:
        labels_filtered = labels_reshaped
        preds_filtered = preds_reshaped

    # Calculate overall multi-class metrics using WEIGHTED averaging
    try:
        accuracy = accuracy_score(labels_filtered, preds_filtered)
        f1_weighted = f1_score(labels_filtered, preds_filtered, average="weighted", zero_division=0)
        recall_weighted = recall_score(labels_filtered, preds_filtered, average="weighted", zero_division=0)
        precision_weighted = precision_score(labels_filtered, preds_filtered, average="weighted", zero_division=0)
    except Exception as e:
        logger.error(f"Error calculating multi-class metrics: {e}")
        accuracy, f1_weighted, recall_weighted, precision_weighted = 0.0, 0.0, 0.0, 0.0

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "recall_weighted": recall_weighted,
        "precision_weighted": precision_weighted
    }

###############################################
# === Helper Function: Log Label Distribution === #
###############################################
def log_condition_label_distribution(dataset: Any, dataset_name: str, conditions: List[str]):
    logger.info(f"--- Label Distribution for {dataset_name} Set ({len(dataset)} samples) ---")
    label_counts = {cond: {0: 0, 1: 0, 2: 0, 3: 0} for cond in conditions}
    total_samples = len(dataset)
    
    # Check if dataset is empty
    if total_samples == 0:
        logger.warning(f"{dataset_name} dataset is empty. Cannot calculate distribution.")
        return

    # Iterate through the dataset to count labels for each condition
    # This might be slow for large datasets, especially train. Consider sampling if needed.
    # Access the 'cond_vec' which holds the list of 8 labels (0-3) for each sample.
    try:
        # for i in tqdm(range(total_samples), desc=f"Calculating {dataset_name} distribution"):
        #     cond_vector = dataset[i]['cond_vec']
        #     if len(cond_vector) == len(conditions):
        #         for cond_idx, label in enumerate(cond_vector):
        #             condition_name = conditions[cond_idx]
        #             if label in label_counts[condition_name]:
        #                 label_counts[condition_name][label] += 1
        #             else:
        #                 # This case should ideally not happen if map_labels clamps values correctly
        #                 logger.warning(f"Unexpected label value '{label}' found for condition '{condition_name}' in {dataset_name} set sample {i}.")
        #     else:
        #          logger.warning(f"Length mismatch between cond_vec ({len(cond_vector)}) and CONDITIONS ({len(conditions)}) in {dataset_name} set sample {i}.")
        pass
    except KeyError:
        logger.error(f"'cond_vec' column not found in {dataset_name} dataset. Cannot calculate distribution. Was map_labels applied correctly?")
        return
    except Exception as e:
        logger.error(f"Error during label distribution calculation for {dataset_name}: {e}")
        return

    # Log the results
    distribution_log = [f"Condition Label Distribution ({dataset_name} Set):"]
    headers = ["Condition"] + [f"Label {i}" for i in range(4)] + ["Total"]
    rows = []
    for condition_name in conditions:
        counts = label_counts[condition_name]
        row = [condition_name] + [counts[j] for j in range(4)] + [sum(counts.values())]
        rows.append(row)
    
    table = tabulate(rows, headers=headers, tablefmt="grid")
    distribution_log.append(table)
    logger.info("\n".join(distribution_log))
    logger.info(f"--- End {dataset_name} Set Distribution ---")

#################
# === Main ===  #
#################
def main():
    logger.info(f"Starting CheXpert IMAGE-ONLY FULL fine-tuning process on device: {DEVICE}") # <-- Updated log
    logger.info(f"Using ViT model for fine-tuning: {FINETUNE_VIT_MODEL_NAME}")
    logger.info(f"Initializing model from Hugging Face ({FINETUNE_VIT_MODEL_NAME}) and fine-tuning all layers.") # <-- Updated log
    
    # Update output dir and exp name based on the ViT model used
    output_dir_suffix = FINETUNE_VIT_MODEL_NAME.replace('.', '_').replace('/', '_') # Sanitize name for path
    # Use the constants defined above
    current_output_dir = os.path.join(OUTPUT_DIR, f"finetune_{output_dir_suffix}")
    current_exp_name = f"{EXP_NAME}" # EXP_NAME already contains the suffix now
    current_output_metrics_file = os.path.join(current_output_dir, "finetuned_image_only_metrics.txt")
    current_metrics_json_path = os.path.join(current_output_dir, "finetuned_test_metrics.json")

    os.makedirs(current_output_dir, exist_ok=True)

    # === Load CheXpert Data from Hugging Face ===
    logger.info(f"Loading {HF_DATASET_NAME} dataset from Hugging Face Hub...")
    try:
        # Load only the train split initially
        dataset = load_dataset(HF_DATASET_NAME, split='train')
        logger.info(f"Loaded {len(dataset)} samples from the train split.")
    except Exception as e:
        logger.error(f"Failed to load dataset '{HF_DATASET_NAME}' from Hugging Face Hub: {e}")
        return

    # === Define Target Conditions and Mapping ===
    # Ensure CONDITIONS are subset of CheXpert columns and in desired order
    chexpert_label_cols = dataset.column_names # Get all column names
    target_conditions_present = [cond for cond in CONDITIONS if cond in chexpert_label_cols]
    if len(target_conditions_present) != len(CONDITIONS):
        logger.warning(f"Not all target CONDITIONS found in CheXpert dataset columns.")
        logger.warning(f"Found: {target_conditions_present}")
        # Decide how to handle this - error out or proceed with subset?
        # For now, proceed with the subset found.
        # Update CONDITIONS and cond2idx globally if proceeding with subset (careful)
        # global CONDITIONS, cond2idx
        # CONDITIONS = target_conditions_present
        # cond2idx = {c: i for i, c in enumerate(CONDITIONS)}
        # logger.info(f"Updated CONDITIONS to: {CONDITIONS}")

    def map_labels(batch):
        # Create the multi-class label vector (0, 1, 2, 3) for each sample in the batch
        batch_cond_vecs = []
        num_samples = len(batch[target_conditions_present[0]]) # Get batch size from one of the columns
        for i in range(num_samples):
            # Initialize with a placeholder if a condition is entirely missing from source, e.g., 0
            cond_vec = [0] * len(CONDITIONS)
            for cond_idx, cond_name in enumerate(CONDITIONS):
                 if cond_name in target_conditions_present:
                     chexpert_label = batch[cond_name][i]
                     # Handle potential None or NaN values, defaulting to 0 (Not mentioned)
                     if chexpert_label is None or (isinstance(chexpert_label, float) and np.isnan(chexpert_label)):
                         label_value = 0 # Default for not mentioned or missing
                     else:
                         label_value = int(chexpert_label)
                         # Ensure label is within 0-3 range (CheXpert standard)
                         if not 0 <= label_value <= 3:
                             logger.warning(f"Unexpected label value {label_value} for {cond_name} at sample index {i}. Clamping to 0.")
                             label_value = 0 # Default to 'Not mentioned' if out of range
                     cond_vec[cond_idx] = label_value
                 # else: If target condition not in chexpert_label_cols, it was already initialized to 0

            batch_cond_vecs.append(cond_vec)
        # Add the list of vectors as a new column
        batch["cond_vec"] = batch_cond_vecs
        return batch

    # === Stratified Split Data (BEFORE map_labels) ===
    logger.info(f"Splitting data into train, validation, and test sets, stratifying by '{STRATIFY_COLUMN_NAME}'.")
    # Ensure the stratification column exists
    if STRATIFY_COLUMN_NAME not in dataset.column_names:
        logger.error(f"Stratification column '{STRATIFY_COLUMN_NAME}' not found in dataset. Proceeding with random split.")
        # Fallback to random split if stratify column is missing
        train_valtest_split = dataset.train_test_split(test_size=0.30, seed=SEED)
    else:
        try:
            # First split: 70% train, 30% temporary (for val+test)
            train_valtest_split = dataset.train_test_split(
                test_size=0.30, 
                seed=SEED, 
                stratify_by_column=STRATIFY_COLUMN_NAME
            )
        except ValueError as e:
            logger.warning(f"Stratification failed for first split (อาจมี class ที่มี sample น้อยเกินไป): {e}. Falling back to random split.")
            train_valtest_split = dataset.train_test_split(test_size=0.30, seed=SEED)

    train_dataset_hf_raw = train_valtest_split['train']
    val_test_dataset_hf_raw = train_valtest_split['test']

    # Second split: Split the 30% into 15% validation and 15% test (relative split 50/50)
    if STRATIFY_COLUMN_NAME not in val_test_dataset_hf_raw.column_names:
        logger.error(f"Stratification column '{STRATIFY_COLUMN_NAME}' not found in val/test dataset. Proceeding with random split for this stage.")
        val_test_split = val_test_dataset_hf_raw.train_test_split(test_size=0.50, seed=SEED)
    else:
        try:
            val_test_split = val_test_dataset_hf_raw.train_test_split(
                test_size=0.50, 
                seed=SEED, 
                stratify_by_column=STRATIFY_COLUMN_NAME
            )
        except ValueError as e:
            logger.warning(f"Stratification failed for second split: {e}. Falling back to random split.")
            val_test_split = val_test_dataset_hf_raw.train_test_split(test_size=0.50, seed=SEED)
            
    val_dataset_hf_raw = val_test_split['train']
    test_dataset_hf_raw = val_test_split['test']

    logger.info(f"Data split complete: Raw Train={len(train_dataset_hf_raw)}, Raw Val={len(val_dataset_hf_raw)}, Raw Test={len(test_dataset_hf_raw)}")

    # === Apply map_labels to each split AFTER splitting ===
    logger.info("Applying label mapping preprocessing (extracting 0-3 CheXpert labels) to TRAIN split...")
    try:
        train_dataset_hf = train_dataset_hf_raw.map(map_labels, batched=True, batch_size=1000, num_proc=4)
        logger.info("Train label mapping complete.")
    except Exception as e:
        logger.error(f"Error during TRAIN preprocessing map function: {e}")
        return

    logger.info("Applying label mapping preprocessing (extracting 0-3 CheXpert labels) to VALIDATION split...")
    try:
        val_dataset_hf = val_dataset_hf_raw.map(map_labels, batched=True, batch_size=1000, num_proc=4)
        logger.info("Validation label mapping complete.")
    except Exception as e:
        logger.error(f"Error during VALIDATION preprocessing map function: {e}")
        return

    logger.info("Applying label mapping preprocessing (extracting 0-3 CheXpert labels) to TEST split...")
    try:
        test_dataset_hf = test_dataset_hf_raw.map(map_labels, batched=True, batch_size=1000, num_proc=4)
        logger.info("Test label mapping complete.")
    except Exception as e:
        logger.error(f"Error during TEST preprocessing map function: {e}")
        return

    # === Log Label Distributions AFTER Mapping ===
    log_condition_label_distribution(train_dataset_hf, "Train", CONDITIONS)
    log_condition_label_distribution(val_dataset_hf, "Validation", CONDITIONS)
    log_condition_label_distribution(test_dataset_hf, "Test", CONDITIONS)

    # Remove original CheXpert label columns if desired, keep only 'image' and 'cond_vec'
    # This should be done on the mapped datasets if you want to clean them up
    # columns_to_keep = ['image', 'cond_vec'] 
    # train_dataset_hf = train_dataset_hf.remove_columns([col for col in train_dataset_hf.column_names if col not in columns_to_keep])
    # val_dataset_hf = val_dataset_hf.remove_columns([col for col in val_dataset_hf.column_names if col not in columns_to_keep])
    # test_dataset_hf = test_dataset_hf.remove_columns([col for col in test_dataset_hf.column_names if col not in columns_to_keep])

    # The rest of the script (Instantiate Custom Datasets, Load Model, etc.) uses train_dataset_hf, val_dataset_hf, test_dataset_hf
    # These are now the datasets that have had map_labels applied.

    # === Instantiate Custom Datasets ===
    train_dataset = CXRImageDataset(train_dataset_hf, split_name="train")
    val_dataset = CXRImageDataset(val_dataset_hf, split_name="val")
    test_dataset = CXRImageDataset(test_dataset_hf, split_name="test")

    # Quick check if datasets are empty after filtering (should not happen if split worked)
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        logger.error("Train, Validation, or Test dataset is empty after splitting/instantiation.")
        return

    # === Load Model ===
    logger.info(f"Initializing ImageOnlyClassifier structure for {len(CONDITIONS)} conditions using {FINETUNE_VIT_MODEL_NAME}.")
    # Pass the selected ViT model name to the constructor
    # Model is initialized directly from HF pre-trained weights for ViT,
    # and random weights for projection and classifier.
    model = ImageOnlyClassifier(len(CONDITIONS), vit_model_name=FINETUNE_VIT_MODEL_NAME)

    # === Freeze Encoder ===
    # model.freeze_encoder() # <-- REMOVED call (all layers trainable by default)

    model.to(DEVICE) # Move model to device

    # === Setup Training ===
    training_args = TrainingArguments(
        output_dir=current_output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LR, # Using the updated LR
        weight_decay=WD,
        logging_dir=os.path.join(current_output_dir, "logs"),
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        dataloader_num_workers=4,
        save_safetensors=True,
        remove_unused_columns=False,
        report_to="none"
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    # === Train ===
    logger.info("Starting fine-tuning of the classifier head...")
    trainer.train()

    # === Final Evaluation ===
    logger.info("Fine-tuning complete. Evaluating on Test Set...")
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test") # Add prefix for test metrics
    logger.info(f"Test metrics: {test_metrics}")

    # === Get predictions using the best model loaded by Trainer ===
    # test_predictions = trainer.predict(test_dataset) # No longer need to redo predict here, test_metrics has it
    # logits = test_predictions.predictions # Get logits array
    # labels = test_predictions.label_ids

    logger.info("\nFine-tuned Image-Only Test Set Metrics (4-Class Multi-Label):")
    logger.info("-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-")
    # Log the metrics dictionary returned by compute_metrics during trainer.evaluate
    for key, value in test_metrics.items():
        if key.startswith("test_"): # Hugging Face prepends "test_"
             # Adjust key display for weighted-averaged metrics
             display_key = key[5:].replace('_', ' ').replace('weighted', '(weighted)').capitalize()
             logger.info(f"{display_key}: {value:.4f}")
    logger.info("-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-")

    # Save the detailed metrics dictionary to a file (already contains what we need)
    # The old per-condition binary table is no longer suitable.
    try:
        with open(current_output_metrics_file, "w") as f:
            f.write("Fine-tuned Image-Only Test Set Metrics (4-Class Multi-Label - CheXpert 0-3):\n")
            f.write("-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-\n")
            for key, value in test_metrics.items():
                if key.startswith("test_"):
                    display_key = key[5:].replace('_', ' ').replace('weighted', '(weighted)').capitalize()
                    f.write(f"{display_key}: {value:.4f}\n")
        logger.info(f"Metrics report saved to {current_output_metrics_file}")
    except Exception as e:
        logger.error(f"Failed to save metrics report: {e}")

    # Save the overall test metrics dictionary (already in test_metrics)
    metrics_json_path = current_metrics_json_path # Use updated path
    try:
        with open(metrics_json_path, "w") as f:
            json.dump(test_metrics, f, indent=4)
        logger.info(f"Overall test metrics saved to {metrics_json_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics JSON: {e}")
    
    # === NEW: Visualize predictions for a few test samples ===
    logger.info("\n=== Visualizing predictions for sample test images ===")
    
    # Get predictions for test set
    test_predictions = trainer.predict(test_dataset)
    logits = test_predictions.predictions
    true_labels = test_predictions.label_ids
    
    # Number of samples to visualize
    num_samples_to_show = min(5, len(test_dataset))
    
    # For each sample, show the predictions
    for i in range(num_samples_to_show):
        # Get the logits for this sample (shape: [num_conditions * 4])
        sample_logits = logits[i]
        
        # Reshape to [num_conditions, 4]
        sample_logits_reshaped = sample_logits.reshape(len(CONDITIONS), 4)
        
        # Get the predicted class (0-3) for each condition
        sample_preds = np.argmax(sample_logits_reshaped, axis=1)
        
        # Get the true labels for this sample
        sample_true_labels = true_labels[i]
        
        # Create a binary vector for conditions with prediction > 0
        # (0 = not mentioned, >0 = some level of presence)
        binary_pred_vector = [1 if pred > 0 else 0 for pred in sample_preds]
        
        # Display prediction info
        logger.info(f"\nSample {i+1}:")
        logger.info(f"Binary prediction vector (1 if condition present): {binary_pred_vector}")
        
        # Display per-condition predictions with condition names
        logger.info("Detailed predictions (CheXpert classes 0-3):")
        for cond_idx, condition in enumerate(CONDITIONS):
            pred_class = sample_preds[cond_idx]
            true_class = sample_true_labels[cond_idx]
            class_meaning = {
                0: "Not Mentioned",
                1: "Negative",
                2: "Uncertain",
                3: "Positive"
            }
            pred_status = class_meaning.get(pred_class, "Unknown")
            true_status = class_meaning.get(true_class, "Unknown")
            
            logger.info(f"  {condition}: Predicted={pred_class} ({pred_status}), True={true_class} ({true_status})")
    
    logger.info("=== End of prediction visualization ===")

    logger.info("Image-only FULL fine-tuning and evaluation complete.") # <-- Updated log

if __name__ == "__main__":
    main()
