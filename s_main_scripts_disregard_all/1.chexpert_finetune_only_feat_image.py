import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# import timm
from torchvision import transforms
from PIL import Image

from transformers import (
    Trainer, TrainingArguments, 
    EvalPrediction, DefaultDataCollator,
    set_seed, EarlyStoppingCallback,
    AutoModel
)

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
logger = logging.getLogger('train_image_only_chexpert')

############################
# 1) Fixed label list
############################
# CONDITIONS = [
#     'Atelectasis', 'Cardiomegaly', 'Edema',
#     'Lung Opacity', 'No Finding', 'Pleural Effusion',
#     'Pneumonia', 'Support Devices'
# ]

CONDITIONS = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
                              'Fracture','Lung Lesion','Lung Opacity','No Finding', 'Pleural Effusion','Pleural Other',
                              'Pneumonia','Pneumothorax','Support Devices']

cond2idx = {c: i for i, c in enumerate(CONDITIONS)}

#####################
# === CONSTANTS === #
#####################
# Use HF dataset instead of CSV files
HF_DATASET_NAME = "danjacobellis/chexpert"

# Path reference for original images (not used with HF dataset)
IMG_DIR = os.path.join("output", "img_png")

# Output directory for the image-only model
OUTPUT_DIR = "output_1.chexpert_finetune_only_feat_image"
EXP_NAME = "1.chexpert_finetune_only_feat_image"

OUTPUT_METRICS_FILE = os.path.join(OUTPUT_DIR, "1.chexpert_finetune_only_feat_image_trainer_image_only_metrics.txt")

# BATCH_SIZE = 32
# EPOCHS = 100
# LR = 1e-4
# WD = 1e-4
# GRAD_ACCUM = 4
# WARMUP_STEPS = 20
# EARLY_STOPPING_PATIENCE = 3


BATCH_SIZE   = 64
GRAD_ACCUM   = 4
LR           = 5e-5
WD           = 1e-4
EPOCHS       = 20
EARLY_STOPPING_PATIENCE = 5
WARMUP_STEPS = 100

# Use HF ViT model
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"

# Stratification column for consistent data splits
STRATIFY_COLUMN_NAME = "Consolidation" # Column to use for stratification

SEED = 42
set_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

#########################
# === DATASET DEF ===   #
#########################
class CXRDataset(Dataset):
    def __init__(self, hf_dataset: Any, split_name: str = "train"):
        logger.info(f"Initializing CXRDataset for {split_name} with {len(hf_dataset)} samples from Hugging Face dataset.")
        self.hf_dataset = hf_dataset
        self.is_train = split_name == "train"
        
        self.img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5) if self.is_train else nn.Identity(),
            transforms.RandomRotation(3) if self.is_train else nn.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
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
            # For binary classification (0, 1)
            labels = torch.tensor(item["cond_vec"], dtype=torch.float32)
        except KeyError:
            logger.error(f"'cond_vec' not found in dataset item {idx}. Ensure preprocessing was applied.")
            # Return a zero vector for labels if not found, matching num_conditions
            labels = torch.zeros(len(CONDITIONS), dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error processing labels for item {idx}: {e}")
            labels = torch.zeros(len(CONDITIONS), dtype=torch.float32)

        # --- Unique Identifier ---
        unique_id = f"sample_{idx}"

        return {
            "pixel_values": img,
            "labels": labels,
            "unique_id": unique_id
        }

###############################
# === MAKE COND VEC FUNC ===  #
###############################
def make_cond_vec(cs: str):
    if pd.isna(cs) or cs == "":
        return [0] * len(CONDITIONS)
    if isinstance(cs, list): conds = cs
    elif isinstance(cs, str):
        if '|' in cs: conds = [c.strip() for c in cs.split('|')]
        elif ',' in cs: conds = [c.strip() for c in cs.split(',')]
        else: conds = [cs.strip()]
    else: conds = []
    vec = [0] * len(CONDITIONS)
    for c in conds: 
        if c in cond2idx: vec[cond2idx[c]] = 1
    return vec

def map_labels(batch):
    # Create the binary label vector (0, 1) for each sample in the batch
    batch_cond_vecs = []
    
    # Find a condition column that exists to get batch size
    condition_cols = [col for col in batch.keys() if col in CONDITIONS]
    if not condition_cols:
        logger.error(f"No condition columns found in batch. Available columns: {list(batch.keys())}")
        return batch
        
    num_samples = len(batch[condition_cols[0]]) # Get batch size
    
    for i in range(num_samples):
        # Initialize with zeros
        cond_vec = [0] * len(CONDITIONS)
        for cond_idx, cond_name in enumerate(CONDITIONS):
            if cond_name in batch:
                # Get CheXpert label (0-3) and convert to binary
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
                
                # Convert CheXpert labels (0-3) to binary (0-1) for this task
                # 0: Not mentioned -> 0
                # 1: Negative -> 0  
                # 2: Uncertain -> 0 (could be different depending on policy)
                # 3: Positive -> 1
                binary_label = 1 if label_value == 3 else 0
                cond_vec[cond_idx] = binary_label
                
        batch_cond_vecs.append(cond_vec)
    
    # Add the list of vectors as a new column
    batch["cond_vec"] = batch_cond_vecs
    return batch

#########################
# === MODEL DEF ===     #
#########################
class ImageOnlyModel(nn.Module):
    def __init__(self, num_conditions:int, model_name:str):
        super().__init__()
        logger.info(f"Initializing ImageOnlyModel with {model_name} for {num_conditions} conditions")
        
        # Use AutoModel from Hugging Face instead of timm
        try:
            self.vit = AutoModel.from_pretrained(model_name)
            img_feature_dim = self.vit.config.hidden_size
            logger.info(f"Loaded HF model {model_name}. Feature dimension: {img_feature_dim}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.warning("Attempting to load fallback model: google/vit-base-patch16-224")
            self.vit = AutoModel.from_pretrained("google/vit-base-patch16-224")
            img_feature_dim = self.vit.config.hidden_size
            
        self.classifier = nn.Linear(img_feature_dim, num_conditions)

    def forward(self, pixel_values=None, labels=None):
        # HF ViT expects pixel_values as input
        img_outputs = self.vit(pixel_values=pixel_values)
        
        # Use pooler_output if available, otherwise CLS token's last_hidden_state
        if hasattr(img_outputs, 'pooler_output') and img_outputs.pooler_output is not None:
            img_feat = img_outputs.pooler_output
        else:
            img_feat = img_outputs.last_hidden_state[:, 0, :]  # CLS token
            
        logits = self.classifier(img_feat)
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        return {
            "loss": loss,
            "logits": logits
        }

#############################
# === METRICS ===         #
#############################
def compute_metrics(pred: EvalPrediction):
    logits = pred.predictions
    labels = pred.label_ids
    
    # Handle case where predictions is a dict
    if isinstance(logits, dict):
        logits = logits['logits']
        
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(np.int32)
    aucs = []
    for i in range(labels.shape[1]):
        try:
            if len(np.unique(labels[:, i])) > 1:
                aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
        except Exception:
            pass
    mean_auc = np.mean(aucs) if aucs else 0.0
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    accuracy = accuracy_score(labels, preds)
    return {
        "auc": mean_auc,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy
    }

###############################################
# === Helper Function: Log Label Distribution === #
###############################################
def log_condition_label_distribution(dataset: Any, dataset_name: str, conditions: List[str]):
    logger.info(f"--- Binary Label Distribution for {dataset_name} Set ({len(dataset)} samples) ---")
    label_counts = {cond: {0: 0, 1: 0} for cond in conditions}
    total_samples = len(dataset)
    
    # Check if dataset is empty
    if total_samples == 0:
        logger.warning(f"{dataset_name} dataset is empty. Cannot calculate distribution.")
        return

    # Iterate through the dataset to count labels for each condition
    # Note: This samples first 1000 for performance on large datasets
    sample_size = min(1000, total_samples)
    logger.info(f"Sampling {sample_size} out of {total_samples} samples for distribution calculation.")
    
    try:
        for i in tqdm(range(sample_size), desc=f"Calculating {dataset_name} distribution"):
            cond_vector = dataset[i]['cond_vec']
            if len(cond_vector) == len(conditions):
                for cond_idx, label in enumerate(cond_vector):
                    condition_name = conditions[cond_idx]
                    if label in label_counts[condition_name]:
                        label_counts[condition_name][label] += 1
                    else:
                        # This case should ideally not happen for binary labels
                        logger.warning(f"Unexpected binary label value '{label}' found for condition '{condition_name}' in {dataset_name} set sample {i}.")
            else:
                 logger.warning(f"Length mismatch between cond_vec ({len(cond_vector)}) and CONDITIONS ({len(conditions)}) in {dataset_name} set sample {i}.")
        
        # Log the results
        distribution_log = [f"Binary Condition Label Distribution ({dataset_name} Set):"]
        headers = ["Condition", "Negative (0)", "Positive (1)", "Total", "Positive %"]
        rows = []
        for condition_name in conditions:
            counts = label_counts[condition_name]
            total = sum(counts.values())
            pos_pct = (counts[1] / total * 100) if total > 0 else 0.0
            row = [condition_name, counts[0], counts[1], total, f"{pos_pct:.1f}%"]
            rows.append(row)
        
        table = tabulate(rows, headers=headers, tablefmt="grid")
        distribution_log.append(table)
        logger.info("\n".join(distribution_log))
        logger.info(f"--- End {dataset_name} Set Distribution ---")
        
    except KeyError:
        logger.error(f"'cond_vec' column not found in {dataset_name} dataset. Cannot calculate distribution. Was map_labels applied correctly?")
        return
    except Exception as e:
        logger.error(f"Error during label distribution calculation for {dataset_name}: {e}")
        return

#############################
# === TRAIN ===            #
#############################
def main():
    # Load CheXpert Data from Hugging Face
    logger.info(f"Loading {HF_DATASET_NAME} dataset from Hugging Face Hub...")
    try:
        # Load the full dataset
        dataset = load_dataset(HF_DATASET_NAME, split='train')
        logger.info(f"Loaded {len(dataset)} samples from the dataset.")
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
        logger.warning(f"Missing: {[c for c in CONDITIONS if c not in target_conditions_present]}")

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
            logger.warning(f"Stratification failed for first split (may have classes with too few samples): {e}. Falling back to random split.")
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
    logger.info("Applying label mapping preprocessing (converting CheXpert 0-3 labels to binary 0-1) to TRAIN split...")
    try:
        train_dataset_hf = train_dataset_hf_raw.map(map_labels, batched=True, batch_size=1000, num_proc=4)
        logger.info("Train label mapping complete.")
    except Exception as e:
        logger.error(f"Error during TRAIN preprocessing map function: {e}")
        return

    logger.info("Applying label mapping preprocessing (converting CheXpert 0-3 labels to binary 0-1) to VALIDATION split...")
    try:
        val_dataset_hf = val_dataset_hf_raw.map(map_labels, batched=True, batch_size=1000, num_proc=4)
        logger.info("Validation label mapping complete.")
    except Exception as e:
        logger.error(f"Error during VALIDATION preprocessing map function: {e}")
        return

    logger.info("Applying label mapping preprocessing (converting CheXpert 0-3 labels to binary 0-1) to TEST split...")
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

    # === Create dataset objects ===
    train_dataset = CXRDataset(train_dataset_hf, split_name="train")
    val_dataset = CXRDataset(val_dataset_hf, split_name="val")
    test_dataset = CXRDataset(test_dataset_hf, split_name="test")

    # Quick check if datasets are empty after filtering (should not happen if split worked)
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        logger.error("Train, Validation, or Test dataset is empty after splitting/instantiation.")
        return

    model = ImageOnlyModel(len(CONDITIONS), VIT_MODEL_NAME)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LR,
        weight_decay=WD,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none"
    )
    
    # Add early stopping callback with patience of 3
    callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    
    # Check for existing checkpoints to resume from
    resume_from_checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoint_dirs = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            # Sort by checkpoint number
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = checkpoint_dirs[-1]
            resume_from_checkpoint = os.path.join(OUTPUT_DIR, latest_checkpoint)
            logger.info(f"Found existing checkpoint: {resume_from_checkpoint}. Will resume training from this point.")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DefaultDataCollator(),
        callbacks=callbacks
    )

    # Train with resume_from_checkpoint
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the model
    model_save_path = os.path.join(OUTPUT_DIR, f"model_{EXP_NAME}")
    trainer.save_model(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    logger.info(f"Test metrics: {test_metrics}")

    # Get predictions for test set for detailed per-condition metrics
    logger.info("Getting predictions for CSV generation...")
    test_predictions = trainer.predict(test_dataset)
    # In ImageOnlyModel, predictions from trainer.predict() are the logits directly
    logits = test_predictions.predictions 
    labels = test_predictions.label_ids
    
    # Ensure logits and labels are numpy arrays
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # === Create CSV with predictions and real labels ===
    logger.info("Creating CSV with real and predicted labels...")
    
    # Convert logits to binary predictions (for binary classification)
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    predicted_labels = (probs > 0.5).astype(int)  # Binary predictions (0 or 1)
    
    # Create DataFrame
    csv_data = []
    batch_size = logits.shape[0]
    for i in range(batch_size):
        row = {"unique_id": f"sample_{i}"}
        
        # Add real labels (binary: 0 or 1)
        for j, condition in enumerate(CONDITIONS):
            row[f"real_{condition}"] = int(labels[i][j])
        
        # Add predicted labels (binary: 0 or 1)
        for j, condition in enumerate(CONDITIONS):
            row[f"pred_{condition}"] = int(predicted_labels[i][j])
            
        # Add prediction probabilities
        for j, condition in enumerate(CONDITIONS):
            row[f"prob_{condition}"] = float(probs[i][j])
            
        csv_data.append(row)
    
    # Save to CSV
    csv_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(OUTPUT_DIR, "test_predictions_and_labels.csv")
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"Saved predictions and labels CSV to: {csv_path}")

    logger.info("Calculating per-condition test metrics...")
    y_pred = (probs > 0.5).astype(int)
    
    # Calculate metrics for each condition
    logger.info("\nTest Set Metrics per Condition:")
    logger.info("-" * 80)
    headers = ["Condition", "Accuracy", "Precision", "Recall", "F1-score", "Support", "AUC"]
    rows = []
    
    for i, condition in enumerate(CONDITIONS):
        accuracy = accuracy_score(labels[:, i], y_pred[:, i])
        precision = precision_score(labels[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(labels[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(labels[:, i], y_pred[:, i], zero_division=0)
        support = np.sum(labels[:, i])
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i]) if len(np.unique(labels[:, i])) > 1 else np.nan
        except ValueError: # Handles cases where only one class is present in y_true for a condition
            auc = np.nan
            
        rows.append([
            condition,
            f"{accuracy:.3f}",
            f"{precision:.3f}",
            f"{recall:.3f}",
            f"{f1:.3f}",
            f"{int(support)}",
            f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        ])

    # Add macro average from trainer.evaluate result (which uses compute_metrics)
    rows.append([
        "Macro Avg",
        f"{test_metrics['test_accuracy']:.3f}",
        f"{test_metrics['test_precision']:.3f}",
        f"{test_metrics['test_recall']:.3f}",
        f"{test_metrics['test_f1']:.3f}",
        f"{len(test_dataset)} (Total Samples)", # Total samples as support for macro
        f"{test_metrics['test_auc']:.3f}"
    ])
    
    table_str = tabulate(rows, headers=headers, tablefmt="grid")
    logger.info("\n" + table_str)
    
    # Save the table to a text file in OUTPUT_DIR
    with open(OUTPUT_METRICS_FILE, "w") as f:
        f.write("Image-Only CheXpert Binary Classification Test Metrics:\n")
        f.write("-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-\n")
        f.write(table_str)
        f.write("\n\n")
        f.write("Overall metrics:\n")
        for key, value in test_metrics.items():
            if key.startswith("test_"):
                display_key = key[5:].replace('_', ' ').capitalize()
                f.write(f"{display_key}: {value:.4f}\n")

    # Save metrics as JSON
    metrics_json_path = os.path.join(OUTPUT_DIR, "metrics_image_only_chexpert.json")
    try:
        with open(metrics_json_path, "w") as f:
            json.dump({
                "test_auc": test_metrics.get("test_auc", 0.0),
                "test_f1": test_metrics.get("test_f1", 0.0),
                "test_recall": test_metrics.get("test_recall", 0.0),
                "test_precision": test_metrics.get("test_precision", 0.0),
                "test_accuracy": test_metrics.get("test_accuracy", 0.0)
            }, f, indent=4)
        logger.info(f"Metrics saved to {metrics_json_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics JSON: {e}")
    
    logger.info("Training and evaluation complete.")

    return test_metrics

if __name__=="__main__":
    main()
