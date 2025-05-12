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

import timm
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, EvalPrediction, PreTrainedModel, EarlyStoppingCallback
import safetensors
from datasets import load_dataset # Import Hugging Face datasets
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)
from tabulate import tabulate
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('finetune_chexpert_image_only') # Updated logger name

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
# Removed CSV Paths
# TRAIN_CSV    = "dataset_splits/train.csv"
# VAL_CSV      = "dataset_splits/val.csv"
# TEST_CSV     = "dataset_splits/test.csv"
IMG_DIR      = os.path.join("output", "img_png") # Keep for reference, but dataset loader handles images now

HF_DATASET_NAME = "danjacobellis/chexpert"

# Path to the *multi-modal* model checkpoint containing the pre-trained image encoder
MULTI_MODAL_MODEL_PATH = "output_full_with_report/model_full_with_report_vit_base_patch16_224/model.safetensors"

# Output directory for the CheXpert fine-tuned image-only model
OUTPUT_DIR   = "output_image_only_chexpert_finetuned"
EXP_NAME     = "image_only_classifier_chexpert_finetuned"

OUTPUT_METRICS_FILE = os.path.join(OUTPUT_DIR, "finetuned_image_only_metrics.txt")

BATCH_SIZE   = 1024
GRAD_ACCUM   = 2 # Can adjust as needed
LR           = 5e-5 # Learning rate for the classifier head
WD           = 1e-4
EPOCHS       = 20 # Number of epochs for fine-tuning the head
EARLY_STOPPING_PATIENCE = 10
WARMUP_STEPS = 50

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Ensure consistency with the trained ViT
VIT_MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
IMG_PROJ_DIM = 768 # Dimension of the image projection layer output

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
            labels = torch.tensor(item["cond_vec"], dtype=torch.float32)
        except KeyError:
            logger.error(f"'cond_vec' not found in dataset item {idx}. Ensure preprocessing was applied.")
            # Return a zero vector for labels if not found, matching num_conditions
            labels = torch.zeros(len(CONDITIONS), dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error processing labels for item {idx}: {e}")
            labels = torch.zeros(len(CONDITIONS), dtype=torch.float32)

        return {
            "img": img,
            "labels": labels
        }

###############################
# === MAKE COND VEC FUNC ===  # (This will be replaced by a preprocessing function)
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

#########################
# === MODEL DEF ===     # (Simplified Image-Only Classifier)
#########################
class ImageOnlyClassifier(nn.Module):
    def __init__(self, num_conditions: int):
        super().__init__()
        logger.info(f"Initializing ImageOnlyClassifier model with {num_conditions} conditions")

        # Image encoder (ViT + Projection) - Load weights from multi-modal training
        self.vit = timm.create_model(VIT_MODEL_NAME, pretrained=False, num_classes=0) # Structure only
        self.img_proj = nn.Linear(self.vit.num_features, IMG_PROJ_DIM) # Match projection dim

        # *NEW* Classifier Head - Takes projected image features
        # Will have random weights initially unless fine-tuned later
        self.classifier = nn.Linear(IMG_PROJ_DIM, num_conditions)

    def freeze_encoder(self):
        logger.info("Freezing image encoder (ViT and projection layer) weights.")
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.img_proj.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        logger.info("Unfreezing image encoder (ViT and projection layer) weights.")
        for param in self.vit.parameters():
            param.requires_grad = True
        for param in self.img_proj.parameters():
            param.requires_grad = True

    def forward(self, img, labels=None):
        # --- Image encoding ---
        img_feat = self.vit(img)       # [B, vit_features]
        img_proj = self.img_proj(img_feat) # [B, IMG_PROJ_DIM]

        # --- Classification ---
        logits = self.classifier(img_proj) # [B, num_conditions]
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        # Hugging Face Trainer expects a dict or a ModelOutput object.
        # If loss is calculated, it must be returned.
        if loss is not None:
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits} # For inference if no labels are provided

def compute_metrics(eval_pred: EvalPrediction):
    # eval_pred.predictions is a dictionary {"logits": ..., "loss": ...} or just {"logits": ...}
    # eval_pred.label_ids are the labels
    predictions_dict = eval_pred.predictions
    labels_raw = eval_pred.label_ids

    if isinstance(predictions_dict, dict):
        logits = predictions_dict['logits']
    else:
        # Fallback if somehow not a dict (shouldn't happen with current forward)
        logger.warning("compute_metrics received predictions not in expected dict format.")
        logits = predictions_dict 

    # Ensure logits and labels are numpy arrays
    if isinstance(logits, torch.Tensor): logits = logits.cpu().numpy()
    if isinstance(labels_raw, torch.Tensor): labels = labels_raw.cpu().numpy()
    else: labels = np.array(labels_raw)

    # Apply sigmoid and threshold
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    # Calculate metrics
    try:
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        rec = recall_score(labels, preds, average="macro", zero_division=0)
        prec = precision_score(labels, preds, average="macro", zero_division=0)
        
        # Ensure labels and probs have the same shape for AUC
        if labels.shape == probs.shape:
            auc_scores = [roc_auc_score(labels[:, i], probs[:, i]) 
                          for i in range(labels.shape[1]) 
                          if len(np.unique(labels[:, i])) > 1]
            mean_auc = np.mean(auc_scores) if auc_scores else 0.0
        else:
            logger.warning(f"Shape mismatch for AUC: labels {labels.shape}, probs {probs.shape}. AUC set to 0.")
            mean_auc = 0.0
            
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        f1, rec, prec, mean_auc = 0.0, 0.0, 0.0, 0.0
        
    return {"auc": mean_auc, "f1": f1, "recall": rec, "precision": prec}

#################
# === Main ===  #
#################
def main():
    logger.info(f"Starting CheXpert IMAGE-ONLY fine-tuning process on device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        # Create the multi-label vector for each sample in the batch
        batch_cond_vecs = []
        num_samples = len(batch[target_conditions_present[0]]) # Get batch size from one of the columns
        for i in range(num_samples):
            cond_vec = [0] * len(CONDITIONS) # Initialize with zeros
            for idx, cond_name in enumerate(CONDITIONS):
                 if cond_name in target_conditions_present:
                     chexpert_label = batch[cond_name][i]
                     # Mapping: 3 (present) -> 1, others (0, 1, 2) -> 0
                     if chexpert_label == 3:
                          cond_vec[idx] = 1
                     # else: label remains 0
            batch_cond_vecs.append(cond_vec)
        # Add the list of vectors as a new column
        batch["cond_vec"] = batch_cond_vecs
        return batch

    logger.info("Applying label mapping preprocessing...")
    try:
        dataset = dataset.map(map_labels, batched=True, batch_size=1000, num_proc=4) # Use batching and multiple processes
        # Remove original CheXpert label columns if desired, keep only 'image' and 'cond_vec'
        # columns_to_keep = ['image', 'cond_vec']
        # columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        # dataset = dataset.remove_columns(columns_to_remove)
        logger.info("Label mapping complete.")
    except Exception as e:
        logger.error(f"Error during preprocessing map function: {e}")
        return

    # === Split Data (Train: 70%, Val: 15%, Test: 15%) ===
    logger.info("Splitting data into train, validation, and test sets...")
    # First split: 70% train, 30% temporary (for val+test)
    train_valtest_split = dataset.train_test_split(test_size=0.30, seed=SEED)
    train_dataset_hf = train_valtest_split['train']
    val_test_dataset_hf = train_valtest_split['test']

    # Second split: Split the 30% into 15% validation and 15% test (relative split 50/50)
    val_test_split = val_test_dataset_hf.train_test_split(test_size=0.50, seed=SEED)
    val_dataset_hf = val_test_split['train']
    test_dataset_hf = val_test_split['test']

    logger.info(f"Data split complete: Train={len(train_dataset_hf)}, Val={len(val_dataset_hf)}, Test={len(test_dataset_hf)}")

    # === Instantiate Custom Datasets ===
    train_dataset = CXRImageDataset(train_dataset_hf, split_name="train")
    val_dataset = CXRImageDataset(val_dataset_hf, split_name="val")
    test_dataset = CXRImageDataset(test_dataset_hf, split_name="test")

    # Quick check if datasets are empty after filtering (should not happen if split worked)
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        logger.error("Train, Validation, or Test dataset is empty after splitting/instantiation.")
        return

    # === Load Model ===
    logger.info(f"Initializing ImageOnlyClassifier structure for {len(CONDITIONS)} conditions.")
    model = ImageOnlyClassifier(len(CONDITIONS))

    logger.info(f"Loading trained weights from MULTI-MODAL checkpoint: {MULTI_MODAL_MODEL_PATH}")
    try:
        multi_modal_state_dict = safetensors.torch.load_file(MULTI_MODAL_MODEL_PATH, device="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(multi_modal_state_dict, strict=False)
        logger.info(f"Loaded pre-trained encoder weights with strict=False.")
        if missing_keys: logger.warning(f"Missing keys (expected classifier): {missing_keys}")
        # if unexpected_keys: logger.info(f"Unexpected keys (from multi-modal): {unexpected_keys}") # Can be verbose

    except FileNotFoundError:
        logger.error(f"Error: Multi-modal model weights file not found at {MULTI_MODAL_MODEL_PATH}")
        return
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        return

    # === Freeze Encoder ===
    model.freeze_encoder()

    model.to(DEVICE) # Move model to device after loading and freezing

    # === Setup Training ===
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
        logging_steps=50, # Log more frequently during fine-tuning
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1", # Use F1 for best model selection
        greater_is_better=True,
        dataloader_num_workers=4,
        save_safetensors=True,
        remove_unused_columns=False, # Keep columns needed by model forward
        report_to="none" # Disable external reporting like W&B
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

    # --- Save final model explicitly ---
    final_model_path = os.path.join(OUTPUT_DIR, f"model_{EXP_NAME}_final")
    trainer.save_model(final_model_path)
    logger.info(f"Final fine-tuned image-only model saved to {final_model_path}")

    # --- Generate and Save Per-Condition Metrics Table ---
    logger.info("Calculating final per-condition test metrics...")
    # Get predictions using the best model loaded by Trainer
    test_predictions = trainer.predict(test_dataset)
    logits = test_predictions.predictions # Trainer.predict often returns the logits array directly
    labels = test_predictions.label_ids

    # Ensure numpy arrays
    if isinstance(logits, torch.Tensor): logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()

    probs = 1 / (1 + np.exp(-logits))
    y_pred = (probs > 0.5).astype(int)

    logger.info("\nFine-tuned Image-Only Test Set Metrics per Condition:")
    logger.info("-" * 80)
    headers = ["Condition", "Precision", "Recall", "F1-score", "Support", "AUC"]
    rows = []

    for i, condition in enumerate(CONDITIONS):
        precision = precision_score(labels[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(labels[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(labels[:, i], y_pred[:, i], zero_division=0)
        support = np.sum(labels[:, i])
        try:
            unique_labels = np.unique(labels[:, i])
            if len(unique_labels) > 1: auc = roc_auc_score(labels[:, i], probs[:, i])
            else: auc = np.nan
        except ValueError: auc = np.nan

        rows.append([
            condition, f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}", f"{int(support)}",
            f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        ])

    # Add macro average from the final evaluate call
    rows.append([
        "Macro Avg",
        f"{test_metrics['test_precision']:.3f}", # Use test_ prefix
        f"{test_metrics['test_recall']:.3f}",
        f"{test_metrics['test_f1']:.3f}",
        f"{len(labels)} (Total Samples)",
        f"{test_metrics['test_auc']:.3f}"
    ])

    table_str = tabulate(rows, headers=headers, tablefmt="grid")
    logger.info("\n" + table_str)

    # Save the table
    try:
        with open(OUTPUT_METRICS_FILE, "w") as f:
            f.write("Fine-tuned Image-Only Test Set Metrics per Condition:\n")
            f.write("-" * 80 + "\n")
            f.write(table_str)
        logger.info(f"Metrics report saved to {OUTPUT_METRICS_FILE}")
    except Exception as e:
        logger.error(f"Failed to save metrics report: {e}")

    # Save overall test metrics dictionary
    metrics_json_path = os.path.join(OUTPUT_DIR, "finetuned_test_metrics.json")
    try:
        with open(metrics_json_path, "w") as f:
            json.dump(test_metrics, f, indent=4)
        logger.info(f"Overall test metrics saved to {metrics_json_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics JSON: {e}")

    logger.info("Image-only fine-tuning and evaluation complete.")

if __name__ == "__main__":
    main()
