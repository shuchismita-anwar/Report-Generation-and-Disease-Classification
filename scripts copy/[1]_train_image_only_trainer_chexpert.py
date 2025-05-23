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
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Edema',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pneumonia', 'Support Devices'
]
cond2idx = {c: i for i, c in enumerate(CONDITIONS)}

#####################
# === CONSTANTS === #
#####################
# Use HF dataset instead of CSV files
HF_DATASET_NAME = "danjacobellis/chexpert"

# Path reference for original images (not used with HF dataset)
IMG_DIR = os.path.join("output", "img_png")

# Output directory for the image-only model
OUTPUT_DIR = "output_image_only_trainer_chexpert"
EXP_NAME = "image_only_trainer_chexpert"

OUTPUT_METRICS_FILE = os.path.join(OUTPUT_DIR, "trainer_image_only_metrics.txt")

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
WD = 1e-4
GRAD_ACCUM = 4
WARMUP_STEPS = 20
EARLY_STOPPING_PATIENCE = 3

# Use HF ViT model
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"

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

        return {
            "pixel_values": img,
            "labels": labels
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
                # Get binary label (1 for positive, -1 for negative, 0 for uncertain/not mentioned)
                chexpert_label = batch[cond_name][i]
                # Convert CheXpert labels (0-3) to binary (0-1) for this task
                # 0: Not mentioned -> 0
                # 1: Negative -> 0
                # 2: Uncertain -> 0 (could be different depending on policy)
                # 3: Positive -> 1
                label_value = 1 if chexpert_label == 3 else 0
                cond_vec[cond_idx] = label_value
                
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

    # Split into train, validation, and test sets
    # First split: 70% train, 30% temporary (for val+test)
    train_valtest_split = dataset.train_test_split(test_size=0.30, seed=SEED)
    train_dataset_hf_raw = train_valtest_split['train']
    val_test_dataset_hf_raw = train_valtest_split['test']

    # Second split: 50% validation, 50% test from the temporary split
    val_test_split = val_test_dataset_hf_raw.train_test_split(test_size=0.50, seed=SEED)
    val_dataset_hf_raw = val_test_split['train']
    test_dataset_hf_raw = val_test_split['test']

    logger.info(f"Data split complete: Train={len(train_dataset_hf_raw)}, Val={len(val_dataset_hf_raw)}, Test={len(test_dataset_hf_raw)}")

    # Apply label mapping
    logger.info("Applying label mapping to datasets...")
    train_dataset_hf = train_dataset_hf_raw.map(map_labels, batched=True, batch_size=1000, num_proc=4)
    val_dataset_hf = val_dataset_hf_raw.map(map_labels, batched=True, batch_size=1000, num_proc=4)
    test_dataset_hf = test_dataset_hf_raw.map(map_labels, batched=True, batch_size=1000, num_proc=4)

    # Create dataset objects
    train_dataset = CXRDataset(train_dataset_hf, split_name="train")
    val_dataset = CXRDataset(val_dataset_hf, split_name="val")
    test_dataset = CXRDataset(test_dataset_hf, split_name="test")

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
    logger.info("Calculating per-condition test metrics...")
    test_predictions = trainer.predict(test_dataset)
    # In ImageOnlyModel, predictions from trainer.predict() are the logits directly
    logits = test_predictions.predictions 
    labels = test_predictions.label_ids
    
    # Ensure logits and labels are numpy arrays
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
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
