import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import timm
from torchvision import transforms
from PIL import Image

from transformers import (
    Trainer, TrainingArguments, 
    EvalPrediction, DefaultDataCollator,
    set_seed, EarlyStoppingCallback
)

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)
from tabulate import tabulate

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
# Correct paths to the dataset files
TRAIN_CSV    = "dataset_splits/train.csv"
VAL_CSV      = "dataset_splits/val.csv" 
TEST_CSV     = "dataset_splits/test.csv"
IMG_DIR      = os.path.join("output", "img_png")
OUTPUT_DIR   = "output_image_only_trainer"
EXP_NAME    = "image_vit_base_patch16_224"
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-4
WD           = 1e-4
GRAD_ACCUM   = 4
WARMUP_STEPS = 100
EARLY_STOPPING_PATIENCE = 3

SEED         = 42
set_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

#########################
# === DATASET DEF ===   #
#########################
class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train=False):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5) if is_train else nn.Identity(),
            transforms.RandomRotation(3) if is_train else nn.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img_path = os.path.join(IMG_DIR, Path(row["image_path"]).stem + ".png")
            img = Image.open(img_path).convert("RGB")
            img = self.img_tf(img)
        except Exception:
            img = torch.zeros(3, 224, 224)
        labels = torch.tensor(row["cond_vec"], dtype=torch.float32)
        return {
            "pixel_values": img,
            "labels": labels
        }

###############################
# === LOAD DATA ===   #
###############################
def make_cond_vec(cs: str):
    if pd.isna(cs) or cs == "":
        return [0] * len(CONDITIONS)
    if isinstance(cs, list):
        conds = cs
    elif isinstance(cs, str):
        if '|' in cs:
            conds = [c.strip() for c in cs.split('|')]
        elif ',' in cs:
            conds = [c.strip() for c in cs.split(',')]
        else:
            conds = [cs.strip()]
    else:
        conds = []
    vec = [0] * len(CONDITIONS)
    for c in conds:
        if c in cond2idx:
            vec[cond2idx[c]] = 1
    return vec

# Load pre-split datasets
# print("Loading pre-split datasets...")
# print(f"Train CSV path: {TRAIN_CSV}")
# print(f"Val CSV path: {VAL_CSV}")
# print(f"Test CSV path: {TEST_CSV}")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

# print(f"Train samples: {len(train_df)}")
# print(f"Validation samples: {len(val_df)}")
# print(f"Test samples: {len(test_df)}")

# Create condition vectors for each dataset
train_df["cond_vec"] = train_df["condition"].apply(make_cond_vec)
val_df["cond_vec"] = val_df["condition"].apply(make_cond_vec)
test_df["cond_vec"] = test_df["condition"].apply(make_cond_vec)

# Create dataset objects
train_dataset = CXRDataset(train_df, is_train=True)
val_dataset = CXRDataset(val_df, is_train=False)
test_dataset = CXRDataset(test_df, is_train=False)

#########################
# === MODEL DEF ===     #
#########################
class ImageOnlyModel(nn.Module):
    def __init__(self, num_conditions:int):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, img_size=224, num_classes=0)
        self.classifier = nn.Linear(self.vit.num_features, num_conditions)

    def forward(self, pixel_values=None, labels=None):
        img_feat = self.vit(pixel_values)
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
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    return {
        "auc": mean_auc,
        "f1": f1,
        "recall": recall,
        "precision": precision
    }

#############################
# === TRAIN ===            #
#############################
def main():
    model = ImageOnlyModel(len(CONDITIONS))
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
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DefaultDataCollator(),
        callbacks=callbacks
    )
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, f"model_{EXP_NAME}"))
    test_metrics = trainer.evaluate(test_dataset)

    # Get predictions for test set for detailed per-condition metrics
    print("Calculating per-condition test metrics...") # Using print as logger is not set up here
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
    print("\nTest Set Metrics per Condition:")
    print("-" * 80)
    headers = ["Condition", "Precision", "Recall", "F1-score", "Support", "AUC"]
    rows = []
    
    for i, condition in enumerate(CONDITIONS):
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
            f"{precision:.3f}",
            f"{recall:.3f}",
            f"{f1:.3f}",
            f"{int(support)}",
            f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        ])
    
    # Add macro average from trainer.evaluate result (which uses compute_metrics)
    rows.append([
        "Macro Avg",
        f"{test_metrics['eval_precision']:.3f}",
        f"{test_metrics['eval_recall']:.3f}",
        f"{test_metrics['eval_f1']:.3f}",
        f"{len(test_dataset)} (Total Samples)", # Total samples as support for macro
        f"{test_metrics['eval_auc']:.3f}"
    ])
    
    table_str = tabulate(rows, headers=headers, tablefmt="grid")
    print("\n" + table_str)
    
    # Save the table to a text file in OUTPUT_DIR
    with open(os.path.join(OUTPUT_DIR, "train_image_only_metrics_table.txt"), "w") as f:
        f.write(table_str)

    with open(os.path.join(OUTPUT_DIR, "metrics_image_only.json"), "w") as f:
        json.dump({
            "test_auc": test_metrics.get("eval_auc", 0.0),
            "test_f1": test_metrics.get("eval_f1", 0.0),
            "test_recall": test_metrics.get("eval_recall", 0.0),
            "test_precision": test_metrics.get("eval_precision", 0.0)
        }, f, indent=4)
    return test_metrics

if __name__=="__main__":
    main()
