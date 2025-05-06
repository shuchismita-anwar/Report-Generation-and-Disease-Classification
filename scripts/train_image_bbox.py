import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.model_selection import train_test_split
from tabulate import tabulate

import math
# import logging

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger('train_image_bbox')

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
DATA_CSV     = "final_dataset_fixed.csv"
IMG_DIR      = os.path.join("output", "img_png")
BBOX_DIR     = os.path.join("output", "bbox_mask")
OUTPUT_DIR   = "output_image_bbox"

BATCH_SIZE   = 32 # Adjusted based on potential memory increase
GRAD_ACCUM   = 4
LR           = 1e-4
WD           = 1e-4
EPOCHS       = 20
GRAD_CLIP    = 1.0
WARMUP_STEPS = 100

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

#########################
# === DATASET DEF ===   #
#########################
class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str):
        # logger.info(f"Initializing {split} dataset with {len(df[df['split'] == split])} samples")
        self.df = df[df["split"] == split].reset_index(drop=True)
        # Filter out rows without bbox
        self.df = self.df[self.df["bbox_path"].notna()].reset_index(drop=True)
        # logger.info(f"After filtering for bbox: {len(self.df)} samples")

        self.img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5) if split=="train" else nn.Identity(),
            transforms.RandomRotation(3) if split=="train" else nn.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Image ---
        try:
            img_path = os.path.join(IMG_DIR, Path(row["image_path"]).stem + ".png")
            img = Image.open(img_path).convert("RGB")
            img = self.img_tf(img)
        except Exception as e:
            # logger.error(f"Error loading image {row['image_path']}: {e}")
            img = torch.zeros(3, 224, 224)
        
        # --- Bbox Mask ---
        try:
            mask_path = os.path.join(BBOX_DIR, Path(row["bbox_path"]).stem + ".png")
            mask = Image.open(mask_path).convert("L") # Grayscale
            mask = self.mask_tf(mask)
        except Exception as e:
            # logger.error(f"Error loading mask {row['bbox_path']}: {e}")
            mask = torch.zeros(1, 224, 224)
        
        # --- Condition Vector ---
        y = torch.tensor(row["cond_vec"], dtype=torch.float32)
        
        return {"img": img, "bbox": mask, "labels": y}

###############################
# === LOAD & SPLIT DATA ===   #
###############################
df = pd.read_csv(DATA_CSV)

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

df["cond_vec"] = df["condition"].apply(make_cond_vec)

if "split" not in df.columns:
    # logger.info("Creating train/val/test split")
    # Stratified split by patient_folder using at least one label as stratification
    # Use the first positive label in cond_vec for stratification
    df["stratify_label"] = df["cond_vec"].apply(lambda x: np.argmax(x) if np.sum(x) > 0 else -1)
    # Group by patient_folder and assign the most common label to each patient
    patient_labels = df.groupby("patient_folder")["stratify_label"].agg(lambda x: x[x!=-1].mode()[0] if any(x!=-1) else -1)
    patients = patient_labels.index.tolist()
    labels = patient_labels.values

    # Ensure labels are not all -1 before stratifying
    valid_indices = [i for i, l in enumerate(labels) if l != -1]
    if not valid_indices:
        # logger.warning("No valid labels found for stratification. Falling back to random split.")
        patients = df["patient_folder"].unique().tolist()
        np.random.shuffle(patients)
        n = len(patients)
        cut1, cut2 = int(0.7*n), int(0.85*n)
        train_patients = patients[:cut1]
        val_patients = patients[cut1:cut2]
        test_patients = patients[cut2:]
    else:
        patients_to_split = [patients[i] for i in valid_indices]
        labels_to_split = [labels[i] for i in valid_indices]
        # First split train vs temp (val+test)
        train_patients, temp_patients, _, _ = train_test_split(
            patients_to_split, labels_to_split, test_size=0.3, random_state=SEED, stratify=labels_to_split
        )
        # Then split temp into val and test
        val_patients, test_patients, _, _ = train_test_split(
            temp_patients, labels_to_split[len(train_patients):], # Use corresponding subset of labels for second split stratification
            test_size=0.5, random_state=SEED, stratify=labels_to_split[len(train_patients):]
        )
        # Add back patients with no labels (-1), distributing them randomly
        no_label_patients = [p for p, l in zip(patients, labels) if l == -1]
        np.random.shuffle(no_label_patients)
        n_no_label = len(no_label_patients)
        cut1_no_label = int(0.7 * n_no_label)
        cut2_no_label = int(0.85 * n_no_label)
        train_patients.extend(no_label_patients[:cut1_no_label])
        val_patients.extend(no_label_patients[cut1_no_label:cut2_no_label])
        test_patients.extend(no_label_patients[cut2_no_label:])

    df.loc[df["patient_folder"].isin(train_patients), "split"] = "train"
    df.loc[df["patient_folder"].isin(val_patients), "split"] = "val"
    df.loc[df["patient_folder"].isin(test_patients), "split"] = "test"
    if "stratify_label" in df.columns: df.drop(columns=["stratify_label"], inplace=True)

# Create Datasets and DataLoaders
train_dataset = CXRDataset(df, "train")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=4, pin_memory=True, prefetch_factor=2)

val_dataset = CXRDataset(df, "val")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, # Increase batch size for validation
                       num_workers=4, pin_memory=True, prefetch_factor=2)

test_dataset = CXRDataset(df, "test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
                        num_workers=4, pin_memory=True, prefetch_factor=2)

#########################
# === MODEL DEF ===     #
#########################
class ImageBboxModel(nn.Module):
    def __init__(self, num_conditions:int):
        super().__init__()
        # logger.info(f"Initializing model with {num_conditions} conditions")
        # Load pre-trained ViT model
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Identity()
        # Bbox mask encoder
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7,7)),
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        # Fusion + classifier
        self.fusion = nn.Sequential(
            nn.Linear(self.vit.num_features + 128, 768),
            nn.LayerNorm(768),
            nn.ReLU()
        )
        self.classifier = nn.Linear(768, num_conditions)

    def forward(self, img=None, bbox=None, labels=None, **kwargs):
        # Accepts img, bbox as separate keyword arguments (from Trainer)
        if img is None or bbox is None:
            raise ValueError("img and bbox must be provided")
        x = self.vit(img)
        m = self.mask_encoder(bbox)
        fused = self.fusion(torch.cat([x, m], dim=1))
        logits = self.classifier(fused)
        output = {"logits": logits}
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            output["loss"] = loss
        return output

from transformers import Trainer, TrainingArguments

def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(int)
    aucs = []
    for i in range(labels.shape[1]):
        try:
            if len(np.unique(labels[:,i])) > 1:
                aucs.append(roc_auc_score(labels[:,i], probs[:,i]))
            else:
                aucs.append(np.nan)
        except Exception:
            aucs.append(np.nan)
    mean_auc = np.nanmean(aucs)
    f1 = f1_score(labels, y_pred, average="macro", zero_division=0)
    rec = recall_score(labels, y_pred, average="macro", zero_division=0)
    prec = precision_score(labels, y_pred, average="macro", zero_division=0)
    return {"auc": mean_auc, "f1": f1, "recall": rec, "precision": prec}

#################
# === Main ===  #
#################
def main():
    # logger.info("Starting training process (Transformers Trainer)")
    # logger.info(f"Using device: {DEVICE}")
    model = ImageBboxModel(len(CONDITIONS))
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
        logging_steps=10,
        evaluation_strategy="epoch",
        # eval_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        # fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # logger.info("=== Final Test Evaluation ===")
    test_metrics = trainer.evaluate(test_dataset)
    
    # Get predictions for test set
    test_predictions = trainer.predict(test_dataset)
    logits = test_predictions.predictions
    labels = test_predictions.label_ids
    
    # Calculate probabilities and predictions
    probs = 1 / (1 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(int)
    
    # Calculate metrics for each condition
    print("\nTest Set Metrics per Condition:")
    print("-" * 80)
    headers = ["Condition", "Precision", "Recall", "F1-score", "Support", "AUC"]
    rows = []
    
    for i, condition in enumerate(CONDITIONS):
        # Calculate metrics for this condition
        precision = precision_score(labels[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(labels[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(labels[:, i], y_pred[:, i], zero_division=0)
        support = np.sum(labels[:, i])
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i]) if len(np.unique(labels[:, i])) > 1 else np.nan
        except:
            auc = np.nan
            
        rows.append([
            condition,
            f"{precision:.3f}",
            f"{recall:.3f}",
            f"{f1:.3f}",
            f"{int(support)}",
            f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        ])
    
    # Add macro average
    rows.append([
        "Macro Avg",
        f"{test_metrics['eval_precision']:.3f}",
        f"{test_metrics['eval_recall']:.3f}",
        f"{test_metrics['eval_f1']:.3f}",
        f"{len(test_dataset)}",
        f"{test_metrics['eval_auc']:.3f}"
    ])
    
    # Print the table
    table_str = tabulate(rows, headers=headers, tablefmt="grid")
    print(table_str)
    
    # Save the table to a text file in OUTPUT_DIR
    with open(os.path.join(OUTPUT_DIR, "train_image_bbox_metrics_table.txt"), "w") as f:
        f.write(table_str)
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, "metrics_img_bbox.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
    print("\nMetrics saved to:", os.path.join(OUTPUT_DIR, "metrics_img_bbox.json"))
    # logger.info("Training complete")

if __name__=="__main__":
    main()
