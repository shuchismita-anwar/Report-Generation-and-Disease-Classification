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
from tabulate import tabulate

import math

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
TRAIN_CSV    = "dataset_splits/train.csv"
VAL_CSV      = "dataset_splits/val.csv"
TEST_CSV     = "dataset_splits/test.csv"
IMG_DIR      = os.path.join("output", "img_png")
BBOX_DIR     = os.path.join("output", "bbox_mask")
OUTPUT_DIR   = "output_image_bbox"
EXP_NAME     = "image_bbox_vit_base_patch16_224"

BATCH_SIZE   = 32
GRAD_ACCUM   = 4
LR           = 1e-4
WD           = 1e-4
EPOCHS       = 100
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
    def __init__(self, df: pd.DataFrame, is_train=False):
        self.df = df.reset_index(drop=True)
        # Filter out rows without bbox
        self.df = self.df[self.df["bbox_path"].notna()].reset_index(drop=True)
        
        self.img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5) if is_train else nn.Identity(),
            transforms.RandomRotation(3) if is_train else nn.Identity(),
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
            img = torch.zeros(3, 224, 224)
        
        # --- Bbox Mask ---
        try:
            mask_path = os.path.join(BBOX_DIR, Path(row["bbox_path"]).stem + ".png")
            mask = Image.open(mask_path).convert("L") # Grayscale
            mask = self.mask_tf(mask)
        except Exception as e:
            mask = torch.zeros(1, 224, 224)
        
        # --- Condition Vector ---
        y = torch.tensor(row["cond_vec"], dtype=torch.float32)
        
        return {"img": img, "bbox": mask, "labels": y}

###############################
# === LOAD DATA ===         #
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

# Create Datasets and DataLoaders
train_dataset = CXRDataset(train_df, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=4, pin_memory=True, prefetch_factor=2)

val_dataset = CXRDataset(val_df, is_train=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, # Increase batch size for validation
                       num_workers=4, pin_memory=True, prefetch_factor=2)

test_dataset = CXRDataset(test_df, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
                        num_workers=4, pin_memory=True, prefetch_factor=2)

# print(f"Train dataset size after filtering: {len(train_dataset)}")
# print(f"Val dataset size after filtering: {len(val_dataset)}")
# print(f"Test dataset size after filtering: {len(test_dataset)}")

#########################
# === MODEL DEF ===     #
#########################
class ImageBboxModel(nn.Module):
    def __init__(self, num_conditions:int):
        super().__init__()
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
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, f"model_{EXP_NAME}"))
    print("=== Final Test Evaluation ===")
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
        f"{test_metrics['test_precision']:.3f}",
        f"{test_metrics['test_recall']:.3f}",
        f"{test_metrics['test_f1']:.3f}",
        f"{len(test_dataset)}",
        f"{test_metrics['test_auc']:.3f}"
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
    print("Training complete")

if __name__=="__main__":
    main()
