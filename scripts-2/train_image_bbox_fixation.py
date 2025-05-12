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

# Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger('train_image_bbox_fixation')

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
FIXSTATS_NPZ = "fixstats.npz"
IMG_DIR      = os.path.join("output", "img_png")
BBOX_DIR     = os.path.join("output", "bbox_mask")
FIXSEQ_DIR   = os.path.join("output", "fix_seq")
OUTPUT_DIR   = "output_image_bbox_fixation"

BATCH_SIZE   = 32 # Adjusted based on potential memory increase
GRAD_ACCUM   = 4
LR           = 1e-4
WD           = 1e-4
EPOCHS       = 20
EARLY_STOPPING_PATIENCE = 3  # Early stopping patience
GRAD_CLIP    = 1.0
WARMUP_STEPS = 100

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

#############################
# === LOAD & FIX STATS ===  #
#############################
try:
    stats = np.load(FIXSTATS_NPZ)
    MU, SIGMA = stats["mu"], stats["sigma"]
    # guard
    SIGMA = SIGMA + 1e-6
    # logger.info(f"Loaded fixation stats - Mean: {MU}, Sigma: {SIGMA}")
except Exception as e:
    # logger.error(f"Error loading fixation stats: {e}")
    # logger.warning("Using default fixation stats")
    MU, SIGMA = np.zeros(4), np.ones(4)

#########################
# === DATASET DEF ===   #
#########################
class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str):
        # logger.info(f"Initializing {split} dataset with {len(df[df['split'] == split])} samples")
        self.df = df[df["split"] == split].reset_index(drop=True)
        # Filter out rows without bounding box or fixation sequence information
        self.df = self.df[
            self.df["bbox_path"].notna() & 
            self.df["fixations_path"].notna()
        ].reset_index(drop=True)
        # logger.info(f"After filtering for bbox and fixations availability: {len(self.df)} samples")
        
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
        
        # --- BBox mask ---
        try:
            bbox_path = os.path.join(BBOX_DIR, Path(row["bbox_path"]).stem + ".png")
            bbox = Image.open(bbox_path).convert("L")
            bbox = self.mask_tf(bbox)
        except Exception as e:
            # logger.error(f"Error loading mask {row['bbox_path']}: {e}")
            bbox = torch.zeros(1, 224, 224)
            
        # --- Fixation Sequence ---
        try:
            fix_path = os.path.join(FIXSEQ_DIR, Path(row["fixations_path"]).stem + ".npz")
            fix_data = np.load(fix_path)
            seq = fix_data["sequence"]
            fix_mask = fix_data["mask"]
            
            # Normalize sequence using precomputed stats
            seq = (seq - MU) / SIGMA
            
            # Convert to tensors and apply safety clipping
            seq = torch.from_numpy(seq).float()
            seq = seq.clamp(-10.0, 10.0)
            fix_mask = torch.from_numpy(fix_mask)
        except Exception as e:
            # logger.error(f"Error loading fixation sequence {row['fixations_path']}: {e}")
            seq = torch.zeros(128, 4)  # Assuming max length of 128
            fix_mask = torch.zeros(128, dtype=torch.bool)
        
        # --- Condition Vector ---
        y = torch.tensor(row["cond_vec"], dtype=torch.float32)
        
        return {"img": img, "bbox": bbox, "fix_seq": seq, "fix_mask": fix_mask, "labels": y}

###############################
# === LOAD & SPLIT DATA ===   #
###############################
df = pd.read_csv(DATA_CSV)

def make_cond_vec(cs):
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
        patients_list = df["patient_folder"].unique().tolist()
        np.random.shuffle(patients_list)
        n = len(patients_list)
        cut1, cut2 = int(0.7*n), int(0.85*n)
        train_patients = patients_list[:cut1]
        val_patients = patients_list[cut1:cut2]
        test_patients = patients_list[cut2:]
    else:
        patients_to_split = [patients[i] for i in valid_indices]
        labels_to_split = [labels[i] for i in valid_indices]
        
        # Adjust test_size calculation for stratification
        temp_size = 0.3 # 30% for val+test
        test_size_in_temp = 0.5 # 50% of temp is test -> 15% of total
        
        # Stratify requires at least 2 members per class, handle potential errors
        try:
            train_patients, temp_patients, _, temp_labels_split = train_test_split(
                patients_to_split, labels_to_split, test_size=temp_size, random_state=SEED, stratify=labels_to_split
            )
        except ValueError as e:
            # logger.warning(f"Stratification failed on first split: {e}. Falling back to random split for this part.")
            train_patients, temp_patients, _, temp_labels_split = train_test_split(
                patients_to_split, labels_to_split, test_size=temp_size, random_state=SEED
            )

        try:
            # Ensure temp_labels_split corresponds to temp_patients for stratification
             val_patients, test_patients, _, _ = train_test_split(
                 temp_patients, temp_labels_split, test_size=test_size_in_temp, random_state=SEED, stratify=temp_labels_split
             )
        except ValueError as e:
            # logger.warning(f"Stratification failed on second split: {e}. Falling back to random split for this part.")
             val_patients, test_patients, _, _ = train_test_split(
                 temp_patients, temp_labels_split, test_size=test_size_in_temp, random_state=SEED
             )

        # Add back patients with no labels (-1), distributing them proportionally
        no_label_patients = [p for p, l in zip(patients, labels) if l == -1]
        np.random.shuffle(no_label_patients)
        n_no_label = len(no_label_patients)
        n_train = len(train_patients)
        n_val = len(val_patients)
        n_test = len(test_patients)
        total_stratified = n_train + n_val + n_test

        # Distribute no_label_patients based on the split ratios derived from stratified part
        train_ratio = n_train / total_stratified if total_stratified > 0 else 0.7
        val_ratio = n_val / total_stratified if total_stratified > 0 else 0.15

        cut1_no_label = int(train_ratio * n_no_label)
        cut2_no_label = cut1_no_label + int(val_ratio * n_no_label)
        
        train_patients.extend(no_label_patients[:cut1_no_label])
        val_patients.extend(no_label_patients[cut1_no_label:cut2_no_label])
        test_patients.extend(no_label_patients[cut2_no_label:])

    df.loc[df["patient_folder"].isin(train_patients), "split"] = "train"
    df.loc[df["patient_folder"].isin(val_patients), "split"] = "val"
    df.loc[df["patient_folder"].isin(test_patients), "split"] = "test"
    if "stratify_label" in df.columns: df.drop(columns=["stratify_label"], inplace=True)

train_dataset = CXRDataset(df, "train")
val_dataset = CXRDataset(df, "val")
test_dataset = CXRDataset(df, "test")

#########################
# === MODEL DEF ===     #
#########################
class ImageBboxFixationModel(nn.Module):
    def __init__(self, num_conditions:int):
        super().__init__()
        # logger.info(f"Initializing model with {num_conditions} conditions")
        # Load pre-trained ViT model
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, img_size=224, num_classes=0)
        
        # Process bounding box masks
        self.bbox_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Process fixation sequences
        self.fix_emb = nn.Linear(4, 64)  # Embedding for each fixation point
        self.fix_gru = nn.GRU(64, 256, batch_first=True, bidirectional=True)
        
        # Final GRU dimensions will be 2*256 = 512
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(768 + 128 + 512, 768),  
            nn.LayerNorm(768),
            nn.ReLU()
        )
        
        # Classifier head
        self.classifier = nn.Linear(768, num_conditions)

    def forward(self, img=None, bbox=None, fix_seq=None, fix_mask=None, labels=None, **kwargs):
        # Extract image features
        img_feat = self.vit(img)
        
        # Extract bbox features
        bbox_feat = self.bbox_encoder(bbox)
        bbox_feat = bbox_feat.flatten(1)
        
        # Process fixation sequences
        batch_size, seq_len = fix_seq.size(0), fix_seq.size(1)
        
        # Apply embedding to each fixation point
        fix_emb = self.fix_emb(fix_seq)  # [B, L, 64]
        
        # Pack for efficient GRU processing
        fix_lens = fix_mask.sum(dim=1).cpu()
        
        if torch.all(fix_lens == 0):
            fix_feat = torch.zeros(batch_size, 512, device=img.device)
        else:
            # Pack padded sequence
            packed_seq = nn.utils.rnn.pack_padded_sequence(
                fix_emb, 
                fix_lens.clamp(min=1), # Ensure at least length 1
                batch_first=True, 
                enforce_sorted=False
            )
            
            # Run through GRU
            _, h_n = self.fix_gru(packed_seq)
            # h_n shape: [2, B, 256] (2 for bidirectional)
            
            # Concatenate directions
            fix_feat = h_n.transpose(0, 1).reshape(batch_size, -1)
        
        # Fuse features
        combined = torch.cat([img_feat, bbox_feat, fix_feat], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        
        output = {"logits": logits}
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            output["loss"] = loss
        return output

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

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

def main():
    print("Starting training process (Transformers Trainer)")
    model = ImageBboxFixationModel(len(CONDITIONS))
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
    # Add early stopping callback with patience of 3
    callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    trainer.train()
    print("=== Test ===")
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
    print("\n" + table_str)
    
    # Save the table to a text file in OUTPUT_DIR
    with open(os.path.join(OUTPUT_DIR, "train_image_bbox_fixation_metrics_table.txt"), "w") as f:
        f.write(table_str)
    
    print(f"Test metrics: {test_metrics}")
    with open(os.path.join(OUTPUT_DIR, "metrics_image_bbox_fixation.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
    print("Metrics saved to:", os.path.join(OUTPUT_DIR, "metrics_image_bbox_fixation.json"))
    print("Training complete")

if __name__=="__main__":
    main()
