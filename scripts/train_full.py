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
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, EvalPrediction, PreTrainedModel
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
import safetensors

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.model_selection import train_test_split

import math
import logging
from tabulate import tabulate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_full')

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
OUTPUT_DIR   = "output_full"

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
        # Filter out rows without any needed component
        self.df = self.df[
            self.df["bbox_path"].notna() & 
            self.df["fixations_path"].notna() & 
            self.df["transcript_path"].notna()
        ].reset_index(drop=True)
        # logger.info(f"After filtering for all components: {len(self.df)} samples")
        
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
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
            # Return a blank image if we can't load
            img = torch.zeros(3, 224, 224)
        
        # --- BBox mask ---
        try:
            mask_path = os.path.join(BBOX_DIR, Path(row["bbox_path"]).stem + ".png")
            bbox = Image.open(mask_path).convert("L")
            bbox = self.mask_tf(bbox)
        except Exception as e:
            # logger.error(f"Error loading mask {row['bbox_path']}: {e}")
            # Return a blank mask if we can't load
            bbox = torch.zeros(1, 224, 224)
        
        # --- Fixation sequence ---
        try:
            fix_path = os.path.join(FIXSEQ_DIR, Path(row["fixations_path"]).stem + ".npz")
            arr = np.load(fix_path)
            seq = arr["seq"].astype(np.float32)  # (L,4)
            mask = arr["mask"].astype(bool)      # (L,)
            
            # Safely handle any remaining NaNs
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Convert to tensors and apply safety clipping
            seq = torch.from_numpy(seq).float()
            seq = seq.clamp(-10.0, 10.0)
            mask = torch.from_numpy(mask)
        except Exception as e:
            # logger.error(f"Error loading fixation sequence {row['fixations_path']}: {e}")
            # Return empty fixation sequence if we can't load
            seq = torch.zeros(128, 4)  # Assuming max length of 128
            mask = torch.zeros(128, dtype=torch.bool)
        
        # --- Transcript tokens ---
        try:
            with open(row["transcript_path"], "r") as f:
                transcript_data = json.load(f)
                full_text = transcript_data["full_text"].lower()
            
            tokens = self.tokenizer(
                full_text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
        except Exception as e:
            # logger.error(f"Error loading transcript {row['transcript_path']}: {e}")
            # Return empty transcript if we can't load
            input_ids = torch.zeros(128, dtype=torch.long)
            attention_mask = torch.zeros(128, dtype=torch.long)
        
        # --- Condition Vector ---
        labels = torch.tensor(row["cond_vec"], dtype=torch.float32)
        
        return {
            "img": img, 
            "bbox": bbox, 
            "fix_seq": seq, 
            "fix_mask": mask, 
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

###############################
# === LOAD & SPLIT DATA ===   #
###############################
df = pd.read_csv(DATA_CSV)

def make_cond_vec(cs: str):
    if pd.isna(cs) or cs == "":
        return [0] * len(CONDITIONS)
    cs = cs.split('|') if isinstance(cs, str) else cs
    vec = [0] * len(CONDITIONS)
    for c in cs:
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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True, prefetch_factor=2)

val_dataset = CXRDataset(df, "val")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=4, pin_memory=True, prefetch_factor=2)

test_dataset = CXRDataset(df, "test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, prefetch_factor=2)

#################################
# Custom Trainer for Contiguous Saving #
#################################
class ContiguousSavingTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"ContiguousSavingTrainer: Saving model checkpoint to {output_dir}")

        if isinstance(self.model, PreTrainedModel):
            # Base class handles PreTrainedModel correctly, including contiguous tensors for safetensors.
            super()._save(output_dir, state_dict=state_dict)
        else:
            # For generic nn.Module, explicitly ensure tensors are contiguous.
            model_to_save = self.model
            
            if state_dict is None:
                current_state_dict = model_to_save.state_dict()
            else:
                current_state_dict = state_dict

            contiguous_state_dict = {}
            for k, v in current_state_dict.items():
                if isinstance(v, torch.Tensor):
                    contiguous_state_dict[k] = v.contiguous()
                else:
                    contiguous_state_dict[k] = v
            
            # Directly save the contiguous state dictionary
            if self.args.save_safetensors:
                safetensors.torch.save_file(
                    contiguous_state_dict, 
                    os.path.join(output_dir, SAFE_WEIGHTS_NAME)
                )
            else:
                torch.save(
                    contiguous_state_dict, 
                    os.path.join(output_dir, WEIGHTS_NAME)
                )

            # Save tokenizer (if any)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            # Save training arguments
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

#########################
# === MODEL DEF ===     #
#########################
class MultiModalModel(nn.Module):
    def __init__(self, num_conditions:int):
        super().__init__()
        # logger.info(f"Initializing model with {num_conditions} conditions")
        
        # Image encoder (Vision Transformer)
        self.vit = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True, num_classes=0) # num_classes=0 for feature extraction
        self.img_proj = nn.Linear(768, 768) # Project to common dimension
        
        # Text encoder (BioClinicalBERT)
        self.txt_enc = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.txt_proj = nn.Linear(768, 768) # Project to common dimension
        
        # BBox map encoder (Simple CNN)
        self.bbox_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1,4)) # Output [B, 32, 1, 4]
        )
        # Output flattened will be 32*4 = 128
        
        # Fixation sequence encoder (GRU)
        self.fix_emb = nn.Linear(4, 64)  # Embed (x,y,t_start,t_dur) -> 64 dims
        self.fix_gru = nn.GRU(64, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.fix_proj = nn.Linear(1024, 768) # (num_layers * num_directions * hidden_size) = 2 * 2 * 256 = 1024 -> 768
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(768 + 768 + 768 + 128, 768),
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        
        # Classifier head
        self.classifier = nn.Linear(768, num_conditions)
        
        # Loss function for multi-label classification
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, img, bbox, fix_seq, fix_mask, input_ids, attention_mask, labels=None):
        # --- Image encoding ---
        img_feat = self.vit(img)  # [B, 768]
        img_proj = self.img_proj(img_feat)  # [B, 768]
        
        # --- Text encoding ---
        txt_outputs = self.txt_enc(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        txt_feat = txt_outputs.last_hidden_state[:, 0, :]  # [B, 768] (CLS token)
        txt_proj = self.txt_proj(txt_feat)  # [B, 768]
        
        # --- BBox encoding ---
        bbox_feat = self.bbox_encoder(bbox)
        bbox_feat = bbox_feat.flatten(1)  # [B, 128]
        
        # --- Fixation sequence encoding ---
        # fix_seq from batch
        # fix_mask from batch
        batch_size, seq_len = fix_seq.size(0), fix_seq.size(1)
        
        # Apply embedding to each fixation point
        fix_emb = self.fix_emb(fix_seq)  # [B, L, 64]
        
        # Pack for efficient GRU processing
        fix_lens = fix_mask.sum(dim=1).cpu()
        
        if torch.all(fix_lens == 0):
            # Handle the case where all sequences are empty
            fix_feat = torch.zeros(batch_size, 1024, device=fix_emb.device)
        else:
            # Process with GRU
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
            fix_feat = h_n.transpose(0, 1).reshape(batch_size, -1)  # [B, 512]
        
        fix_proj = self.fix_proj(fix_feat)  # [B, 768]
        
        # --- Fuse all modalities ---
        combined = torch.cat([img_proj, txt_proj, fix_proj, bbox_feat], dim=1)
        fused = self.fusion(combined)
        
        # --- Classification ---
        logits_output = self.classifier(fused)
        
        # Create attention map between image and fixations (not used for classification)
        attn_map_output = torch.zeros_like(bbox)  # Placeholder
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits_output, labels)

        # Hugging Face Trainer expects a dict or a ModelOutput object.
        output = {
            "logits": logits_output,
            # The following are custom outputs, Trainer might ignore them unless specified for metrics or other callbacks
            "zi": img_proj,  # image projection for contrastive loss
            "zt": txt_proj,  # text projection for contrastive loss
            "zg": fix_proj,  # fixation projection for contrastive loss
            "attn_map": attn_map_output  # attention map for visualization
        }
        if loss is not None:
            output["loss"] = loss

        # If labels are provided, the Trainer will compute the loss outside.
        # Or, we can compute loss here if preferred.
        # For simplicity with Trainer, usually just return logits and let Trainer handle loss if labels are present.
        return output

def compute_metrics(eval_pred: EvalPrediction):
    # eval_pred.predictions is a tuple (logits, zi, zt, zg, attn_map)
    # eval_pred.label_ids are the labels
    
    predictions_tuple = eval_pred.predictions
    labels_raw = eval_pred.label_ids

    # The actual logits are the first element in the predictions tuple
    logits = predictions_tuple[0]
    
    # Ensure logits and labels are numpy arrays for sklearn metrics
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    elif not isinstance(logits, np.ndarray):
        logits = np.array(logits)

    if isinstance(labels_raw, torch.Tensor):
        labels = labels_raw.cpu().numpy()
    elif not isinstance(labels_raw, np.ndarray):
        labels = np.array(labels_raw)
    else: # already a numpy array
        labels = labels_raw

    # Assuming logits are raw scores, apply sigmoid for probabilities
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    preds = (probs > 0.5).astype(int)
    
    # Calculate metrics
    try:
        # For multi-label, 'samples' is a good average for f1, recall, precision
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        rec = recall_score(labels, preds, average="macro", zero_division=0)
        prec = precision_score(labels, preds, average="macro", zero_division=0)
        
        # Ensure labels and probs have the same shape
        if labels.shape == probs.shape:
            auc_scores = [roc_auc_score(labels[:, i], probs[:, i]) for i in range(labels.shape[1]) if len(np.unique(labels[:, i])) > 1]
            mean_auc = np.mean(auc_scores) if auc_scores else 0.0
        else:
            # logger.warning(f"Shape mismatch for AUC: labels {labels.shape}, probs {probs.shape}. AUC set to 0.")
            mean_auc = 0.0
            
    except Exception as e:
        # logger.error(f"Error calculating metrics: {e}")
        f1, rec, prec, mean_auc = 0.0, 0.0, 0.0, 0.0
        
    return {"auc": mean_auc, "f1": f1, "recall": rec, "precision": prec}


#################
# === Main ===  #
#################
def main():
    # logger.info("Starting training process (Transformers Trainer)")
    logger.info(f"Using device: {DEVICE}")
    model = MultiModalModel(len(CONDITIONS))
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
    trainer = ContiguousSavingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # logger.info("=== Test ===")
    test_metrics = trainer.evaluate(test_dataset)
    
    # Get predictions for test set for detailed per-condition metrics
    logger.info("Calculating per-condition test metrics...")
    test_predictions = trainer.predict(test_dataset)
    # In MultiModalModel, predictions are a tuple, logits are the first element
    logits = test_predictions.predictions[0] 
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
    
    logger.info("\n" + tabulate(rows, headers=headers, tablefmt="grid"))

    # logger.info(f"Test metrics: {test_metrics}")
    with open(os.path.join(OUTPUT_DIR, "metrics_full.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
    logger.info(f"Overall test metrics saved to {os.path.join(OUTPUT_DIR, 'metrics_full.json')}")
    # logger.info("Training complete")

if __name__=="__main__":
    main()
