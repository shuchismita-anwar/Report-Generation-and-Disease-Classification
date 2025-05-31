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

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, EvalPrediction, PreTrainedModel, EarlyStoppingCallback
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
import safetensors

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.model_selection import train_test_split
from tabulate import tabulate

import math
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_image_bbox')

############################
# 1) Fixed label list
############################
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Edema',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pneumonia', 'Support Devices'
]

# CONDITIONS = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
#                               'Fracture','Lung Lesion','Lung Opacity','No Finding', 'Pleural Effusion','Pleural Other',
#                               'Pneumonia','Pneumothorax','Support Devices']

cond2idx = {c: i for i, c in enumerate(CONDITIONS)}

#####################
# === CONSTANTS === #
#####################
TRAIN_CSV    = "dataset_splits/train.csv"
VAL_CSV      = "dataset_splits/val.csv"
TEST_CSV     = "dataset_splits/test.csv"

IMG_DIR      = os.path.join("output", "img_png")
BBOX_DIR     = os.path.join("output", "bbox_mask")

OUTPUT_DIR   = "output_multimodal_vit_features_image_bbox"

EXP_NAME     = "multimodal_vit_features_image_bbox"

BATCH_SIZE   = 32
GRAD_ACCUM   = 4
LR           = 1e-5  # Match the working transcript version
WD           = 1e-4
EPOCHS       = 30
EARLY_STOPPING_PATIENCE = 10
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
        # logger.info(f"Initializing {split} dataset with {len(df)} samples")
        # No need to filter by split since datasets are already split into separate files
        self.df = df.reset_index(drop=True)
        # Filter out rows without any needed component (only bbox requirement now)
        self.df = self.df[
            self.df["bbox_path"].notna()
        ].reset_index(drop=True)
        # logger.info(f"After filtering for bbox availability: {len(self.df)} samples")
        
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
            bbox_path = os.path.join(BBOX_DIR, Path(row["bbox_path"]).stem + ".png")
            bbox = Image.open(bbox_path).convert("L")
            bbox = self.mask_tf(bbox)
        except Exception as e:
            # logger.error(f"Error loading mask {row['bbox_path']}: {e}")
            # Return a blank mask if we can't load
            bbox = torch.zeros(1, 224, 224)
        
        # --- Condition Vector ---
        labels = torch.tensor(row["cond_vec"], dtype=torch.float32)
        
        return {
            "img": img, 
            "bbox": bbox, 
            "labels": labels
        }

###############################
# === LOAD & SPLIT DATA ===   #
###############################
# Load pre-split datasets
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

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

train_df["cond_vec"] = train_df["condition"].apply(make_cond_vec)
val_df["cond_vec"] = val_df["condition"].apply(make_cond_vec)
test_df["cond_vec"] = test_df["condition"].apply(make_cond_vec)

train_dataset = CXRDataset(train_df, split="train")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True, prefetch_factor=2)

val_dataset = CXRDataset(val_df, split="val")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=4, pin_memory=True, prefetch_factor=2)

test_dataset = CXRDataset(test_df, split="test")
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
    def __init__(self, num_conditions:int, contrastive_temperature:float=0.1):
        super().__init__()
        logger.info(f"Initializing model with {num_conditions} conditions using HF image encoder (IMAGE + BBOX ONLY).")
        self.contrastive_temperature = contrastive_temperature

        # --- Image encoder (Hugging Face) ---
        # hf_img_model_name = "microsoft/rad-dino"
        hf_img_model_name = "google/vit-base-patch16-224-in21k"
        logger.info(f"Loading image encoder from Hugging Face Hub: {hf_img_model_name}")
        try:
            # Load the model directly from Hugging Face Hub
            self.vit = AutoModel.from_pretrained(hf_img_model_name)
            # Get the feature dimension from the model's config
            img_feature_dim = self.vit.config.hidden_size
            logger.info(f"Loaded {hf_img_model_name}. Feature dimension: {img_feature_dim}")

        except Exception as e:
            logger.error(f"Failed to load {hf_img_model_name} from Hugging Face Hub: {e}")
            # Fallback using a standard HF ViT model
            hf_fallback_model_name = "google/vit-base-patch16-224-in21k"
            logger.warning(f"Falling back to default HF ViT: {hf_fallback_model_name}")
            try:
                self.vit = AutoModel.from_pretrained(hf_fallback_model_name)
                img_feature_dim = self.vit.config.hidden_size
            except Exception as fallback_e:
                logger.critical(f"FATAL: Failed to load fallback HF image model {hf_fallback_model_name}: {fallback_e}")
                # If even fallback fails, raise the error as the model is unusable
                raise fallback_e

        # Ensure projection layer input matches the feature dimension
        self.img_proj = nn.Linear(img_feature_dim, 768) # Project features to common dimension
        
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
        
        self.bbox_proj = nn.Linear(128, 768) # Project bbox features to common dimension
        
        # Fusion layer (Updated dimensions: img + bbox only)
        self.fusion = nn.Sequential(
            nn.Linear(768 + 768, 768), # img(768) + bbox(768)
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        
        # Classifier head
        self.classifier = nn.Linear(768, num_conditions)
        
        # Initialize classifier with small weights for better convergence
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
        nn.init.constant_(self.classifier.bias, 0.0)
        
        # Loss function for multi-label classification (no class weights like working version)
        self.loss_fn = nn.BCEWithLogitsLoss()

    # Helper method for InfoNCE Loss (moved inside the class)
    def info_nce_loss(self, features1, features2, temperature=None, eps=1e-6):
        """
        Calculates the InfoNCE loss for contrastive learning.
        Uses the model's stored temperature if not provided.
        Args:
            features1 (torch.Tensor): Batch of embeddings for the first view (e.g., images), shape (B, D).
            features2 (torch.Tensor): Batch of embeddings for the second view (e.g., text), shape (B, D).
            temperature (float, optional): Temperature scaling factor. Defaults to self.contrastive_temperature.
            eps (float): Small value to prevent division by zero.
        Returns:
            torch.Tensor: The InfoNCE loss.
        """
        temp = temperature if temperature is not None else self.contrastive_temperature
        
        if features1.shape[0] <= 1 or features2.shape[0] <= 1:
            return torch.tensor(0.0, device=features1.device, requires_grad=True)

        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        similarity_matrix = torch.matmul(features1, features2.T)
        similarity_matrix = torch.clamp(similarity_matrix, -1.0 + eps, 1.0 - eps)
        similarity_matrix = similarity_matrix / temp
        batch_size = features1.shape[0]
        labels = torch.arange(batch_size, device=features1.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def forward(self, img, bbox, labels=None, contrastive_weight=1.0):
        # --- Image encoding ---
        # Get image features from the Hugging Face model output
        # Option 1: Use pooler_output if available and suitable
        # Option 2: Use last_hidden_state and take the CLS token (index 0)
        img_outputs = self.vit(pixel_values=img)
        if hasattr(img_outputs, 'pooler_output') and img_outputs.pooler_output is not None:
            img_feat = img_outputs.pooler_output # [B, hidden_size]
        else:
            img_feat = img_outputs.last_hidden_state[:, 0, :] # [B, hidden_size] (CLS token)

        img_proj = self.img_proj(img_feat)  # [B, 768]
        
        # --- BBox encoding ---
        bbox_feat = self.bbox_encoder(bbox).flatten(1)  # [B, 128]
        bbox_proj = self.bbox_proj(bbox_feat) # [B, 768]
        
        # --- Fuse all modalities (IMAGE + BBOX ONLY) ---
        combined = torch.cat([img_proj, bbox_proj], dim=1)
        fused = self.fusion(combined)
        
        # --- Classification ---
        logits_output = self.classifier(fused)
        
        # Create attention map placeholder
        attn_map_output = torch.zeros_like(bbox)  # Placeholder
        
        loss = None
        loss_cls = torch.tensor(0.0, device=img.device) # Default to 0 if no labels
        loss_contrastive = torch.tensor(0.0, device=img.device)

        if labels is not None:
            # 1. Classification Loss
            loss_cls = self.loss_fn(logits_output, labels)

            # 2. Contrastive Loss (only if batch size > 1, IMAGE + BBOX ONLY)
            if img_proj.shape[0] > 1:
                loss_img_bbox = self.info_nce_loss(img_proj, bbox_proj)
                loss_contrastive = loss_img_bbox
            else:
                 # Log warning if batch size is 1 during training/eval with labels
                 # logger.warning("Batch size is 1, skipping contrastive loss calculation.")
                 loss_contrastive = torch.tensor(0.0, device=img.device)

            # 3. Combine Losses - make classification loss dominant
            loss = loss_cls + contrastive_weight * loss_contrastive

        # Hugging Face Trainer expects a dict or a ModelOutput object.
        output = {
            "logits": logits_output,
            # The following are custom outputs, Trainer might ignore them unless specified for metrics or other callbacks
            "zi": img_proj,  # image projection for contrastive loss
            "zb": bbox_proj,  # bbox projection for contrastive loss
            "attn_map": attn_map_output,  # attention map for visualization
            "loss_cls": loss_cls.detach() if loss is not None else None,  # Log separate losses
            "loss_con": loss_contrastive.detach() if loss is not None else None  # Log separate losses
        }
        if loss is not None:
            output["loss"] = loss

        # If labels are provided, the Trainer will compute the loss outside.
        # Or, we can compute loss here if preferred.
        # For simplicity with Trainer, usually just return logits and let Trainer handle loss if labels are present.
        return output

def compute_metrics(eval_pred: EvalPrediction):
    # eval_pred.predictions is a tuple/dict depending on model output
    # eval_pred.label_ids are the labels
    
    # Assuming the model output is a dictionary as defined above
    predictions_dict = eval_pred.predictions 
    labels_raw = eval_pred.label_ids

    # Extract logits (ensure they are the first element if predictions were somehow tupled)
    if isinstance(predictions_dict, dict):
         logits = predictions_dict['logits']
    elif isinstance(predictions_dict, tuple):
         # Fallback: assume logits are first if it became a tuple
         logits = predictions_dict[0] 
    else:
         # Handle unexpected format if necessary
         logits = predictions_dict # Or raise error
    
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
        # For multi-label, 'macro' is a good average for f1, recall, precision
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,

        metric_for_best_model="loss",
        greater_is_better=False,
        
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none"
    )
    # Add early stopping callback with patience
    callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    
    trainer = ContiguousSavingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    trainer.train()
    # logger.info("=== Test ===")
    trainer.save_model(os.path.join(OUTPUT_DIR, f"model_{EXP_NAME}"))
    print("=== Final Test Evaluation ===")
    test_metrics = trainer.evaluate(test_dataset)
    
    # Get predictions for test set for detailed per-condition metrics
    logger.info("Calculating per-condition test metrics...")
    test_predictions = trainer.predict(test_dataset)
    # In MultiModalModel, predictions are a dict, extract logits (match working version)
    predictions_dict = test_predictions.predictions
    if isinstance(predictions_dict, dict):
        logits = predictions_dict['logits']
    else:
        # Fallback: assume logits are first if it became a tuple
        logits = predictions_dict[0] 
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
    
    table_str = tabulate(rows, headers=headers, tablefmt="grid")
    logger.info("\n" + table_str)
    
    # Save the table to a text file in OUTPUT_DIR
    with open(os.path.join(OUTPUT_DIR, "multimodal_image_bbox_metrics_table.txt"), "w") as f:
        f.write(table_str)
    
    # logger.info(f"Test metrics: {test_metrics}")
    with open(os.path.join(OUTPUT_DIR, "metrics_image_bbox.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
    logger.info(f"Overall test metrics saved to {os.path.join(OUTPUT_DIR, 'metrics_image_bbox.json')}")
    # logger.info("Training complete")

if __name__=="__main__":
    main()
