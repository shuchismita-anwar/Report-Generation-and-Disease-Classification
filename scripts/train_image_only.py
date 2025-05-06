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

import math
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_image_only')

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
OUTPUT_DIR   = "output_image_only"

BATCH_SIZE   = 12
ACCUM_STEPS  = 4
LR           = 1e-4
WD           = 1e-4
EPOCHS       = 50
GRAD_CLIP    = 1.0

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
        logger.info(f"Initializing {split} dataset with {len(df[df['split'] == split])} samples")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.img_tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(0.5) if split=="train" else nn.Identity(),
            transforms.RandomRotation(3) if split=="train" else nn.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
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
            logger.error(f"Error loading image {row['image_path']}: {e}")
            # Return a blank image if we can't load
            img = torch.zeros(3, 512, 512)
        
        # --- Condition Vector ---
        y = torch.tensor(row["cond_vec"], dtype=torch.float32)
        
        return {"img": img, "y": y}

###############################
# === LOAD & SPLIT DATA ===   #
###############################
df = pd.read_csv(DATA_CSV)

def make_cond_vec(cs: str):
    if pd.isna(cs) or cs == "":
        return [0] * len(CONDITIONS)
    # If already a list, use it
    if isinstance(cs, list):
        conds = cs
    elif isinstance(cs, str):
        # Try to split by pipe or comma
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
    logger.info("Creating train/val/test split")
    patients = df["patient_folder"].unique().tolist()
    np.random.shuffle(patients)
    n = len(patients)
    cut1, cut2 = int(0.7*n), int(0.85*n)
    train_patients = patients[:cut1]
    val_patients = patients[cut1:cut2]
    test_patients = patients[cut2:]
    
    df.loc[df["patient_folder"].isin(train_patients), "split"] = "train"
    df.loc[df["patient_folder"].isin(val_patients), "split"] = "val"
    df.loc[df["patient_folder"].isin(test_patients), "split"] = "test"

train_dataset = CXRDataset(df, "train")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True, prefetch_factor=2)

val_dataset = CXRDataset(df, "val")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=4, pin_memory=True, prefetch_factor=2)

test_dataset = CXRDataset(df, "test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, prefetch_factor=2)

#########################
# === MODEL DEF ===     #
#########################
class ImageOnlyModel(nn.Module):
    def __init__(self, num_conditions:int):
        super().__init__()
        logger.info(f"Initializing model with {num_conditions} conditions")
        # Load pre-trained ViT model as feature extractor
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, img_size=512, num_classes=0)
        # Classifier head
        self.classifier = nn.Linear(self.vit.num_features, num_conditions)

    def forward(self, batch):
        # Extract image features
        img_feat = self.vit(batch["img"])
        logits = self.classifier(img_feat)
        return {"logits": logits}

#############################
# === TRAIN & EVAL ===      #
#############################
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train")
    for i, batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(DEVICE)
        # Use new autocast signature
        with torch.amp.autocast(device_type='cuda'):
            try:
                out = model(batch)
                L_cls = F.binary_cross_entropy_with_logits(out["logits"], batch["y"])
                L = L_cls
                if not math.isfinite(L.item()):
                    logger.warning(f"Non-finite loss detected: {L.item()}")
                    L = torch.tensor(0.0, device=DEVICE)
            except Exception as e:
                logger.error(f"Error in forward pass: {e}")
                L = torch.tensor(0.0, device=DEVICE)
        scaler.scale(L/ACCUM_STEPS).backward()
        if (i+1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += L.item()
        pbar.set_postfix(loss=total_loss/(i+1))

    return total_loss/len(loader)

@torch.no_grad()
def evaluate(model, loader, split="Val"):
    model.eval()
    ys, ps, losses = [], [], []
    pbar = tqdm(loader, desc=f"Eval {split}")
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(DEVICE)
        
        try:
            out = model(batch)
            L = F.binary_cross_entropy_with_logits(out["logits"], batch["y"])
            losses.append(L.item())
            ys.append(batch["y"].cpu().numpy())
            ps.append(torch.sigmoid(out["logits"]).cpu().numpy())
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            continue
    
    if not ys:
        logger.error("No valid predictions for evaluation")
        return 0.0, 0.0, 0.0, 0.0
        
    ys = np.vstack(ys)
    ps = np.vstack(ps)
    aucs = []
    for i in range(ys.shape[1]):
        try:
            aucs.append(roc_auc_score(ys[:,i], ps[:,i]))
        except Exception as e:
            logger.warning(f"Error calculating AUC for condition {i}: {e}")
            aucs.append(np.nan)
    
    mean_auc = np.nanmean(aucs)
    y_pred = (ps >= 0.5).astype(int)
    f1 = f1_score(ys, y_pred, average="macro", zero_division=0)
    rec = recall_score(ys, y_pred, average="macro", zero_division=0)
    prec = precision_score(ys, y_pred, average="macro", zero_division=0)
    
    return mean_auc, f1, rec, prec

def main():
    logger.info("Starting training process")
    model = ImageOnlyModel(len(CONDITIONS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val = 0.0
    model_path = os.path.join(OUTPUT_DIR, "best_image_only.pt")
    
    for ep in range(1, EPOCHS+1):
        logger.info(f"=== Epoch {ep}/{EPOCHS} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        val_auc, f1, rec, prec = evaluate(model, val_loader, "Val")
        logger.info(f"Val AUC={val_auc:.4f} F1={f1:.4f} R={rec:.4f} P={prec:.4f}")
        
        if val_auc > best_val:
            best_val = val_auc
            logger.info(f"New best model with val AUC: {val_auc:.4f}")
            torch.save(model.state_dict(), model_path)
        
        scheduler.step()

    logger.info("=== Test ===")
    model.load_state_dict(torch.load(model_path))
    test_auc, test_f1, test_rec, test_prec = evaluate(model, test_loader, "Test")
    logger.info(f"Test AUC={test_auc:.4f} F1={test_f1:.4f} R={test_rec:.4f} P={test_prec:.4f}")
    
    # Save final metrics
    with open(os.path.join(OUTPUT_DIR, "metrics_image_only.json"), "w") as f:
        json.dump({
            "test_auc": float(test_auc),
            "test_f1": float(test_f1),
            "test_recall": float(test_rec), 
            "test_precision": float(test_prec)
        }, f, indent=4)
    
    logger.info("Training complete")

if __name__=="__main__":
    main()
