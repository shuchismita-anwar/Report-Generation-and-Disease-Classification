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
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)

import math

def log_tensor_stats(name, tensor):
    pass

############################
# 1) Fixed label list
############################
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
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
OUTPUT_DIR   = "output"

BATCH_SIZE   = 12
ACCUM_STEPS  = 4
LR           = 1e-4
WD           = 1e-4
EPOCHS       = 50
TAU          = 0.07
GRAD_CLIP    = 1.0

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

#############################
# === LOAD & FIX STATS ===  #
#############################
stats = np.load(FIXSTATS_NPZ)
MU, SIGMA = stats["mu"], stats["sigma"]
# guard
SIGMA = SIGMA + 1e-6

#########################
# === DATASET DEF ===   #
#########################
class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
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
        img = Image.open(os.path.join(IMG_DIR, Path(row["image_path"]).stem + ".png")).convert("RGB")
        img = self.img_tf(img)
        # log_tensor_stats("img", img)

        # --- BBox mask ---
        bbox = Image.open(os.path.join(BBOX_DIR, Path(row["bbox_path"]).stem + ".png")).convert("L")
        bbox = transforms.ToTensor()(bbox)
        # log_tensor_stats("bbox", bbox)

        # --- Fixation sequence ---
        arr = np.load(os.path.join(FIXSEQ_DIR, Path(row["fixations_path"]).stem + ".npz"))
        seq = arr["seq"].astype(np.float32)  # (L,4)
        mask = arr["mask"].astype(bool)      # (L,)
        
        # Safely handle any remaining NaNs - but don't re-normalize
        # The data was already normalized during preprocessing
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to tensors and apply safety clipping
        seq = torch.from_numpy(seq).float()
        seq = seq.clamp(-10.0, 10.0)
        mask = torch.from_numpy(mask)
        # log_tensor_stats("fix_seq", seq)
        # log_tensor_stats("fix_mask", mask)

        # --- Transcript tokens ---
        with open(row["transcript_path"], "r") as f:
            full = json.load(f)["full_text"].lower()
        tok = self.tokenizer(full,
                             max_length=128,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")
        input_ids = tok.input_ids.squeeze(0)
        attn_mask = tok.attention_mask.squeeze(0)
        # log_tensor_stats("input_ids", input_ids)
        # log_tensor_stats("attn_mask", attn_mask)

        # --- Condition multi-hot ---
        y = torch.from_numpy(row["cond_vec"].astype("float32"))
        # log_tensor_stats("y", y)

        return {
            "image": img,
            "bbox": bbox,
            "fix_seq": seq,
            "fix_mask": mask,
            "input_ids": input_ids,
            "attn_mask": attn_mask,
            "y": y,
        }

###############################
# === LOAD & SPLIT DATA ===   #
###############################
df = pd.read_csv(DATA_CSV)
def make_cond_vec(cs: str):
    vec = np.zeros(len(CONDITIONS), dtype=np.int8)
    for c in cs.split("|"):
        c = c.strip()
        if c in cond2idx:
            vec[cond2idx[c]] = 1
    return vec
df["cond_vec"] = df["condition"].apply(make_cond_vec)

if "split" not in df.columns:
    patients = df["patient_folder"].unique().tolist()
    np.random.shuffle(patients)
    n = len(patients)
    cut1, cut2 = int(0.7*n), int(0.85*n)
    train_p, val_p, test_p = set(patients[:cut1]), set(patients[cut1:cut2]), set(patients[cut2:])
    df["split"] = df["patient_folder"].map(
        lambda p: "train" if p in train_p else "val" if p in val_p else "test"
    )

train_loader = DataLoader(CXRDataset(df, "train"),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, prefetch_factor=4)
val_loader   = DataLoader(CXRDataset(df, "val"),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, prefetch_factor=2)
test_loader  = DataLoader(CXRDataset(df, "test"),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, prefetch_factor=2)

#########################
# === MODEL DEF ===     #
#########################
class MultiModalModel(nn.Module):
    def __init__(self, num_conditions:int):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, img_size=512)
        # gaze
        self.fix_mlp   = nn.Sequential(nn.Linear(4,128), nn.GELU(), nn.Linear(128,256))
        enc_layer      = nn.TransformerEncoderLayer(d_model=256,nhead=4,dim_feedforward=512,batch_first=True)
        self.fix_trans = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.cls_gaze  = nn.Parameter(torch.randn(1,1,256))
        # cross-attn
        self.gaze_to_768 = nn.Linear(256,768)
        self.cross_attn  = nn.MultiheadAttention(embed_dim=768, num_heads=1, batch_first=True)
        # text
        self.txt_enc    = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # projections
        self.proj_img   = nn.Sequential(nn.Linear(768,512),nn.ReLU(),nn.Linear(512,512))
        self.proj_gaze  = nn.Sequential(nn.Linear(256,512),nn.ReLU(),nn.Linear(512,512))
        self.proj_txt   = nn.Sequential(nn.Linear(768,512),nn.ReLU(),nn.Linear(512,512))
        # heads
        self.classifier = nn.Linear(768, num_conditions)
        self.ctx_decoder= nn.GRU(768,768, batch_first=True)
        self.out_ctx    = nn.Linear(768, self.txt_enc.config.vocab_size)

    def forward(self, batch):
        # image
        x = self.vit.patch_embed(batch["image"])
        cls_tok = self.vit.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tok,x], dim=1) + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks: x = blk(x)
        x = self.vit.norm(x)
        img_cls = x[:,0]
        
        # Add safety check for NaNs
        if torch.isnan(img_cls).any():
            img_cls = torch.nan_to_num(img_cls, nan=0.0)

        attn_map = torch.zeros(batch["image"].size(0),512,512, device=img_cls.device)

        # gaze
        seq = batch["fix_seq"]
        B = seq.size(0)
        
        # Add safety check for NaNs in fixation sequences
        if torch.isnan(seq).any():
            seq = torch.nan_to_num(seq, nan=0.0)
            
        gs = self.fix_mlp(seq)
        
        # Check for NaNs after MLP
        if torch.isnan(gs).any():
            gs = torch.nan_to_num(gs, nan=0.0)
            
        cls_g = self.cls_gaze.expand(B,-1,-1)
        gs = torch.cat([cls_g,gs], dim=1)
        
        # Create padding mask with safety check
        mask = ~batch["fix_mask"]
        pad_mask = torch.cat([
            torch.zeros(B,1,dtype=torch.bool,device=mask.device),
            mask
        ], dim=1)
        gs = self.fix_trans(gs, src_key_padding_mask=pad_mask)
        gaze_cls = gs[:,0]
        
        # Check for NaNs after transformer
        if torch.isnan(gaze_cls).any():
            gaze_cls = torch.nan_to_num(gaze_cls, nan=0.0)

        # text
        ids, mask = batch["input_ids"], batch["attn_mask"]
        txt_out = self.txt_enc(ids, attention_mask=mask)
        txt_cls = txt_out.last_hidden_state[:,0]
        
        # Check for NaNs in text encoder output
        if torch.isnan(txt_cls).any():
            txt_cls = torch.nan_to_num(txt_cls, nan=0.0)

        # cross-attention with safety checks
        gaze_proj = self.gaze_to_768(gaze_cls).unsqueeze(1)
        try:
            fused, attn_w = self.cross_attn(gaze_proj, x, x)
            fused = fused.squeeze(1)
        except Exception as e:
            # Fallback: use simple addition instead of attention
            fused = gaze_proj.squeeze(1) + img_cls
            
        # Check for NaNs after cross-attention
        if torch.isnan(fused).any():
            fused = torch.nan_to_num(fused, nan=0.0)
        
        # contrastive
        zi = F.normalize(self.proj_img(img_cls), dim=-1)
        zg = F.normalize(self.proj_gaze(gaze_cls), dim=-1)
        zt = F.normalize(self.proj_txt(txt_cls), dim=-1)
        
        # Check for NaNs in projections
        if torch.isnan(zi).any() or torch.isnan(zg).any() or torch.isnan(zt).any():
            zi = torch.nan_to_num(zi, nan=0.0)
            zg = torch.nan_to_num(zg, nan=0.0)
            zt = torch.nan_to_num(zt, nan=0.0)

        # classifier
        logits = self.classifier(fused)
        
        # Check for NaNs in logits
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0)

        # decode with safety checks
        try:
            emb_in = self.txt_enc.embeddings.word_embeddings(batch["input_ids"])
            dec, _ = self.ctx_decoder(emb_in, fused.unsqueeze(0))
            ctx_logits = self.out_ctx(dec)
            
            # Check for NaNs in decoder
            if torch.isnan(ctx_logits).any():
                ctx_logits = torch.nan_to_num(ctx_logits, nan=0.0)
        except Exception as e:
            # Create fallback context logits filled with padding token
            ctx_logits = torch.zeros(
                batch["input_ids"].shape[0],
                batch["input_ids"].shape[1],
                self.txt_enc.config.vocab_size,
                device=fused.device
            )

        return {
            "img_cls": img_cls, "gaze_cls": gaze_cls, "txt_cls": txt_cls,
            "zi": zi, "zg": zg, "zt": zt,
            "logits": logits, "ctx_logits": ctx_logits,
            "attn_map": attn_map
        }

#########################
# === LOSSES & OPT ===  #
#########################
def info_nce_loss(zA,zB,tau=0.07):
    B = zA.size(0)
    logits = (zA @ zB.t())/tau
    labels = torch.arange(B,device=logits.device)
    return (F.cross_entropy(logits,labels)
          + F.cross_entropy(logits.t(),labels)) * 0.5

def kl_map(attn:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
    B = attn.size(0)
    af = attn.view(B,-1)
    mf = mask.view(B,-1).float()
    losses=[]
    for i in range(B):
        m,s = mf[i], mf[i].sum()
        if s<1e-6: continue
        a = af[i]
        log_pa = F.log_softmax(a, dim=0)
        p_m = m/(s+1e-6)
        losses.append(F.kl_div(log_pa,p_m,reduction="batchmean"))
    if not losses:
        return torch.tensor(0.,device=attn.device)
    return torch.stack(losses).mean()

#############################
# === TRAIN & EVAL ===      #
#############################
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss=0.0
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train")
    for i, batch in pbar:
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(DEVICE)
        with torch.cuda.amp.autocast():
            try:
                out = model(batch)
                L_cls = F.binary_cross_entropy_with_logits(out["logits"], batch["y"])
                L_ctx = F.cross_entropy(out["ctx_logits"].reshape(-1, out["ctx_logits"].size(-1)), 
                                       batch["input_ids"].reshape(-1), 
                                       ignore_index=0)
                L_IT = info_nce_loss(out["zi"], out["zt"], tau=TAU)
                L_IF = info_nce_loss(out["zi"], out["zg"], tau=TAU)
                attn = out["attn_map"]
                L_kd = kl_map(attn, batch["bbox"])
                L = L_cls + L_ctx + 0.5*(L_IT+L_IF) + 0.1*L_kd
                if not math.isfinite(L.item()):
                    L = torch.tensor(0.0, device=DEVICE)
            except Exception as e:
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
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k]=v.to(DEVICE)
        out = model(batch)
        L = F.binary_cross_entropy_with_logits(out["logits"], batch["y"])
        losses.append(L.item())
        ys.append(batch["y"].cpu().numpy())
        ps.append(torch.sigmoid(out["logits"]).cpu().numpy())
    ys = np.vstack(ys); ps = np.vstack(ps)
    aucs=[]
    for i in range(ys.shape[1]):
        try:
            aucs.append(roc_auc_score(ys[:,i],ps[:,i]))
        except Exception as e:
            aucs.append(np.nan)
    mean_auc = np.nanmean(aucs)
    y_pred = (ps>=0.5).astype(int)
    f1   = f1_score(ys, y_pred, average="macro", zero_division=0)
    rec  = recall_score(ys, y_pred, average="macro", zero_division=0)
    prec = precision_score(ys, y_pred, average="macro", zero_division=0)
    return mean_auc, f1, rec, prec

def main():
    model = MultiModalModel(len(CONDITIONS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, fused=True)
    scaler    = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val = 0.0
    for ep in range(1, EPOCHS+1):
        print(f"\n=== Epoch {ep}/{EPOCHS} ===")
        train_one_epoch(model, train_loader, optimizer, scaler)
        val_auc, f1, rec, prec = evaluate(model, val_loader, "Val")
        print(f"Val Loss/AUC/F1/Recall/Precision: AUC={val_auc:.4f} F1={f1:.4f} R={rec:.4f} P={prec:.4f}")
        if val_auc > best_val:
            best_val = val_auc
            torch.save(model.state_dict(), "best.pt")
        scheduler.step()

    print("\n=== Test ===")
    model.load_state_dict(torch.load("best.pt"))
    test_auc, test_f1, test_rec, test_prec = evaluate(model, test_loader, "Test")
    print(f"Test Loss/AUC/F1/Recall/Precision: AUC={test_auc:.4f} F1={test_f1:.4f} R={test_rec:.4f} P={test_prec:.4f}")

if __name__=="__main__":
    main()
