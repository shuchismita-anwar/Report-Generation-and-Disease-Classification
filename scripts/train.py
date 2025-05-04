# scripts/train.py

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


# scripts/train.py (near the top)

# 1) Your fixed label list
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

# 2) Build a mapping from condition → index
cond2idx = {c: i for i, c in enumerate(CONDITIONS)}


#####################
# === CONSTANTS === #
#####################
# Paths
DATA_CSV       = "final_dataset_fixed.csv"
FIXSTATS_NPZ   = "fixstats.npz"
IMG_DIR        = os.path.join("output", "img_png")
BBOX_DIR       = os.path.join("output", "bbox_mask")
FIXSEQ_DIR     = os.path.join("output", "fix_seq")
OUTPUT_DIR     = "output"

# Hyperparams
BATCH_SIZE     = 12
ACCUM_STEPS    = 4
LR             = 1e-4
WD             = 1e-4
EPOCHS         = 20
TAU            = 0.07

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42

# Random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# Make output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)


#############################
# === LOAD FIX STATS ===    #
#############################
stats = np.load(FIXSTATS_NPZ)
MU, SIGMA = stats["mu"], stats["sigma"]  # shape (4,)


#########################
# === DATASET DEF ===   #
#########################
class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str):
        """
        df: full dataframe loaded from DATA_CSV
        split: one of "train","val","test"; df must contain a 'split' column
        """
        self.df = df[df["split"] == split].reset_index(drop=True)

        # text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )

        # image transforms
        self.img_tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(0.5) if split=="train" else nn.Identity(),
            transforms.RandomRotation(3) if split=="train" else nn.Identity(),
            transforms.ToTensor(),               # -> [0,1]
            transforms.Normalize(                # simple norm
                mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Image ---
        img_name = Path(row["image_path"]).stem + ".png"
        img = Image.open(os.path.join(IMG_DIR, img_name)).convert("RGB")
        img = self.img_tf(img)  # (3,512,512)

        # --- BBox mask ---
        bbox_name = Path(row["bbox_path"]).stem + ".png"
        bbox = Image.open(os.path.join(BBOX_DIR, bbox_name)).convert("L")
        bbox = transforms.ToTensor()(bbox)  # (1,512,512)

        # --- Fixation sequence ---
        fix_name = Path(row["fixations_path"]).stem + ".npz"
        arr = np.load(os.path.join(FIXSEQ_DIR, fix_name))
        seq = arr["seq"].astype(np.float32)  # Ensure float32
        mask = torch.from_numpy(arr["mask"]).bool()  # Only convert once
        seq = torch.from_numpy(seq).float()   # Ensure torch.float32
        seq = torch.clamp(seq, min=0.0, max=1.0)

        # --- Transcript tokens ---
        with open(row["transcript_path"], "r") as f:
            full = json.load(f)["full_text"].lower()
        tok = self.tokenizer(
            full,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.squeeze(0)     # (128,)
        attn_mask = tok.attention_mask.squeeze(0)

        # --- Condition multi-hot ---
        y = torch.from_numpy(row["cond_vec"].astype("float32"))  # (num_conditions,)

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

# Create the multi-hot vectors
def make_cond_vec(cs: str):
    vec = np.zeros(len(CONDITIONS), dtype=np.int8)
    for c in cs.split("|"):
        c = c.strip()
        if c in cond2idx:
            vec[cond2idx[c]] = 1
    return vec

df["cond_vec"] = df["condition"].apply(make_cond_vec)

# If no 'split' column, do patient-level split now
if "split" not in df.columns:
    patients = df["patient_folder"].unique().tolist()
    np.random.shuffle(patients)
    n = len(patients)
    cut1, cut2 = int(0.7*n), int(0.85*n)
    train_p = set(patients[:cut1])
    val_p   = set(patients[cut1:cut2])
    test_p  = set(patients[cut2:])
    def assign_split(r):
        if r in train_p: return "train"
        if r in val_p:   return "val"
        return "test"
    df["split"] = df["patient_folder"].map(assign_split)

# instantiate datasets & loaders
train_ds = CXRDataset(df, "train")
val_ds   = CXRDataset(df, "val")
test_ds  = CXRDataset(df, "test")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=8, pin_memory=True, prefetch_factor=4,
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, prefetch_factor=2,
)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, prefetch_factor=2,
)


#########################
# === MODEL DEF ===     #
#########################
class MultiModalModel(nn.Module):
    def __init__(self, num_conditions: int):
        super().__init__()

        # ViT image encoder (BioViL weights)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, img_size=512)
        # load BioViL state dict here if available...
        # self.vit.load_state_dict(torch.load("biovil_vit_base.pth"), strict=False)

        # fixation seq → embedding
        self.fix_mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.GELU(),
            nn.Linear(128, 256),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=512, dropout=0.1,
            batch_first=True
        )
        self.fix_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2
        )
        self.cls_gaze = nn.Parameter(torch.randn(1,1,256))

        # cross-attention: 1-head, project gaze→768
        self.gaze_to_768 = nn.Linear(256, 768)
        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=1, batch_first=True)

        # text encoder
        self.txt_enc = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # projection heads
        self.proj_img = nn.Sequential(nn.Linear(768,512), nn.ReLU(), nn.Linear(512,512))
        self.proj_gaze = nn.Sequential(nn.Linear(256,512), nn.ReLU(), nn.Linear(512,512))
        self.proj_txt = nn.Sequential(nn.Linear(768,512), nn.ReLU(), nn.Linear(512,512))

        # classifier
        self.classifier = nn.Linear(768, num_conditions)

        # context decoder (GRU) + output
        self.ctx_decoder = nn.GRU(768, 768, batch_first=True)
        self.out_ctx = nn.Linear(768, self.txt_enc.config.vocab_size)

    def forward(self, batch):
        # image → ViT
        x = self.vit.patch_embed(batch["image"])
        cls_tok = self.vit.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tok, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        img_cls = x[:, 0]                   # (B,768)

        # placeholder attention map (to implement later)
        attn_map = torch.zeros(
            batch["image"].size(0), 512, 512, device=img_cls.device
        )

        # fixation sequence
        seq = batch["fix_seq"]             # (B,L,4)
        B = seq.size(0)
        gs = self.fix_mlp(seq)              # (B,L,256)
        cls_g = self.cls_gaze.expand(B, -1, -1)  # (B,1,256)
        gs = torch.cat((cls_g, gs), dim=1)       # (B,L+1,256)
        # fix_mask: (B,L)
        mask = batch["fix_mask"]  # (B,L)
        # prepend False for CLS token
        cls_pad = torch.zeros((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)
        mask = torch.cat([cls_pad, mask], dim=1)  # (B,L+1)
        gaze_out = self.fix_transformer(gs, src_key_padding_mask=mask)
        gaze_cls = gaze_out[:, 0]                 # (B,256)

        # cross-attn fuse
        k = v = self.gaze_to_768(gaze_cls).unsqueeze(1)  # (B,1,768)
        q = img_cls.unsqueeze(1)                        # (B,1,768)
        fused, _ = self.cross_attn(q, k, v)              # (B,1,768)
        fused = fused.squeeze(1)                         # (B,768)

        # text encoding
        txt = self.txt_enc(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attn_mask"].to(DEVICE),
        )
        txt_cls = txt.pooler_output                      # (B,768)

        # projections
        zi = F.normalize(self.proj_img(img_cls), dim=-1)
        zg = F.normalize(self.proj_gaze(gaze_cls), dim=-1)
        zt = F.normalize(self.proj_txt(txt_cls), dim=-1)

        # classification logits
        logits = self.classifier(fused)

        # context decoding (teacher forcing)
        emb_in = self.txt_enc.embeddings.word_embeddings(
            batch["input_ids"].to(DEVICE)
        )                                                # (B,128,768)
        h0 = fused.unsqueeze(0)                          # (1,B,768)
        dec_out, _ = self.ctx_decoder(emb_in, h0)        # (B,128,768)
        ctx_logits = self.out_ctx(dec_out)               # (B,128,Vocab)

        return {
            "img_cls": img_cls,
            "gaze_cls": gaze_cls,
            "txt_cls": txt_cls,
            "zi": zi,
            "zg": zg,
            "zt": zt,
            "logits": logits,
            "ctx_logits": ctx_logits,
            "attn_map": attn_map,
        }


###################################
# === LOSS FUNCTIONS & OPT ===    #
###################################
def info_nce_loss(zA, zB, tau=0.07):
    """
    zA, zB: (B, D), normalized
    """
    B = zA.size(0)
    logits = (zA @ zB.t()) / tau            # (B,B)
    labels = torch.arange(B, device=logits.device)
    lossA = F.cross_entropy(logits, labels)
    lossB = F.cross_entropy(logits.t(), labels)
    return (lossA + lossB) * 0.5

def dice_loss(attn, mask):
    """
    attn: (B,512,512) [0,1]
    mask: (B,1,512,512) {0,1}
    """
    B = attn.size(0)
    attn = attn.view(B, -1)
    mask = mask.view(B, -1)
    inter = (attn * mask).sum(1)
    union = attn.sum(1) + mask.sum(1)
    return 1 - ((2*inter + 1e-6) / (union + 1e-6)).mean()

def kl_map(attn, mask):
    """
    attn: (B,512,512) raw scores
    mask: (B,1,512,512)
    """
    B = attn.size(0)
    attn_flat = attn.view(B, -1)
    attn_p = F.log_softmax(attn_flat, dim=1)
    mask_flat = mask.view(B, -1).float()
    mask_p = mask_flat / (mask_flat.sum(1, keepdim=True) + 1e-6)
    return F.kl_div(attn_p, mask_p, reduction="batchmean")


#############################
# === TRAIN & EVAL ===      #
#############################
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train")
    for i, batch in pbar:
        # move to device
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(DEVICE)

        with torch.cuda.amp.autocast():
            out = model(batch)
            # losses
            L_cls  = F.binary_cross_entropy_with_logits(out["logits"], batch["y"])
            L_ctx  = F.cross_entropy(
                out["ctx_logits"].permute(0,2,1),
                batch["input_ids"],
                ignore_index=0
            )
            L_IT   = info_nce_loss(out["zi"], out["zt"], tau=TAU)
            L_IF   = info_nce_loss(out["zi"], out["zg"], tau=TAU)
            L_kld  = kl_map(out["attn_map"], batch["bbox"])
            L = 1.0*L_cls + 1.0*L_ctx + 0.5*L_IT + 0.5*L_IF + 0.1*L_kld

        scaler.scale(L/ACCUM_STEPS).backward()

        if (i+1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += L.item()
        pbar.set_postfix(loss=total_loss/(i+1))
    return total_loss/len(loader)


@torch.no_grad()
def evaluate(model, loader, split="Val"):
    model.eval()
    ys, ps = [], []
    losses = []
    pbar = tqdm(loader, desc=f"Eval {split}")
    for batch in pbar:
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(DEVICE)
        out = model(batch)
        L = F.binary_cross_entropy_with_logits(out["logits"], batch["y"])
        losses.append(L.item())

        y_true = batch["y"].cpu().numpy()
        y_prob = torch.sigmoid(out["logits"]).cpu().numpy()
        ys.append(y_true); ps.append(y_prob)

    ys = np.vstack(ys); ps = np.vstack(ps)
    # metrics
    aucs = []
    for i in range(ys.shape[1]):
        try:
            aucs.append(roc_auc_score(ys[:,i], ps[:,i]))
        except ValueError:
            aucs.append(np.nan)
    mean_auc = np.nanmean(aucs)
    y_pred = (ps >= 0.5).astype(int)
    f1 = f1_score(ys, y_pred, average="macro", zero_division=0)
    recall = recall_score(ys, y_pred, average="macro", zero_division=0)
    precision = precision_score(ys, y_pred, average="macro", zero_division=0)

    print(f"{split} Loss: {np.mean(losses):.4f} | AUC: {mean_auc:.4f} | "
          f"F1: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}")
    return mean_auc, f1, recall, precision


def main():
    # model, optimizer, scaler
    num_conditions = df["cond_vec"].iloc[0].shape[0]
    model = MultiModalModel(num_conditions).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WD, fused=True
    )
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val_auc = 0.0
    for ep in range(1, EPOCHS+1):
        print(f"\n=== Epoch {ep}/{EPOCHS} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        val_auc, _, _, _ = evaluate(model, val_loader, split="Val")

        # save best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best.pt"))

        scheduler.step()

    # final test
    print("\n=== Testing on best model ===")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best.pt")))
    evaluate(model, test_loader, split="Test")


if __name__ == "__main__":
    main()
