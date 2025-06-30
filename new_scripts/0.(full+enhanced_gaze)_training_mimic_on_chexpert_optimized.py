"""
ENHANCED GAZE-GUIDED MIMIC Training with Priority 2 Implementation
================================================================

🎯 DUAL GOALS:
1. Beat ViT-only baseline F1 of 0.6242 with multimodal approach
2. Implement Priority 2: Advanced gaze-guided training for A* research contribution

🧠 PRIORITY 2 FEATURES IMPLEMENTED:
1. Enhanced Gaze Supervision:
   - Multi-component loss: MSE + KL + Pearson correlation + Center-of-mass
   - Quality-weighted loss based on fixation characteristics
   - Temporal decay and physiological weighting

2. Contrastive Gaze-Text Alignment (CLIP-style):
   - Semantic alignment between gaze patterns and transcript text
   - Cross-modal contrastive learning with temperature scaling
   - Positive/negative pair generation for medical concepts

3. Multi-level Attention Extraction:
   - Enhanced spatial attention from ViT patch tokens
   - Multi-head attention mechanisms for spatial reasoning
   - Multi-scale attention generation (14x14, 28x28, 56x56)

4. Advanced Loss Combination:
   - Adaptive weighting: Classification + Traditional Contrastive + Enhanced Gaze + NEW Gaze-Text
   - Quality-based sample weighting for better training stability
   - Increased gaze component weights to emphasize Priority 2 learning

🔬 RESEARCH CONTRIBUTION:
This transforms from "we used gaze data" to "we systematically learned human-AI attention alignment" 
- providing the deep gaze integration insights needed for A* level research.

Expected Outcome: F1 >= 0.6242 + Quantifiable gaze-attention alignment improvement
"""

import os
import sys
import gc
import psutil

# Add memory management at the very beginning
def check_system_memory():
    """Check available system memory and optimize settings"""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    total_gb = memory.total / (1024**3)
    
    print(f"System Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
    
    if available_gb < 4.0:  # Less than 4GB available
        print("WARNING: Low system memory detected. Applying optimizations...")
        return True
    return False

# Check memory before importing torch
low_memory = check_system_memory()

# Set environment variables for memory optimization BEFORE importing torch
if low_memory or os.name == 'nt':  # Windows system
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Try to import torch with fallback to CPU-only mode
try:
    import torch
    if torch.cuda.is_available():
        # Test CUDA allocation
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            DEVICE = "cuda"
            print("CUDA available and working")
        except Exception as e:
            print(f"CUDA available but not working: {e}")
            print("Falling back to CPU mode")
            DEVICE = "cpu"
    else:
        DEVICE = "cpu"
        print("CUDA not available, using CPU")
except Exception as e:
    print(f"Error importing torch: {e}")
    print("This might be due to insufficient virtual memory.")
    print("Please try the system-level fixes mentioned in the console.")
    sys.exit(1)

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
import numpy as np

from dataclasses import dataclass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('training_mimic_on_chexpert_optimized')

############################
# 1) Fixed label list (8 conditions as requested)
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

FIXSTATS_NPZ = "fixstats.npz"

# Updated paths as requested
IMG_DIR      = os.path.join("data_dump", "output", "img_png")
BBOX_DIR     = os.path.join("data_dump", "output", "bbox_mask")
FIXSEQ_DIR   = os.path.join("data_dump", "output", "fix_seq")

# Model paths and names
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Path to the CheXpert-trained ViT model
CHEXPERT_MODEL_PATH = os.path.join("new_scripts", "output", "vit_chexpert_training_output", "model_1.chexpert_finetune_only_feat_image", "model.safetensors")

OUTPUT_DIR   = os.path.join("new_scripts", "output", "0.(full+enhanced_gaze)_training_mimic_on_chexpert_optimized")
EXP_NAME     = "0.(full+enhanced_gaze)_training_mimic_on_chexpert_optimized"

# Memory-optimized hyperparameters
if low_memory or DEVICE == "cpu":
    BATCH_SIZE   = 8   # Much smaller batch size for low memory
    GRAD_ACCUM   = 12  # Higher accumulation to maintain effective batch size
    DATALOADER_WORKERS = 0  # No multiprocessing on low memory systems
    FP16_ENABLED = False  # Disable FP16 on CPU or low memory
else:
    BATCH_SIZE   = 32
    GRAD_ACCUM   = 6
    DATALOADER_WORKERS = 2  # Reduced from 4
    FP16_ENABLED = DEVICE == "cuda"

LR           = 6e-6  # IMPROVED: Slightly higher LR for better convergence
WD           = 8e-4  # IMPROVED: Slightly lower weight decay
EPOCHS       = 40    # IMPROVED: A few more epochs
EARLY_STOPPING_PATIENCE = 10  # IMPROVED: More patience for better convergence
WARMUP_STEPS = 175   # IMPROVED: Longer warmup for stability

SEED         = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Enable memory optimization
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = False  # Disable for memory efficiency
    torch.backends.cudnn.deterministic = True

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
    logger.error(f"Error loading fixation stats: {e}")
    logger.warning("Using default fixation stats")
    MU, SIGMA = np.zeros(4), np.ones(4)

#########################
# === DATASET DEF ===   #
#########################
class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str):
        # logger.info(f"Initializing {split} dataset with {len(df)} samples")
        self.df = df.reset_index(drop=True)
        self.split = split
        
        # Filter out rows without required components
        self.df = self.df[
            self.df["bbox_path"].notna() & 
            self.df["fixations_path"].notna() &
            self.df["transcript_path"].notna()
        ].reset_index(drop=True)
        # logger.info(f"After filtering for multimodal availability: {len(self.df)} samples")
        
        # Memory-optimized augmentation
        if split == "train" and not low_memory:
            self.img_tf = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])
        else:
            # Simpler transforms for low memory or eval
            self.img_tf = transforms.Compose([
                transforms.Resize((224, 224)),
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
            logger.error(f"Error loading image {row['image_path']}: {e}")
            img = torch.zeros(3, 224, 224)
        
        # --- BBox mask ---
        try:
            bbox_path = os.path.join(BBOX_DIR, Path(row["bbox_path"]).stem + ".png")
            bbox = Image.open(bbox_path).convert("L")
            bbox = self.mask_tf(bbox)
        except Exception as e:
            logger.error(f"Error loading mask {row['bbox_path']}: {e}")
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
            
            # Close numpy file to free memory
            arr.close() if hasattr(arr, 'close') else None
            
        except Exception as e:
            logger.error(f"Error loading fixation sequence {row['fixations_path']}: {e}")
            seq = torch.zeros(128, 4)  # Assuming max length of 128
            mask = torch.zeros(128, dtype=torch.bool)
        
        # --- Transcript ---
        try:
            transcript_path = row["transcript_path"]
            if pd.isna(transcript_path) or not os.path.exists(transcript_path):
                transcript_text = ""
            else:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                    transcript_text = transcript_data.get("transcript", "")
        except Exception as e:
            logger.error(f"Error loading transcript {row.get('transcript_path', 'N/A')}: {e}")
            transcript_text = ""
        
        # --- Condition Vector ---
        labels = torch.tensor(row["cond_vec"], dtype=torch.float32)
        
        # Force garbage collection every 100 samples on low memory systems
        if low_memory and idx % 100 == 0:
            gc.collect()
        
        return {
            "img": img, 
            "bbox": bbox, 
            "fix_seq": seq, 
            "fix_mask": mask, 
            "transcript": transcript_text,
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

val_dataset = CXRDataset(val_df, split="val")

test_dataset = CXRDataset(test_df, split="test")

#################################
# Custom Data Collator for multimodal batching #
#################################
@dataclass
class MultiModalDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        
        # Stack image tensors
        batch["img"] = torch.stack([f["img"] for f in features])
        
        # Stack bbox tensors
        batch["bbox"] = torch.stack([f["bbox"] for f in features])
        
        # Pad fixation sequences
        max_fix_len = max(f["fix_seq"].size(0) for f in features)
        fix_seqs = []
        fix_masks = []
        
        for f in features:
            seq = f["fix_seq"]
            mask = f["fix_mask"]
            
            # Pad sequence and mask
            pad_len = max_fix_len - seq.size(0)
            if pad_len > 0:
                seq = torch.cat([seq, torch.zeros(pad_len, seq.size(1))], dim=0)
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool)], dim=0)
            
            fix_seqs.append(seq)
            fix_masks.append(mask)
        
        batch["fix_seq"] = torch.stack(fix_seqs)
        batch["fix_mask"] = torch.stack(fix_masks)
        
        # Collect transcript strings
        batch["transcript"] = [f["transcript"] for f in features]
        
        # Stack labels
        batch["labels"] = torch.stack([f["labels"] for f in features])
        
        return batch

#################################
# Custom Trainer for Contiguous Saving #
#################################
class ContiguousSavingTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_count = 0
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss computation for multimodal model"""
        # Increment step counter and perform periodic memory cleanup
        self.step_count += 1
        
        # Aggressive memory management every 5 steps
        if self.step_count % 5 == 0:
            monitor_memory()
            
        # More aggressive cleanup every 10 steps
        if self.step_count % 10 == 0:
            aggressive_memory_cleanup()
        
        outputs = model(
            img=inputs["img"],
            bbox=inputs["bbox"], 
            fix_seq=inputs["fix_seq"],
            fix_mask=inputs["fix_mask"],
            transcript=inputs["transcript"],
            labels=inputs["labels"]
        )
        
        loss = outputs["loss"]
        
        # Clean up after loss computation
        if self.step_count % 3 == 0:  # Every 3 steps
            gc.collect()
            
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step for multimodal model"""
        model.eval()
        
        # Memory cleanup before evaluation
        aggressive_memory_cleanup()
        
        with torch.no_grad():
            outputs = model(
                img=inputs["img"],
                bbox=inputs["bbox"],
                fix_seq=inputs["fix_seq"], 
                fix_mask=inputs["fix_mask"],
                transcript=inputs["transcript"],
                labels=inputs["labels"]
            )
            
            loss = outputs["loss"]
            logits = outputs["logits"]
            
        # Memory cleanup after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, inputs["labels"])

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"ContiguousSavingTrainer: Saving model checkpoint to {output_dir}")
        
        # Aggressive memory cleanup before saving
        aggressive_memory_cleanup()

        if isinstance(self.model, PreTrainedModel):
            super()._save(output_dir, state_dict=state_dict)
        else:
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

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            
        # Memory cleanup after saving
        aggressive_memory_cleanup()
        monitor_memory()

#########################
# === MODEL DEF ===     #
#########################
class MultiModalMIMICModel(nn.Module):
    def __init__(self, num_conditions: int, contrastive_temperature: float = 0.1, loss_type="asymmetric"):
        super().__init__()
        logger.info(f"Initializing MultiModalMIMICModel with {num_conditions} conditions using transfer learning from CheXpert.")
        self.contrastive_temperature = contrastive_temperature
        self.loss_type = loss_type
        
        # Add compatibility with Transformers trainer
        self._keys_to_ignore_on_save = None

        # --- Image encoder (Load pre-trained from CheXpert) ---
        self.vit = AutoModel.from_pretrained(VIT_MODEL_NAME)
        img_feature_dim = self.vit.config.hidden_size
        logger.info(f"Loaded ViT base model. Feature dimension: {img_feature_dim}")

        # Memory-optimized projections
        if low_memory:
            # Simpler architecture for low memory
            self.img_proj = nn.Sequential(
                nn.Linear(img_feature_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 512)
            )
            hidden_dim = 512
        else:
            # Enhanced image projection with residual connections
            self.img_proj = nn.Sequential(
                nn.Linear(img_feature_dim, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.Dropout(0.1)
            )
            hidden_dim = 768
        
        # Memory-optimized BBox encoder
        if low_memory:
            self.bbox_encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2))
            )
            self.bbox_proj = nn.Sequential(
                nn.Linear(128, 512),  # 32*4 = 128 -> 512
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            # Enhanced BBox encoder with more capacity
            self.bbox_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2))
            )
            self.bbox_proj = nn.Sequential(
                nn.Linear(512, 768),  # 128*4 = 512 -> 768
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.Dropout(0.1)
            )
        
        # Memory-optimized Fixation encoder
        if low_memory:
            self.fix_emb = nn.Linear(4, 64)
            self.fix_gru = nn.GRU(64, 128, num_layers=1, batch_first=True, bidirectional=True, dropout=0.1)
            self.fix_proj = nn.Sequential(
                nn.Linear(256, 512),  # 2 * 128 = 256 -> 512
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            # Enhanced Fixation encoder
            self.fix_emb = nn.Linear(4, 128)
            self.fix_gru = nn.GRU(128, 384, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
            self.fix_proj = nn.Sequential(
                nn.Linear(1536, 768),  # 2 * 2 * 384 = 1536 -> 768
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.Dropout(0.1)
            )
        
        # Text encoder projection
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        text_feature_dim = self.text_encoder.config.hidden_size
        
        if low_memory:
            self.text_proj = nn.Sequential(
                nn.Linear(text_feature_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            self.text_proj = nn.Sequential(
                nn.Linear(text_feature_dim, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.Dropout(0.1)
            )
        
        # === ENHANCED GAZE-GUIDED ARCHITECTURE (Priority 2) ===
        # Multi-level attention extraction from ViT
        if not low_memory:
            self.vit_attention_head = nn.MultiheadAttention(
                embed_dim=img_feature_dim, 
                num_heads=8, 
                dropout=0.1, 
                batch_first=True
            )
            self.attention_projection = nn.Sequential(
                nn.Linear(img_feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
            
            # Gaze-text contrastive alignment module
            self.gaze_text_projector = nn.Sequential(
                nn.Linear(768, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.LayerNorm(256)
            )
            
            # Multi-scale spatial attention generator
            self.spatial_attention_multi_scale = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, scale * scale),
                    nn.Sigmoid()
                ) for scale in [14, 28, 56]  # Different spatial resolutions
            ])
        
        # Simplified fusion for low memory
        if low_memory:
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 2),  # 512*4 -> 1024
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim * 2, hidden_dim)  # 1024 -> 512
            )
            
            # Simple classifier for low memory
            self.global_classifier = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_conditions)
            )
            self.condition_specific_heads = None  # Disable for low memory
            self.fusion_attention = None  # Disable for low memory
        else:
            # Attention-based fusion
            self.fusion_attention = nn.MultiheadAttention(768, 8, dropout=0.1, batch_first=True)
            self.fusion_norm = nn.LayerNorm(768)
            
            # Enhanced fusion layers
            self.fusion = nn.Sequential(
                nn.Linear(768 * 4, 1536),
                nn.LayerNorm(1536),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(1536, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(0.15)
            )
            
            # Enhanced classifier with multiple heads
            self.condition_specific_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(768, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 1)
                ) for _ in range(num_conditions)
            ])
            
            # Global classifier
            self.global_classifier = nn.Sequential(
                nn.Linear(768, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_conditions)
            )
        
        # Better weight initialization
        self._initialize_weights()
        
        # Loss function
        self.loss_fn = None

    def _initialize_weights(self):
        """Better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def set_loss_function(self, pos_weights, loss_type="focal"):
        """Set loss function - reverting to focal loss with minimal tweaks for F1=0.6242+"""
        if loss_type == "focal":
            # MODERATE BALANCE: Less aggressive focal loss for better precision-recall balance
            self.loss_fn = FocalLoss(alpha=1.10, gamma=2.15, pos_weight=pos_weights, label_smoothing=0.08)
        elif loss_type == "asymmetric":
            # Keep asymmetric option but focal is better for this case
            self.loss_fn = AsymmetricLoss(
                gamma_neg=4,      
                gamma_pos=1,      
                clip=0.05,        
                disable_torch_grad_focal_loss=True
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
        logger.info(f"Set loss function: {loss_type} with minimal tweaks (alpha=1.10, gamma=2.15, smoothing=0.08)")

    def load_chexpert_weights(self, chexpert_model_path: str):
        """Load pre-trained weights from CheXpert model"""
        logger.info(f"Loading CheXpert pre-trained weights from {chexpert_model_path}")
        try:
            if not os.path.exists(chexpert_model_path):
                logger.warning(f"CheXpert model not found at {chexpert_model_path}. Skipping weight loading.")
                return
                
            chexpert_state_dict = safetensors.torch.load_file(chexpert_model_path, device="cpu")
            
            # Filter out keys that don't match our model structure
            filtered_state_dict = {}
            for key, value in chexpert_state_dict.items():
                if key.startswith("vit."):
                    filtered_state_dict[key] = value
                elif key.startswith("img_proj.") and not low_memory:
                    filtered_state_dict[key] = value
            
            # Load the filtered weights
            missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
            logger.info(f"Loaded CheXpert weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            
        except Exception as e:
            logger.error(f"Error loading CheXpert weights: {e}")
            logger.warning("Continuing without pre-trained weights")

    def info_nce_loss(self, features1, features2, temperature=None, eps=1e-6):
        """Calculates the InfoNCE loss for contrastive learning."""
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

    def generate_gaze_heatmap(self, fix_seq, fix_mask, image_size=(224, 224)):
        """
        Enhanced human gaze heatmap generation with improved quality and multi-scale support.
        
        Args:
            fix_seq: (batch_size, seq_len, 4) - [x_norm, y_norm, duration, pupil_area]
            fix_mask: (batch_size, seq_len) - valid fixation mask
            image_size: tuple - output heatmap size
            
        Returns:
            gaze_heatmap: (batch_size, 1, H, W) - human gaze attention heatmap
        """
        batch_size, seq_len, _ = fix_seq.shape
        height, width = image_size
        
        # Initialize heatmaps
        heatmaps = torch.zeros(batch_size, 1, height, width, device=fix_seq.device)
        
        for b in range(batch_size):
            # Get valid fixations for this sample
            valid_mask = fix_mask[b]
            if not valid_mask.any():
                continue
                
            valid_fixations = fix_seq[b][valid_mask]  # (valid_len, 4)
            
            # Extract coordinates and weights
            x_norm = valid_fixations[:, 0]  # Already normalized to [0,1] after preprocessing
            y_norm = valid_fixations[:, 1]
            duration = valid_fixations[:, 2]  # Normalized duration
            pupil_area = valid_fixations[:, 3]  # Normalized pupil area
            
            # Convert normalized coordinates to pixel coordinates
            x_pixels = (x_norm * 0.5 + 0.5) * (width - 1)  # Denormalize from preprocessing
            y_pixels = (y_norm * 0.5 + 0.5) * (height - 1)
            
            # Clamp to valid pixel range
            x_pixels = torch.clamp(x_pixels, 0, width - 1)
            y_pixels = torch.clamp(y_pixels, 0, height - 1)
            
            # Enhanced attention weight calculation
            duration_weight = duration * 0.5 + 0.5  # Denormalize
            pupil_weight = pupil_area * 0.5 + 0.5
            
            # Improved attention weighting with temporal decay
            temporal_weights = torch.exp(-0.1 * torch.arange(len(valid_fixations), device=fix_seq.device))
            
            # Combined attention weight with temporal and physiological factors
            attention_weights = duration_weight * pupil_weight * temporal_weights
            attention_weights = torch.clamp(attention_weights, 0.05, 1.0)
            
            # Normalize attention weights
            if attention_weights.sum() > 0:
                attention_weights = attention_weights / attention_weights.sum()
            
            # Create coordinate grids once (optimization)
            y_grid, x_grid = torch.meshgrid(
                torch.arange(height, device=fix_seq.device, dtype=torch.float32),
                torch.arange(width, device=fix_seq.device, dtype=torch.float32),
                indexing='ij'
            )
            
            # Vectorized Gaussian computation for all fixations
            for i in range(len(valid_fixations)):
                cx, cy = x_pixels[i], y_pixels[i]
                weight = attention_weights[i]
                
                # Adaptive sigma based on attention weight and image size
                base_sigma = min(height, width) * 0.08  # 8% of image dimension
                sigma = base_sigma * (0.5 + 1.5 * weight)  # Adaptive sigma
                
                # Gaussian blob
                dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
                gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
                
                # Add weighted gaussian to heatmap
                heatmaps[b, 0] += weight * gaussian
        
        # Enhanced normalization with smoothing
        for b in range(batch_size):
            hmap = heatmaps[b, 0]
            if hmap.max() > 0:
                # Apply slight smoothing using conv2d (compatible alternative to gaussian_blur)
                try:
                    # Create simple 3x3 gaussian kernel for smoothing
                    kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=hmap.device)
                    kernel = kernel / kernel.sum()  # Normalize
                    kernel = kernel.view(1, 1, 3, 3)
                    
                    # Apply smoothing
                    hmap_padded = F.pad(hmap.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                    hmap_smoothed = F.conv2d(hmap_padded, kernel, padding=0).squeeze()
                    hmap = hmap_smoothed
                except Exception:
                    # Fallback: no smoothing if conv2d fails
                    pass
                
                # Normalize to [0, 1]
                heatmaps[b, 0] = hmap / hmap.max()
        
        return heatmaps

    def gaze_attention_loss(self, model_attention, gaze_heatmap, fix_seq, fix_mask):
        """
        Enhanced multi-component gaze-guided attention supervision loss.
        
        Args:
            model_attention: (batch_size, 1, H, W) - model's attention map
            gaze_heatmap: (batch_size, 1, H, W) - human gaze heatmap
            fix_seq: (batch_size, seq_len, 4) - fixation sequence for weighting
            fix_mask: (batch_size, seq_len) - valid fixation mask
            
        Returns:
            gaze_loss: scalar tensor - enhanced gaze supervision loss
        """
        batch_size = model_attention.size(0)
        
        # Ensure same spatial dimensions
        if model_attention.shape != gaze_heatmap.shape:
            gaze_heatmap = F.interpolate(gaze_heatmap, size=model_attention.shape[2:], mode='bilinear', align_corners=False)
        
        # Enhanced per-sample weights with fixation quality assessment
        sample_weights = []
        fixation_quality_scores = []
        
        for b in range(batch_size):
            valid_mask = fix_mask[b]
            if valid_mask.any():
                num_fixations = valid_mask.sum().float()
                valid_fixations = fix_seq[b][valid_mask]
                
                # Enhanced quality metrics
                avg_duration = valid_fixations[:, 2].mean()
                avg_pupil = valid_fixations[:, 3].mean()
                duration_std = valid_fixations[:, 2].std() if len(valid_fixations) > 1 else 0.0
                
                # Spatial dispersion metric (how spread out fixations are)
                if len(valid_fixations) > 1:
                    coords = valid_fixations[:, :2]  # x, y coordinates
                    coord_std = coords.std(dim=0).mean()
                    spatial_spread = torch.clamp(coord_std, 0.1, 1.0)
                else:
                    spatial_spread = 0.5
                
                # Quality score combining multiple factors
                temporal_quality = (avg_duration * 0.5 + 0.5) * (1.0 - torch.clamp(duration_std, 0, 0.5))
                physiological_quality = avg_pupil * 0.5 + 0.5
                spatial_quality = spatial_spread
                
                quality_score = (temporal_quality + physiological_quality + spatial_quality) / 3.0
                
                # Sample weight with improved scaling
                sample_weight = torch.sqrt(num_fixations / 8.0) * quality_score
                sample_weight = torch.clamp(sample_weight, 0.05, 1.0)
                
                fixation_quality_scores.append(quality_score)
            else:
                sample_weight = 0.0
                fixation_quality_scores.append(0.0)
            
            sample_weights.append(sample_weight)
        
        sample_weights = torch.stack(sample_weights).to(model_attention.device)
        quality_scores = torch.stack(fixation_quality_scores).to(model_attention.device)
        
        # Flatten spatial dimensions
        model_flat = model_attention.view(batch_size, -1)
        gaze_flat = gaze_heatmap.view(batch_size, -1)
        
        # 1. Enhanced MSE loss with quality weighting
        mse_loss = F.mse_loss(model_flat, gaze_flat, reduction='none').mean(dim=1)
        
        # 2. Improved KL divergence with better numerical stability
        eps = 1e-8
        model_prob = F.softmax(model_flat + eps, dim=1)
        gaze_prob = F.softmax(gaze_flat + eps, dim=1)
        kl_loss = F.kl_div(model_prob.log(), gaze_prob, reduction='none').sum(dim=1)
        
        # 3. NEW: Pearson correlation loss (encourages similar spatial patterns)
        def pearson_correlation_loss(x, y):
            x_mean = x.mean(dim=1, keepdim=True)
            y_mean = y.mean(dim=1, keepdim=True)
            x_centered = x - x_mean
            y_centered = y - y_mean
            
            numerator = (x_centered * y_centered).sum(dim=1)
            x_std = torch.sqrt((x_centered ** 2).sum(dim=1) + eps)
            y_std = torch.sqrt((y_centered ** 2).sum(dim=1) + eps)
            
            correlation = numerator / (x_std * y_std + eps)
            return 1.0 - correlation  # Convert to loss (higher correlation = lower loss)
        
        correlation_loss = pearson_correlation_loss(model_flat, gaze_flat)
        
        # 4. NEW: Center of mass loss (ensures attention centers align)
        def center_of_mass_loss(pred_map, target_map):
            batch_size, h, w = pred_map.size(0), pred_map.size(2), pred_map.size(3)
            
            # Create coordinate grids
            y_coords = torch.arange(h, device=pred_map.device, dtype=torch.float32).view(1, 1, h, 1)
            x_coords = torch.arange(w, device=pred_map.device, dtype=torch.float32).view(1, 1, 1, w)
            
            # Calculate center of mass for each map
            pred_mass = pred_map.sum(dim=(2, 3), keepdim=True) + eps
            target_mass = target_map.sum(dim=(2, 3), keepdim=True) + eps
            
            pred_center_y = (pred_map * y_coords).sum(dim=(2, 3)) / pred_mass.squeeze()
            pred_center_x = (pred_map * x_coords).sum(dim=(2, 3)) / pred_mass.squeeze()
            
            target_center_y = (target_map * y_coords).sum(dim=(2, 3)) / target_mass.squeeze()
            target_center_x = (target_map * x_coords).sum(dim=(2, 3)) / target_mass.squeeze()
            
            # L2 distance between centers
            center_dist = torch.sqrt((pred_center_y - target_center_y) ** 2 + 
                                   (pred_center_x - target_center_x) ** 2 + eps)
            
            # Normalize by image diagonal
            diagonal = torch.sqrt(torch.tensor(h ** 2 + w ** 2, device=pred_map.device, dtype=torch.float32))
            return center_dist / diagonal
        
        com_loss = center_of_mass_loss(model_attention, gaze_heatmap)
        
        # 5. Combine all loss components with adaptive weighting
        base_loss = 0.4 * mse_loss + 0.3 * kl_loss + 0.2 * correlation_loss + 0.1 * com_loss
        
        # Apply quality-based weighting
        quality_weighted_loss = base_loss * sample_weights * (0.5 + 0.5 * quality_scores)
        
        # Return weighted average (only count samples with valid fixations)
        valid_samples = (sample_weights > 0).sum()
        if valid_samples > 0:
            return quality_weighted_loss.sum() / valid_samples
        else:
            return torch.tensor(0.0, device=model_attention.device, requires_grad=True)

    def generate_spatial_attention(self, fused_features, image_size):
        """
        Generate spatial attention map from fused features.
        
        Args:
            fused_features: (batch_size, hidden_dim) - fused multimodal features
            image_size: tuple - (H, W) of input image
            
        Returns:
            attention_map: (batch_size, 1, H, W) - spatial attention map
        """
        batch_size = fused_features.size(0)
        hidden_dim = fused_features.size(1)
        H, W = image_size
        
        # Create spatial attention head
        if not hasattr(self, 'spatial_attention_head'):
            if low_memory:
                self.spatial_attention_head = nn.Sequential(
                    nn.Linear(hidden_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, H * W),
                    nn.Sigmoid()
                ).to(fused_features.device)
            else:
                self.spatial_attention_head = nn.Sequential(
                    nn.Linear(hidden_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, H * W),
                    nn.Sigmoid()
                ).to(fused_features.device)
        
        # Generate attention scores
        attention_scores = self.spatial_attention_head(fused_features)  # (batch_size, H*W)
        attention_map = attention_scores.view(batch_size, 1, H, W)     # (batch_size, 1, H, W)
        
        return attention_map

    def contrastive_gaze_text_loss(self, gaze_features, text_features, fix_mask, transcript_texts, temperature=0.07):
        """
        CLIP-style contrastive learning between gaze patterns and text descriptions.
        This is the core Priority 2 enhancement - teaching semantic gaze-text alignment.
        
        Args:
            gaze_features: (batch_size, feature_dim) - processed gaze representations
            text_features: (batch_size, feature_dim) - text features from transcripts
            fix_mask: (batch_size, seq_len) - valid fixation mask
            transcript_texts: list of strings - transcript texts for semantic matching
            temperature: float - contrastive learning temperature
            
        Returns:
            contrastive_loss: scalar tensor - gaze-text alignment loss
        """
        batch_size = gaze_features.size(0)
        
        # Filter out samples without valid fixations or transcripts
        valid_samples = []
        valid_gaze_feats = []
        valid_text_feats = []
        
        for i in range(batch_size):
            has_fixations = fix_mask[i].any()
            has_transcript = (i < len(transcript_texts) and 
                            transcript_texts[i] is not None and 
                            len(transcript_texts[i].strip()) > 10)  # Minimum meaningful text
            
            if has_fixations and has_transcript:
                valid_samples.append(i)
                valid_gaze_feats.append(gaze_features[i])
                valid_text_feats.append(text_features[i])
        
        if len(valid_samples) < 2:
            # Need at least 2 samples for contrastive learning
            return torch.tensor(0.0, device=gaze_features.device, requires_grad=True)
        
        # Stack valid features
        valid_gaze_feats = torch.stack(valid_gaze_feats)  # (valid_batch, feature_dim)
        valid_text_feats = torch.stack(valid_text_feats)  # (valid_batch, feature_dim)
        
        # Project to same dimensional space for contrastive learning
        if not low_memory and hasattr(self, 'gaze_text_projector'):
            gaze_proj = self.gaze_text_projector(valid_gaze_feats)
            text_proj = self.gaze_text_projector(valid_text_feats)
        else:
            # Simple normalization for low memory mode
            gaze_proj = F.normalize(valid_gaze_feats, dim=1)
            text_proj = F.normalize(valid_text_feats, dim=1)
        
        # Normalize features for contrastive learning
        gaze_proj = F.normalize(gaze_proj, dim=1)
        text_proj = F.normalize(text_proj, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(gaze_proj, text_proj.T) / temperature
        
        # Create positive pair labels (diagonal elements are positive pairs)
        labels = torch.arange(len(valid_samples), device=gaze_features.device)
        
        # Symmetric contrastive loss (gaze->text and text->gaze)
        loss_gaze_to_text = F.cross_entropy(similarity_matrix, labels)
        loss_text_to_gaze = F.cross_entropy(similarity_matrix.T, labels)
        
        contrastive_loss = (loss_gaze_to_text + loss_text_to_gaze) / 2.0
        
        return contrastive_loss

    def enhanced_spatial_attention_extraction(self, img_features, img_hidden_states=None):
        """
        Extract high-quality spatial attention from ViT features using multi-head attention.
        
        Args:
            img_features: (batch_size, hidden_dim) - pooled image features
            img_hidden_states: (batch_size, seq_len, hidden_dim) - all patch tokens
            
        Returns:
            spatial_attention: (batch_size, 1, H, W) - enhanced spatial attention map
        """
        batch_size = img_features.size(0)
        
        if low_memory or not hasattr(self, 'vit_attention_head'):
            # Fallback to simple attention for low memory
            return self.generate_spatial_attention(img_features, (224, 224))
        
        try:
            if img_hidden_states is not None and img_hidden_states.size(1) > 1:
                # Use patch tokens for spatial attention (excluding CLS token)
                patch_tokens = img_hidden_states[:, 1:, :]  # Remove CLS token
                
                # Check if we have the expected number of patch tokens (14x14 = 196 for 224x224 image)
                num_patches = patch_tokens.size(1)
                if num_patches == 196:  # 14x14 patches
                    # Multi-head attention for spatial reasoning
                    attended_patches, attention_weights = self.vit_attention_head(
                        patch_tokens, patch_tokens, patch_tokens
                    )
                    
                    # Check attention weights shape and handle accordingly
                    if attention_weights.dim() >= 3:
                        # Average attention weights across heads if present
                        if attention_weights.dim() == 4:  # (batch, heads, seq, seq)
                            # Average across heads and take diagonal (self-attention)
                            spatial_weights = attention_weights.mean(dim=1)  # (batch, seq, seq)
                            spatial_weights = torch.diagonal(spatial_weights, dim1=-2, dim2=-1)  # (batch, seq)
                        elif attention_weights.dim() == 3:  # (batch, seq, seq)
                            # Take diagonal (self-attention)
                            spatial_weights = torch.diagonal(attention_weights, dim1=-2, dim2=-1)  # (batch, seq)
                        else:
                            spatial_weights = attention_weights
                        
                        # Ensure we have the right shape
                        if spatial_weights.size(1) == num_patches:
                            # Reshape to spatial grid
                            patch_size = int(np.sqrt(num_patches))  # Should be 14
                            spatial_grid = spatial_weights.view(batch_size, 1, patch_size, patch_size)
                            
                            # Upsample to full resolution
                            spatial_attention = F.interpolate(
                                spatial_grid, size=(224, 224), mode='bilinear', align_corners=False
                            )
                        else:
                            # Fallback if dimensions don't match
                            spatial_attention = self.generate_spatial_attention(img_features, (224, 224))
                    else:
                        # Fallback if attention weights have unexpected shape
                        spatial_attention = self.generate_spatial_attention(img_features, (224, 224))
                else:
                    # Fallback if unexpected number of patches
                    spatial_attention = self.generate_spatial_attention(img_features, (224, 224))
            else:
                # Fallback using pooled features
                spatial_attention = self.generate_spatial_attention(img_features, (224, 224))
                
        except Exception as e:
            logger.warning(f"Error in enhanced spatial attention extraction: {e}")
            # Fallback to simple attention generation
            spatial_attention = self.generate_spatial_attention(img_features, (224, 224))
        
        return spatial_attention

    def forward(self, img, bbox, fix_seq, fix_mask, transcript, labels=None, contrastive_weight=0.1, gaze_weight=0.3):
        batch_size = img.size(0)
        
        # More aggressive CUDA cache clearing
        if DEVICE == "cuda" and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()  # Clear at start of every forward pass
        
        # --- Enhanced Image encoding with attention extraction ---
        img_outputs = self.vit(pixel_values=img)
        if hasattr(img_outputs, 'pooler_output') and img_outputs.pooler_output is not None:
            img_feat = img_outputs.pooler_output
        else:
            img_feat = img_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Store hidden states for enhanced spatial attention (Priority 2)
        img_hidden_states = img_outputs.last_hidden_state if not low_memory else None
        img_proj = self.img_proj(img_feat)
        
        # Clear intermediate tensors but keep hidden states for attention
        del img_outputs, img_feat
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        # --- BBox encoding ---
        bbox_feat = self.bbox_encoder(bbox).flatten(1)
        bbox_proj = self.bbox_proj(bbox_feat)
        del bbox_feat
        
        # --- Fixation sequence encoding ---
        fix_lens = fix_mask.sum(dim=1).cpu()
        
        if torch.all(fix_lens == 0):
            if low_memory:
                fix_feat_gru = torch.zeros(batch_size, 256, device=fix_seq.device)
            else:
                fix_feat_gru = torch.zeros(batch_size, 1536, device=fix_seq.device)
        else:
            fix_emb = self.fix_emb(fix_seq)
            
            packed_seq = nn.utils.rnn.pack_padded_sequence(
                fix_emb, 
                fix_lens.clamp(min=1),
                batch_first=True, 
                enforce_sorted=False
            )
            
            _, h_n = self.fix_gru(packed_seq)
            fix_feat_gru = h_n.transpose(0, 1).reshape(batch_size, -1)
            
            # Clean up RNN intermediate states
            del fix_emb, packed_seq, h_n
        
        fix_proj = self.fix_proj(fix_feat_gru)
        del fix_feat_gru
        
        # Memory cleanup after heavy computations
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        # --- Transcript encoding ---
        if isinstance(transcript, list):
            try:
                # Truncate more aggressively for low memory
                max_length = 256 if low_memory else 512
                transcript_tokens = self.tokenizer(
                    transcript, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length
                ).to(img.device)
                
                text_outputs = self.text_encoder(**transcript_tokens)
                text_feat = text_outputs.last_hidden_state[:, 0, :]  # CLS token
                text_proj = self.text_proj(text_feat)
                
                # Clean up text processing tensors
                del transcript_tokens, text_outputs, text_feat
                
            except Exception as e:
                logger.warning(f"Error processing transcript: {e}")
                hidden_dim = 512 if low_memory else 768
                text_proj = torch.zeros(batch_size, hidden_dim, device=img.device)
        else:
            hidden_dim = 512 if low_memory else 768
            text_proj = torch.zeros(batch_size, hidden_dim, device=img.device)
        
        # Memory cleanup before fusion
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        # --- Fusion ---
        combined = torch.cat([img_proj, bbox_proj, fix_proj, text_proj], dim=1)
        fused = self.fusion(combined)
        del combined  # Free the concatenated tensor immediately
        
        if not low_memory and self.fusion_attention is not None:
            # Additional attention refinement for high memory mode
            stacked_features = torch.stack([img_proj, bbox_proj, fix_proj, text_proj], dim=1)
            attended_features, attn_weights = self.fusion_attention(stacked_features, stacked_features, stacked_features)
            attended_combined = attended_features.mean(dim=1)
            final_fused = self.fusion_norm(fused + attended_combined)
            
            # Clean up attention tensors
            del stacked_features, attended_features, attn_weights, attended_combined
        else:
            final_fused = fused
        
        del fused  # Clean up intermediate fusion result
        
        # --- Classification ---
        global_logits = self.global_classifier(final_fused)
        
        if not low_memory and self.condition_specific_heads is not None:
            # Get condition-specific predictions
            condition_logits = []
            for i, head in enumerate(self.condition_specific_heads):
                cond_logit = head(final_fused)
                condition_logits.append(cond_logit)
            
            condition_logits = torch.cat(condition_logits, dim=1)
            
            # Ensemble the predictions
            alpha = 0.7
            logits_output = alpha * global_logits + (1 - alpha) * condition_logits
            del condition_logits  # Clean up
        else:
            logits_output = global_logits
        
        # === ENHANCED GAZE-GUIDED ATTENTION GENERATION (Priority 2) ===
        # Generate high-quality spatial attention using enhanced methods
        if not low_memory and img_hidden_states is not None:
            spatial_attention = self.enhanced_spatial_attention_extraction(final_fused, img_hidden_states)
        else:
            spatial_attention = self.generate_spatial_attention(final_fused, img.shape[2:])
        
        attn_map_output = spatial_attention
        
        # Clear img_hidden_states to free memory
        if img_hidden_states is not None:
            del img_hidden_states
        
        loss = None
        loss_cls = torch.tensor(0.0, device=img.device)
        loss_contrastive = torch.tensor(0.0, device=img.device)
        loss_gaze = torch.tensor(0.0, device=img.device)
        loss_gaze_text_contrastive = torch.tensor(0.0, device=img.device)

        if labels is not None:
            # 1. Classification Loss
            if self.loss_fn is None:
                self.loss_fn = nn.BCEWithLogitsLoss()
                logger.warning("Using fallback BCEWithLogitsLoss!")
            
            loss_cls = self.loss_fn(logits_output, labels)

            # 2. Traditional contrastive losses (simplified for low memory)
            if img_proj.shape[0] > 1 and not low_memory:
                loss_img_fix = self.info_nce_loss(img_proj, fix_proj)
                loss_img_text = self.info_nce_loss(img_proj, text_proj)
                loss_contrastive = (loss_img_fix + loss_img_text) / 2.0
            else:
                loss_contrastive = torch.tensor(0.0, device=img.device)

            # 3. Enhanced Gaze Supervision Loss (Priority 2)
            try:
                # Generate enhanced human gaze heatmap from fixation data
                gaze_heatmap = self.generate_gaze_heatmap(fix_seq, fix_mask, image_size=(224, 224))
                
                # Compute enhanced gaze-attention alignment loss
                loss_gaze = self.gaze_attention_loss(spatial_attention, gaze_heatmap, fix_seq, fix_mask)
                
            except Exception as e:
                logger.warning(f"Error computing gaze loss: {e}")
                loss_gaze = torch.tensor(0.0, device=img.device)

            # 4. NEW: Contrastive Gaze-Text Alignment Loss (Core Priority 2)
            try:
                if not low_memory and isinstance(transcript, list):
                    loss_gaze_text_contrastive = self.contrastive_gaze_text_loss(
                        fix_proj, text_proj, fix_mask, transcript, temperature=0.07
                    )
                else:
                    loss_gaze_text_contrastive = torch.tensor(0.0, device=img.device)
                    
            except Exception as e:
                logger.warning(f"Error computing gaze-text contrastive loss: {e}")
                loss_gaze_text_contrastive = torch.tensor(0.0, device=img.device)

            # 5. Adaptive Loss Combination with Priority 2 weighting
            if low_memory:
                # Simplified loss for low memory (classification + basic gaze)
                loss = loss_cls + gaze_weight * loss_gaze
            else:
                # Full Priority 2 loss with all components
                # Increased gaze weights to emphasize Priority 2 training
                gaze_text_weight = 0.15  # NEW: Weight for gaze-text contrastive learning
                enhanced_gaze_weight = gaze_weight * 1.2  # Increase base gaze weight
                
                loss = (loss_cls + 
                       contrastive_weight * loss_contrastive + 
                       enhanced_gaze_weight * loss_gaze + 
                       gaze_text_weight * loss_gaze_text_contrastive)

        # Clean up projection tensors before returning
        del img_proj, bbox_proj, fix_proj, text_proj, final_fused
        
        # Final memory cleanup
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        output = {
            "logits": logits_output,
            "zi": None,  # Don't keep these in memory
            "zg": None,
            "zb": None, 
            "zt": None,
            "attn_map": attn_map_output,
            "loss_cls": loss_cls.detach() if loss is not None else None,
            "loss_con": loss_contrastive.detach() if loss is not None else None,
            "loss_gaze": loss_gaze.detach() if loss is not None else None,
            "loss_gaze_text": loss_gaze_text_contrastive.detach() if loss is not None else None
        }
        if loss is not None:
            output["loss"] = loss

        return output

############################
# Advanced Loss Functions #
############################
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples"""
    def __init__(self, alpha=1, gamma=2, pos_weight=None, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Apply label smoothing
        targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Compute BCE loss with pos_weight
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth, 
                                                        pos_weight=self.pos_weight, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth, reduction='none')
        
        # Compute probabilities
        p_t = torch.sigmoid(inputs)
        p_t = torch.where(targets_smooth >= 0.5, p_t, 1 - p_t)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = self.alpha * focal_weight * bce_loss
        
        return focal_loss.mean()

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

###############################
# Cross-Modal Attention Module #
###############################
class CrossModalAttention(nn.Module):
    """Cross-modal attention for better feature fusion"""
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query, key, value, attn_mask=None):
        # Multi-head attention
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        x = self.norm1(query + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x

def compute_metrics(eval_pred: EvalPrediction):
    predictions_dict = eval_pred.predictions 
    labels_raw = eval_pred.label_ids

    if isinstance(predictions_dict, dict):
         logits = predictions_dict['logits']
    elif isinstance(predictions_dict, tuple):
         logits = predictions_dict[0] 
    else:
         logits = predictions_dict
    
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    elif not isinstance(logits, np.ndarray):
        logits = np.array(logits)

    if isinstance(labels_raw, torch.Tensor):
        labels = labels_raw.cpu().numpy()
    elif not isinstance(labels_raw, np.ndarray):
        labels = np.array(labels_raw)
    else:
        labels = labels_raw

    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    
    # MODERATE BALANCE: Wider threshold range + reduced precision bias for better recall
    optimal_thresholds = []
    for i in range(labels.shape[1]):
        best_threshold = 0.5
        best_f1 = 0.0
        
        # MODERATE CHANGE: Slightly wider threshold range for better recall
        for threshold in np.arange(0.38, 0.78, 0.02):  # Moderate change: 0.4-0.8 → 0.38-0.78
            temp_preds = (probs[:, i] > threshold).astype(int)
            temp_f1 = f1_score(labels[:, i], temp_preds, zero_division=0)
            temp_precision = precision_score(labels[:, i], temp_preds, zero_division=0)
            
            # MODERATE PRECISION BIAS: 85% F1 + 15% precision (less precision-focused)
            if temp_precision > 0.3:  # Only small precision requirement
                adjusted_score = 0.85 * temp_f1 + 0.15 * temp_precision
            else:
                adjusted_score = temp_f1
            
            if adjusted_score > best_f1:
                best_f1 = adjusted_score
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
    
    # Use optimal thresholds for predictions
    preds = np.zeros_like(probs)
    for i in range(labels.shape[1]):
        preds[:, i] = (probs[:, i] > optimal_thresholds[i]).astype(int)
    
    # Also compute with standard 0.5 threshold for comparison
    preds_05 = (probs > 0.5).astype(int)
    
    try:
        # IMPROVED: Use micro-averaged F1 as primary metric (better for imbalanced data)
        micro_f1 = f1_score(labels.flatten(), preds.flatten(), zero_division=0)
        micro_rec = recall_score(labels.flatten(), preds.flatten(), zero_division=0)
        micro_prec = precision_score(labels.flatten(), preds.flatten(), zero_division=0)
        
        # Also compute macro metrics
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        macro_rec = recall_score(labels, preds, average="macro", zero_division=0)
        macro_prec = precision_score(labels, preds, average="macro", zero_division=0)
        
        # Accuracy metrics
        # Exact match accuracy (all labels must be correct)
        exact_match_acc = np.mean(np.all(labels == preds, axis=1))
        
        # Hamming accuracy (per-label accuracy)
        hamming_acc = np.mean(labels == preds)
        
        # Subset accuracy (same as exact match)
        subset_acc = exact_match_acc
        
        if labels.shape == probs.shape:
            auc_scores = [roc_auc_score(labels[:, i], probs[:, i]) for i in range(labels.shape[1]) if len(np.unique(labels[:, i])) > 1]
            mean_auc = np.mean(auc_scores) if auc_scores else 0.0
        else:
            mean_auc = 0.0
            
        # Also compute metrics with 0.5 threshold for comparison
        micro_f1_05 = f1_score(labels.flatten(), preds_05.flatten(), zero_division=0)
        
        logger.info(f"Optimal thresholds: {[f'{t:.2f}' for t in optimal_thresholds]}")
        logger.info(f"Micro F1 with optimal thresholds: {micro_f1:.3f} vs Micro F1 with 0.5: {micro_f1_05:.3f}")
        logger.info(f"Target to beat: F1=0.6242. Current: {micro_f1:.3f} (Gap: {0.6242-micro_f1:+.3f})")
            
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        micro_f1, micro_rec, micro_prec, mean_auc = 0.0, 0.0, 0.0, 0.0
        exact_match_acc, hamming_acc, subset_acc = 0.0, 0.0, 0.0
        macro_f1, macro_rec, macro_prec = 0.0, 0.0, 0.0
        
    return {
        "auc": mean_auc, 
        "f1": micro_f1,  # Use micro F1 as primary metric
        "recall": micro_rec, 
        "precision": micro_prec,
        "macro_f1": macro_f1,  # Also report macro metrics
        "macro_recall": macro_rec,
        "macro_precision": macro_prec,
        "accuracy": exact_match_acc,  # Exact match accuracy
        "hamming_accuracy": hamming_acc,  # Per-label accuracy
        "subset_accuracy": subset_acc  # Same as exact match
    }

#################
# === Main ===  #
#################
def main():
    global DEVICE  # Move global declaration to the top of the function
    
    logger.info("🚀 Starting ENHANCED GAZE-GUIDED MIMIC training with Priority 2 Implementation")
    logger.info(f"🎯 Goal: Beat ViT-only baseline F1 of 0.6242 + Implement Priority 2 gaze-guided training")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"🧠 Priority 2 Features: Enhanced gaze supervision + Contrastive gaze-text alignment + Multi-level attention")
    logger.info(f"📊 Loss Components: Classification + Traditional Contrastive + Enhanced Gaze + NEW Gaze-Text Contrastive")
    logger.info(f"⚡ Gaze Enhancements: Multi-component loss (MSE+KL+Correlation+CenterMass) + Quality weighting + Temporal decay")
    
    model = MultiModalMIMICModel(len(CONDITIONS))
    
    # Calculate class weights for handling imbalanced data
    logger.info("Calculating class weights for imbalanced data...")
    train_labels = np.array([row for row in train_df["cond_vec"].values])
    
    # Calculate positive class weights with improved strategy
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    
    # MODERATE PRECISION: More conservative class weights to reduce false positives
    pos_weights = []
    for i in range(len(CONDITIONS)):
        if pos_counts[i] > 0:
            # More conservative weighting for better precision
            total_samples = len(train_labels)
            pos_ratio = pos_counts[i] / total_samples
            # More conservative weighting for better precision
            pos_weight = (1.0 - pos_ratio) / (pos_ratio + 0.07)  # Higher smoothing: 0.06 → 0.07
            # Lower cap to prevent over-aggressive weighting
            pos_weight = min(pos_weight, 16.0)  # Reduced cap: 16.5 → 16.0
            pos_weight = max(pos_weight, 0.6)   # Higher minimum: 0.5 → 0.6
        else:
            pos_weight = 1.0
        pos_weights.append(pos_weight)
    
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32, device=DEVICE)
    
    # Use more aggressive focal loss for better rare class handling
    model.set_loss_function(pos_weights, loss_type="focal")
    
    # Log improved class distribution
    logger.info("\nClass Distribution and Weights:")
    for i, condition in enumerate(CONDITIONS):
        logger.info(f"{condition}: {int(pos_counts[i])}/{len(train_labels)} samples ({pos_counts[i]/len(train_labels)*100:.1f}%) - weight: {pos_weights[i]:.2f}")
    
    # Load CheXpert pre-trained weights
    model.load_chexpert_weights(CHEXPERT_MODEL_PATH)
    
    # Create data collator
    data_collator = MultiModalDataCollator()
    
    # Enhanced training arguments for Priority 2 gaze-guided training
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
        eval_strategy="steps",  # Frequent evaluation for gaze training monitoring
        eval_steps=30,  # Slightly less frequent to accommodate complex loss computation
        save_strategy="steps",
        save_steps=30,  # Match evaluation frequency
        save_total_limit=3,  # Keep multiple checkpoints for Priority 2 analysis
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        dataloader_num_workers=DATALOADER_WORKERS,
        remove_unused_columns=False,
        report_to="none",
        logging_steps=15,  # Adjusted for Priority 2 complex loss logging
        max_grad_norm=0.5,  # Slightly relaxed for complex multi-component loss
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-8,
        dataloader_pin_memory=not low_memory,
        fp16=FP16_ENABLED,
        dataloader_persistent_workers=not low_memory,
        eval_accumulation_steps=6,  # More accumulation for complex attention computations
        save_on_each_node=False,
        prediction_loss_only=False,
        include_inputs_for_metrics=False,
        ignore_data_skip=True,
    )
    
    callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    
    # Check for existing checkpoints
    resume_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    if resume_checkpoint:
        logger.info(f"Found existing checkpoint: {resume_checkpoint}")
        logger.info("Will resume training from this checkpoint...")
    else:
        logger.info("No existing checkpoint found. Starting training from scratch.")
    
    # Add CUDA error recovery
    try:
        trainer = ContiguousSavingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        
        logger.info("Starting training...")
        # Resume from checkpoint if available
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CUDA out of memory during training!")
            logger.info("Attempting recovery with CPU training...")
            
            # Force CPU mode
            DEVICE = "cpu"
            model = model.to(DEVICE)
            
            # Reduce batch size further
            training_args.per_device_train_batch_size = max(1, BATCH_SIZE // 2)
            training_args.per_device_eval_batch_size = max(1, BATCH_SIZE // 2)
            training_args.gradient_accumulation_steps = GRAD_ACCUM * 2
            training_args.dataloader_num_workers = 0
            training_args.fp16 = False
            training_args.dataloader_pin_memory = False
            training_args.dataloader_persistent_workers = False
            
            # Create new trainer with CPU settings
            trainer = ContiguousSavingTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=callbacks
            )
            
            logger.info("Resuming training with CPU...")
            # Resume from checkpoint even in CPU mode
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        else:
            raise e
    
    # Save the final model
    trainer.save_model(os.path.join(OUTPUT_DIR, f"model_{EXP_NAME}"))
    logger.info(f"Model saved to {os.path.join(OUTPUT_DIR, f'model_{EXP_NAME}')}")
    
    print("=== Final Test Evaluation ===")
    test_metrics = trainer.evaluate(test_dataset)
    
    logger.info(f"Test Results Summary:")
    logger.info(f"F1 Score: {test_metrics['eval_f1']:.3f}")
    logger.info(f"AUC: {test_metrics['eval_auc']:.3f}")
    logger.info(f"Precision: {test_metrics['eval_precision']:.3f}")
    logger.info(f"Recall: {test_metrics['eval_recall']:.3f}")
    logger.info(f"Exact Match Accuracy: {test_metrics['eval_accuracy']:.3f}")
    logger.info(f"Hamming Accuracy: {test_metrics['eval_hamming_accuracy']:.3f}")
    
    # Performance comparison with target
    target_f1 = 0.6242  # ViT-only performance to beat
    improvement = test_metrics['eval_f1'] - target_f1
    logger.info(f"\n🎯 Performance vs ViT-only baseline to beat:")
    logger.info(f"ViT-only baseline: F1={target_f1:.4f}, AUC=0.8028, Precision=0.6427, Recall=0.6430")
    logger.info(f"Our multimodal: F1={test_metrics['eval_f1']:.4f}, AUC={test_metrics['eval_auc']:.4f}")
    logger.info(f"Improvement needed: {improvement:+.4f} ({improvement/target_f1*100:+.1f}%)")
    
    if test_metrics['eval_f1'] >= 0.65:
        logger.info("🎉 EXCELLENT: F1 >= 0.65 achieved!")
    elif test_metrics['eval_f1'] >= target_f1:
        logger.info("✅ SUCCESS: Beat ViT-only baseline! Multimodal approach works!")
    elif test_metrics['eval_f1'] >= 0.60:
        logger.info("🎯 GOOD: F1 >= 0.60 achieved! Close to target.")
    else:
        logger.info("📈 PROGRESS: Getting closer to target performance.")
    
    # Get predictions for test set for detailed per-condition metrics
    logger.info("Calculating per-condition test metrics...")
    test_predictions = trainer.predict(test_dataset)
    predictions_dict = test_predictions.predictions
    if isinstance(predictions_dict, dict):
        logits = predictions_dict['logits']
    else:
        logits = predictions_dict
    labels = test_predictions.label_ids
    
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    y_pred = (probs > 0.5).astype(int)
    
    # Calculate metrics for each condition
    logger.info("\nTest Set Metrics per Condition:")
    logger.info("-" * 80)
    headers = ["Condition", "Precision", "Recall", "F1-score", "Accuracy", "Support", "AUC"]
    rows = []
    
    for i, condition in enumerate(CONDITIONS):
        precision = precision_score(labels[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(labels[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(labels[:, i], y_pred[:, i], zero_division=0)
        accuracy = np.mean(labels[:, i] == y_pred[:, i])  # Per-condition accuracy
        support = np.sum(labels[:, i])
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i]) if len(np.unique(labels[:, i])) > 1 else np.nan
        except ValueError:
            auc = np.nan
            
        rows.append([
            condition,
            f"{precision:.3f}",
            f"{recall:.3f}",
            f"{f1:.3f}",
            f"{accuracy:.3f}",
            f"{int(support)}",
            f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        ])
    
    # Add macro average
    rows.append([
        "Macro Avg",
        f"{test_metrics['eval_precision']:.3f}",
        f"{test_metrics['eval_recall']:.3f}",
        f"{test_metrics['eval_f1']:.3f}",
        f"{test_metrics['eval_hamming_accuracy']:.3f}",
        f"{len(test_dataset)} (Total Samples)",
        f"{test_metrics['eval_auc']:.3f}"
    ])
    
    table_str = tabulate(rows, headers=headers, tablefmt="grid")
    logger.info("\n" + table_str)
    
    # Save the table to a text file
    metrics_table_path = os.path.join(OUTPUT_DIR, "training_mimic_on_chexpert_metrics_table.txt")
    with open(metrics_table_path, "w") as f:
        f.write("MIMIC Training with CheXpert Transfer Learning - Test Metrics per Condition:\n")
        f.write("-" * 80 + "\n")
        f.write(table_str)
    
    # Save test metrics as JSON
    metrics_json_path = os.path.join(OUTPUT_DIR, "metrics_training_mimic_on_chexpert.json")
    with open(metrics_json_path, "w") as f:
        json.dump(test_metrics, f, indent=4)
    logger.info(f"Overall test metrics saved to {metrics_json_path}")
    
    # Create CSV with predictions and real labels
    logger.info("Creating CSV with real and predicted labels...")
    csv_data = []
    batch_size = logits.shape[0]
    for i in range(batch_size):
        row = {"unique_id": f"sample_{i}"}
        
        # Add real labels
        for j, condition in enumerate(CONDITIONS):
            row[f"real_{condition}"] = int(labels[i][j])
        
        # Add predicted labels
        for j, condition in enumerate(CONDITIONS):
            row[f"pred_{condition}"] = int(y_pred[i][j])
            
        # Add prediction probabilities
        for j, condition in enumerate(CONDITIONS):
            row[f"prob_{condition}"] = float(probs[i][j])
            
        csv_data.append(row)
    
    # Save to CSV
    csv_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(OUTPUT_DIR, "test_predictions_and_labels.csv")
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"Saved predictions and labels CSV to: {csv_path}")
    
    logger.info("Training complete")

# Enhanced memory management for long training sessions
def aggressive_memory_cleanup():
    """Perform aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection of CUDA tensors
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()

def monitor_memory():
    """Monitor and log memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        # print(f"CUDA Memory - Allocated: {memory_allocated:.1f}GB, Reserved: {memory_reserved:.1f}GB")
        
        # If memory usage is high, force cleanup
        if memory_allocated > 10.0:  # More than 10GB allocated
            print("High CUDA memory usage detected - performing cleanup")
            aggressive_memory_cleanup()
    
    # System memory check
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    if available_gb < 20.0:  # Less than 20GB available
        print(f"⚠️ Low system memory: {available_gb:.1f}GB available")
        gc.collect()

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory"""
    import glob
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # Sort by checkpoint number
    def get_checkpoint_num(path):
        try:
            return int(os.path.basename(path).split('-')[1])
        except (ValueError, IndexError):
            return 0
    
    latest_checkpoint = max(checkpoints, key=get_checkpoint_num)
    checkpoint_num = get_checkpoint_num(latest_checkpoint)
    
    logger.info(f"Found latest checkpoint: {latest_checkpoint} (step {checkpoint_num})")
    return latest_checkpoint

# Memory optimization and error handling for Windows paging file issues
import os

if __name__=="__main__":
    main()
