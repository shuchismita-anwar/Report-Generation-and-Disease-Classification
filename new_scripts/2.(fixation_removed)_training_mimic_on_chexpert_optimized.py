"""
Multimodal MIMIC Training with CheXpert Transfer Learning
========================================================

Multimodal model combining chest X-ray images, bounding box masks, and radiology transcripts 
for medical condition classification.

Modalities: Image + BBox + Transcript (fixation removed)
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

# Updated paths as requested
IMG_DIR      = os.path.join("data_dump", "output", "img_png")
BBOX_DIR     = os.path.join("data_dump", "output", "bbox_mask")

# Model paths and names
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Path to the CheXpert-trained ViT model
CHEXPERT_MODEL_PATH = os.path.join("new_scripts", "output", "vit_chexpert_training_output", "model_1.chexpert_finetune_only_feat_image", "model.safetensors")

OUTPUT_DIR   = os.path.join("new_scripts", "output", "2.(fixation_removed)_training_mimic_on_chexpert_optimized")
EXP_NAME     = "2.(fixation_removed)_training_mimic_on_chexpert_optimized"

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
        
        # Simplified fusion for low memory
        if low_memory:
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),  # 512*3 -> 1024 (3 modalities now)
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
                nn.Linear(768 * 3, 1536),  # 768*3 -> 1536 (3 modalities now)
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
        """Set loss function for training"""
        if loss_type == "focal":
            self.loss_fn = FocalLoss(alpha=1.10, gamma=2.15, pos_weight=pos_weights, label_smoothing=0.08)
        elif loss_type == "asymmetric":
            self.loss_fn = AsymmetricLoss(
                gamma_neg=4,      
                gamma_pos=1,      
                clip=0.05,        
                disable_torch_grad_focal_loss=True
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
        logger.info(f"Set loss function: {loss_type}")

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

    def forward(self, img, bbox, transcript, labels=None, contrastive_weight=0.1):
        batch_size = img.size(0)
        
        # More aggressive CUDA cache clearing
        if DEVICE == "cuda" and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()  # Clear at start of every forward pass
        
        # --- Image encoding ---
        img_outputs = self.vit(pixel_values=img)
        if hasattr(img_outputs, 'pooler_output') and img_outputs.pooler_output is not None:
            img_feat = img_outputs.pooler_output
        else:
            img_feat = img_outputs.last_hidden_state[:, 0, :]  # CLS token
        img_proj = self.img_proj(img_feat)
        
        # Clear intermediate tensors
        del img_outputs, img_feat
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        # --- BBox encoding ---
        bbox_feat = self.bbox_encoder(bbox).flatten(1)
        bbox_proj = self.bbox_proj(bbox_feat)
        del bbox_feat
        
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
        combined = torch.cat([img_proj, bbox_proj, text_proj], dim=1)
        fused = self.fusion(combined)
        del combined  # Free the concatenated tensor immediately
        
        if not low_memory and self.fusion_attention is not None:
            # Additional attention refinement for high memory mode
            stacked_features = torch.stack([img_proj, bbox_proj, text_proj], dim=1)
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
        
        # Create attention map placeholder
        attn_map_output = torch.zeros_like(bbox)
        
        loss = None
        loss_cls = torch.tensor(0.0, device=img.device)
        loss_contrastive = torch.tensor(0.0, device=img.device)

        if labels is not None:
            # 1. Classification Loss
            if self.loss_fn is None:
                self.loss_fn = nn.BCEWithLogitsLoss()
                logger.warning("Using fallback BCEWithLogitsLoss!")
            
            loss_cls = self.loss_fn(logits_output, labels)

            # 2. Contrastive losses (simplified for low memory)
            if img_proj.shape[0] > 1 and not low_memory:
                loss_img_bbox = self.info_nce_loss(img_proj, bbox_proj)
                loss_img_text = self.info_nce_loss(img_proj, text_proj)
                loss_contrastive = (loss_img_bbox + loss_img_text) / 2.0
            else:
                loss_contrastive = torch.tensor(0.0, device=img.device)

            # 3. Combine losses
            if low_memory:
                loss = loss_cls  # Skip contrastive loss for low memory
            else:
                loss = loss_cls + contrastive_weight * loss_contrastive

        # Clean up projection tensors before returning
        del img_proj, bbox_proj, text_proj, final_fused
        
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
            "loss_con": loss_contrastive.detach() if loss is not None else None
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
    
    logger.info("🚀 Starting multimodal MIMIC training with CheXpert transfer learning")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Modalities: Image + BBox + Transcript")
    
    model = MultiModalMIMICModel(len(CONDITIONS))
    
    # Calculate class weights for handling imbalanced data
    logger.info("Calculating class weights for imbalanced data...")
    train_labels = np.array([row for row in train_df["cond_vec"].values])
    
    # Calculate positive class weights with improved strategy
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    
    pos_weights = []
    for i in range(len(CONDITIONS)):
        if pos_counts[i] > 0:
            total_samples = len(train_labels)
            pos_ratio = pos_counts[i] / total_samples
            pos_weight = (1.0 - pos_ratio) / (pos_ratio + 0.07)
            pos_weight = min(pos_weight, 16.0)
            pos_weight = max(pos_weight, 0.6)
        else:
            pos_weight = 1.0
        pos_weights.append(pos_weight)
    
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32, device=DEVICE)
    
    # Use focal loss for better class handling
    model.set_loss_function(pos_weights, loss_type="focal")
    
    # Log class distribution
    logger.info("\nClass Distribution and Weights:")
    for i, condition in enumerate(CONDITIONS):
        logger.info(f"{condition}: {int(pos_counts[i])}/{len(train_labels)} samples ({pos_counts[i]/len(train_labels)*100:.1f}%) - weight: {pos_weights[i]:.2f}")
    
    # Load CheXpert pre-trained weights
    model.load_chexpert_weights(CHEXPERT_MODEL_PATH)
    
    # Create data collator
    data_collator = MultiModalDataCollator()
    
    # Training arguments
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
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,  # Keep more checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        dataloader_num_workers=DATALOADER_WORKERS,
        remove_unused_columns=False,
        report_to="none",
        logging_steps=10,
        max_grad_norm=0.4,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-8,
        dataloader_pin_memory=not low_memory,  # Disable pin memory for low memory systems
        fp16=FP16_ENABLED,
        dataloader_persistent_workers=not low_memory,  # Disable for low memory systems
        eval_accumulation_steps=4,  # More accumulation to avoid memory spikes
        save_on_each_node=False,  # Avoid extra saving overhead
        prediction_loss_only=False,
        include_inputs_for_metrics=False,  # Reduce memory in metrics computation
        ignore_data_skip=True,  # Skip problematic samples rather than crash
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
