import os
import sys
import gc
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import safetensors
import logging
from typing import Optional, Union, Dict, List, Tuple
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import requests
import time
import base64
from io import BytesIO
import textwrap
from datetime import datetime
import random
from dotenv import load_dotenv

# Import real anatomical regions
try:
    from .keyword_prediction.anatomical_results.anatomical_regions_dict import ANATOMICAL_REGIONS
except ImportError:
    # Fallback for direct execution
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'keyword_prediction', 'anatomical_results'))
        from anatomical_regions_dict import ANATOMICAL_REGIONS
    except ImportError:
        # Final fallback - define empty regions
        ANATOMICAL_REGIONS = {}

# Constants from training script
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Edema',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pneumonia', 'Support Devices'
]

VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Real chest X-ray image directory - try multiple possible paths
def get_chest_xray_directory():
    """Find the correct chest X-ray directory based on current working directory"""
    possible_paths = [
        os.path.join("..", "data_dump", "output", "img_png"),  # From new_scripts/
        os.path.join("data_dump", "output", "img_png"),        # From root/
        os.path.join(".", "data_dump", "output", "img_png"),   # From root/ explicit
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            # Check if it has PNG files
            png_files = [f for f in os.listdir(abs_path) if f.lower().endswith('.png')]
            if png_files:
                return abs_path
    
    # Fallback to first path if none found
    return os.path.abspath(possible_paths[0])

CHEST_XRAY_DIR = get_chest_xray_directory()

# Load dynamic medical terminology mapping for keyword extraction
def load_condition_keywords():
    """Load condition keywords from the deduplicated JSON file"""
    try:
        # Define the path relative to the current file
        keywords_file_path = os.path.join(os.path.dirname(__file__), 'keyword_prediction', 'extracted_keywords_result_deduplicated.json')
        
        with open(keywords_file_path, 'r', encoding='utf-8') as f:
            keywords_data = json.load(f)
        
        # Extract the conditions section
        condition_keywords = keywords_data.get('conditions', {})
        
        logging.info(f"Successfully loaded keywords for {len(condition_keywords)} conditions from {keywords_file_path}")
        return condition_keywords
        
    except FileNotFoundError:
        logging.error(f"Keywords file not found: {keywords_file_path}")
        # Return fallback static keywords
        return _get_fallback_keywords()
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing keywords JSON file: {e}")
        return _get_fallback_keywords()
    except Exception as e:
        logging.error(f"Unexpected error loading keywords: {e}")
        return _get_fallback_keywords()

def _get_fallback_keywords():
    """Fallback static keywords in case dynamic loading fails"""
    return {
        'Atelectasis': {
            'high_confidence': ['atelectasis detected', 'lung collapse present', 'volume loss identified'],
            'medium_confidence': ['possible atelectasis', 'suspected collapse', 'potential volume loss'],
            'low_confidence': ['no atelectasis', 'no collapse', 'normal lung expansion']
        },
        'Cardiomegaly': {
            'high_confidence': ['cardiomegaly detected', 'heart enlargement present', 'enlarged cardiac silhouette'],
            'medium_confidence': ['possible cardiomegaly', 'borderline heart size', 'cardiac enlargement suspected'],
            'low_confidence': ['normal heart size', 'no cardiomegaly', 'normal cardiac silhouette']
        },
        'Edema': {
            'high_confidence': ['pulmonary edema detected', 'lung fluid present', 'edema identified'],
            'medium_confidence': ['possible edema', 'suspected fluid', 'potential pulmonary edema'],
            'low_confidence': ['no edema', 'no lung fluid', 'clear lung fields']
        },
        'Lung Opacity': {
            'high_confidence': ['lung opacity detected', 'infiltrate present', 'consolidation identified'],
            'medium_confidence': ['possible opacity', 'suspected infiltrate', 'potential consolidation'],
            'low_confidence': ['no opacity', 'no infiltrates', 'clear lungs']
        },
        'No Finding': {
            'high_confidence': ['normal chest X-ray', 'no acute findings', 'unremarkable study'],
            'medium_confidence': ['essentially normal', 'largely unremarkable', 'no significant abnormalities'],
            'low_confidence': ['abnormalities present', 'findings detected', 'pathology identified']
        },
        'Pleural Effusion': {
            'high_confidence': ['pleural effusion detected', 'pleural fluid present', 'effusion identified'],
            'medium_confidence': ['possible effusion', 'suspected pleural fluid', 'potential effusion'],
            'low_confidence': ['no effusion', 'no pleural fluid', 'clear pleural spaces']
        },
        'Pneumonia': {
            'high_confidence': ['pneumonia detected', 'infection present', 'pneumonia identified'],
            'medium_confidence': ['possible pneumonia', 'suspected infection', 'potential pneumonia'],
            'low_confidence': ['no pneumonia', 'no infection', 'no inflammatory changes']
        },
        'Support Devices': {
            'high_confidence': ['support devices present', 'medical devices detected', 'equipment identified'],
            'medium_confidence': ['possible devices', 'suspected equipment', 'potential support devices'],
            'low_confidence': ['no devices', 'no support equipment', 'no medical devices']
        }
    }

# Load the condition keywords at module level
CONDITION_KEYWORDS = load_condition_keywords()

# Static condition-to-region mapping based on medical knowledge
CONDITION_TO_REGIONS = {
    'Atelectasis': [
        'left_lung', 'right_lung', 
        'left_lower_lung_zone', 'right_lower_lung_zone',
        'left_upper_lung_zone', 'right_upper_lung_zone'
    ],
    'Cardiomegaly': [
        'cardiac_silhouette'
    ],
    'Edema': [
        'left_lung', 'right_lung',
        'left_lower_lung_zone', 'right_lower_lung_zone',
        'left_hilar_structures', 'right_hilar_structures'
    ],
    'Lung Opacity': [
        'left_lung', 'right_lung',
        'left_lower_lung_zone', 'right_lower_lung_zone',
        'left_upper_lung_zone', 'right_upper_lung_zone',
        'left_mid_lung_zone', 'right_mid_lung_zone'
    ],
    'No Finding': [
        # All regions are relevant for "normal" findings
        'cardiac_silhouette', 'left_lung', 'right_lung',
        'left_upper_lung_zone', 'right_upper_lung_zone',
        'left_mid_lung_zone', 'right_mid_lung_zone',
        'left_lower_lung_zone', 'right_lower_lung_zone'
    ],
    'Pleural Effusion': [
        'left_costophrenic_angle', 'right_costophrenic_angle',
        'left_lower_lung_zone', 'right_lower_lung_zone'
    ],
    'Pneumonia': [
        'left_lung', 'right_lung',
        'left_lower_lung_zone', 'right_lower_lung_zone',
        'left_upper_lung_zone', 'right_upper_lung_zone',
        'left_mid_lung_zone', 'right_mid_lung_zone'
    ],
    'Support Devices': [
        'upper_mediastinum', 'cardiac_silhouette', 'trachea',
        'left_hilar_structures', 'right_hilar_structures'
    ]
}

# LLM Integration Configuration - Gemini 2.5 Pro
LLM_CONFIG = {
    'api_type': 'gemini',
    'base_url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent',
    'model_name': 'gemini-2.5-pro',
    'max_tokens': 65535,
    'temperature': 0.3,  # Conservative for medical reports
    'timeout': 120,  # 120 seconds timeout
    'max_retries': 3,
    'retry_delay': 3  # seconds
}

# Medical Report Templates - Optimized for Clinical Accuracy
REPORT_TEMPLATES = {
    'standard': {
        'sections': ['findings', 'impression', 'recommendations'],
        'style': 'professional radiological report',
        'length': 'appropriate clinical detail',
        'confidence_based': True
    },
    'detailed': {
        'sections': ['technique', 'findings', 'impression', 'recommendations'],
        'style': 'comprehensive radiological assessment',
        'length': 'detailed clinical analysis',
        'confidence_based': True
    },
    'concise': {
        'sections': ['findings', 'impression'],
        'style': 'concise clinical summary',
        'length': 'brief and direct',
        'confidence_based': True
    }
}

def check_system_memory():
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    return available_gb < 4.0

def detect_device():
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            return "cuda"
        except Exception:
            return "cpu"
    else:
        return "cpu"

class MultiModalMIMICModel(nn.Module):
    def __init__(self, num_conditions: int, contrastive_temperature: float = 0.1, loss_type="asymmetric"):
        super().__init__()
        self.contrastive_temperature = contrastive_temperature
        self.loss_type = loss_type
        self._keys_to_ignore_on_save = None

        low_memory = check_system_memory()
        
        # Image encoder (ViT)
        self.vit = AutoModel.from_pretrained(VIT_MODEL_NAME)
        img_feature_dim = self.vit.config.hidden_size

        if low_memory:
            hidden_dim = 512
            self.img_proj = nn.Sequential(
                nn.Linear(img_feature_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 512)
            )
        else:
            hidden_dim = 768
            self.img_proj = nn.Sequential(
                nn.Linear(img_feature_dim, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.Dropout(0.1)
            )

        # BBox encoder
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
                nn.Linear(128, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
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
                nn.Linear(512, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.Dropout(0.1)
            )

        # Fixation encoder
        if low_memory:
            self.fix_emb = nn.Linear(4, 64)
            self.fix_gru = nn.GRU(64, 128, num_layers=1, batch_first=True, bidirectional=True, dropout=0.1)
            self.fix_proj = nn.Sequential(
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            self.fix_emb = nn.Linear(4, 128)
            self.fix_gru = nn.GRU(128, 384, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
            self.fix_proj = nn.Sequential(
                nn.Linear(1536, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.Dropout(0.1)
            )

        # Text encoder
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

        # Fusion and classification
        if low_memory:
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.global_classifier = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_conditions)
            )
            self.condition_specific_heads = None
            self.fusion_attention = None
        else:
            self.fusion_attention = nn.MultiheadAttention(768, 8, dropout=0.1, batch_first=True)
            self.fusion_norm = nn.LayerNorm(768)
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
            self.condition_specific_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(768, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 1)
                ) for _ in range(num_conditions)
            ])
            self.global_classifier = nn.Sequential(
                nn.Linear(768, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_conditions)
            )

        self._initialize_weights()
        self.loss_fn = None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, img, bbox, fix_seq, fix_mask, transcript, labels=None, contrastive_weight=0.1):
        # Basic forward pass implementation for inference
        batch_size = img.size(0)
        
        # Image encoding
        img_outputs = self.vit(pixel_values=img)
        if hasattr(img_outputs, 'pooler_output') and img_outputs.pooler_output is not None:
            img_feat = img_outputs.pooler_output
        else:
            img_feat = img_outputs.last_hidden_state[:, 0, :]
        img_proj = self.img_proj(img_feat)
        
        # BBox encoding
        bbox_feat = self.bbox_encoder(bbox).flatten(1)
        bbox_proj = self.bbox_proj(bbox_feat)
        
        # Fixation encoding
        fix_lens = fix_mask.sum(dim=1).cpu()
        if torch.all(fix_lens == 0):
            fix_feat_gru = torch.zeros(batch_size, 256 if check_system_memory() else 1536, device=fix_seq.device)
        else:
            fix_emb = self.fix_emb(fix_seq)
            packed_seq = nn.utils.rnn.pack_padded_sequence(
                fix_emb, fix_lens.clamp(min=1), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.fix_gru(packed_seq)
            fix_feat_gru = h_n.transpose(0, 1).reshape(batch_size, -1)
        fix_proj = self.fix_proj(fix_feat_gru)
        
        # Text encoding
        if isinstance(transcript, list):
            try:
                max_length = 256 if check_system_memory() else 512
                transcript_tokens = self.tokenizer(
                    transcript, return_tensors="pt", padding=True, 
                    truncation=True, max_length=max_length
                ).to(img.device)
                text_outputs = self.text_encoder(**transcript_tokens)
                text_feat = text_outputs.last_hidden_state[:, 0, :]
                text_proj = self.text_proj(text_feat)
            except Exception:
                hidden_dim = 512 if check_system_memory() else 768
                text_proj = torch.zeros(batch_size, hidden_dim, device=img.device)
        else:
            hidden_dim = 512 if check_system_memory() else 768
            text_proj = torch.zeros(batch_size, hidden_dim, device=img.device)
        
        # Fusion
        combined = torch.cat([img_proj, bbox_proj, fix_proj, text_proj], dim=1)
        fused = self.fusion(combined)
        
        if not check_system_memory() and self.fusion_attention is not None:
            stacked_features = torch.stack([img_proj, bbox_proj, fix_proj, text_proj], dim=1)
            attended_features, _ = self.fusion_attention(stacked_features, stacked_features, stacked_features)
            attended_combined = attended_features.mean(dim=1)
            final_fused = self.fusion_norm(fused + attended_combined)
        else:
            final_fused = fused
        
        # Classification
        global_logits = self.global_classifier(final_fused)
        
        if not check_system_memory() and self.condition_specific_heads is not None:
            condition_logits = []
            for head in self.condition_specific_heads:
                cond_logit = head(final_fused)
                condition_logits.append(cond_logit)
            condition_logits = torch.cat(condition_logits, dim=1)
            logits_output = 0.7 * global_logits + 0.3 * condition_logits
        else:
            logits_output = global_logits
        
        return {
            "logits": logits_output,
            "zi": img_proj,
            "zg": fix_proj,
            "zb": bbox_proj,
            "zt": text_proj,
            "attn_map": torch.zeros_like(bbox)
        }

class MedicalReportGenerator:
    def __init__(self, model_path: str = None, log_level: int = logging.INFO):
        # Load environment variables for Gemini API
        load_dotenv('.env')
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        self.device = detect_device()
        self.low_memory = check_system_memory()
        self.model = None
        
        # Fix model path - try multiple possible locations
        if model_path is None:
            possible_paths = [
                os.path.join("output", "0.(full+enhanced_gaze)_training_mimic_on_chexpert_optimized", "model_0.(full+enhanced_gaze)_training_mimic_on_chexpert_optimized", "model.safetensors"),
                os.path.join("output", "training_mimic_on_chexpert_optimized", "model_training_mimic_on_chexpert_optimized", "model.safetensors"),
                os.path.join("results", "final_model", "model.safetensors"),
                "model.safetensors"  # Current directory fallback
            ]
            
            self.model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.model_path = path
                    break
            
            if self.model_path is None:
                self.model_path = possible_paths[0]  # Use first path as default even if not found
        else:
            self.model_path = model_path
        
        # Initialize test data directory (fix missing initialization)
        self.test_data_dir = "test_data"
        
        # Enhanced logging setup
        self._setup_logging(log_level)
        
        # Initialize preprocessing transforms
        self._setup_preprocessing()
        
        # Initialize keyword extraction system (Phase 2)
        self.condition_keywords = CONDITION_KEYWORDS
        self.confidence_thresholds = {
            'high': 0.7,    # Strong indication of condition
            'medium': 0.5,  # Moderate indication  
            'low': 0.3      # Weak indication (below this = no keywords)
        }
        
        # Log keyword loading statistics
        self._log_keyword_statistics()
        
        # Initialize attention visualization system (Phase 3)
        try:
            self.anatomical_regions = ANATOMICAL_REGIONS
        except NameError:
            # Fallback if import failed
            self.logger.warning("ANATOMICAL_REGIONS not available, using empty regions")
            self.anatomical_regions = {}
        self.attention_threshold = 0.6  # Minimum attention intensity to consider
        self.grad_cam_enabled = True
        self.attention_cache = {}  # Cache attention maps for efficiency
        
        # Initialize LLM integration system (Phase 4)
        self.llm_config = LLM_CONFIG
        self.report_templates = REPORT_TEMPLATES
        self.llm_available = False
        self.llm_connection_tested = False
        
        # Initialize with real data paths
        self.chest_xray_dir = CHEST_XRAY_DIR
        
        self.logger.info(f"Initialized MedicalReportGenerator")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Low memory mode: {self.low_memory}")
        self.logger.info(f"Model path: {self.model_path} (exists: {os.path.exists(self.model_path)})")
        self.logger.info(f"Real chest X-ray directory: {self.chest_xray_dir}")
        self.logger.info(f"Test data directory: {self.test_data_dir}")
        self.logger.info(f"Keyword extraction system loaded with {len(self.condition_keywords)} conditions")
        self.logger.info(f"Attention visualization system loaded with {len(self.anatomical_regions)} anatomical regions")
        self.logger.info(f"LLM integration configured for {self.llm_config['model_name']} via Gemini API")

    def _setup_logging(self, log_level: int):
        """Enhanced logging configuration"""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler for detailed logs
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler("logs/medical_report_generator.log")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _log_keyword_statistics(self):
        """Log statistics about the loaded keywords"""
        try:
            total_keywords = 0
            keyword_stats = {}
            
            for condition, keywords_dict in self.condition_keywords.items():
                condition_total = 0
                condition_stats = {}
                
                for confidence_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
                    if confidence_level in keywords_dict:
                        count = len(keywords_dict[confidence_level])
                        condition_stats[confidence_level] = count
                        condition_total += count
                
                keyword_stats[condition] = {
                    'total': condition_total,
                    **condition_stats
                }
                total_keywords += condition_total
            
            self.logger.info(f"=== KEYWORD STATISTICS ===")
            self.logger.info(f"Total keywords loaded: {total_keywords}")
            self.logger.info(f"Conditions with keywords: {len(self.condition_keywords)}")
            
            for condition, stats in keyword_stats.items():
                self.logger.info(f"{condition}: {stats['total']} total keywords "
                               f"(High: {stats.get('high_confidence', 0)}, "
                               f"Medium: {stats.get('medium_confidence', 0)}, "
                               f"Low: {stats.get('low_confidence', 0)})")
            
            self.logger.info("=" * 30)
            
        except Exception as e:
            self.logger.error(f"Error logging keyword statistics: {e}")

    def _setup_preprocessing(self):
        # Match training transforms exactly
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
        
        self.bbox_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def get_available_chest_xrays(self) -> List[str]:
        """
        Get list of available chest X-ray images from the real dataset.
        
        Returns:
            List of available image file paths
        """
        try:
            self.logger.debug(f"Current working directory: {os.getcwd()}")
            self.logger.debug(f"Checking chest X-ray directory: {self.chest_xray_dir}")
            self.logger.debug(f"Absolute path: {os.path.abspath(self.chest_xray_dir)}")
            
            if not os.path.exists(self.chest_xray_dir):
                self.logger.error(f"Chest X-ray directory not found: {self.chest_xray_dir}")
                
                # Try to find alternative paths
                self.logger.info("Searching for data_dump directory...")
                alternative_paths = [
                    os.path.join("..", "data_dump", "output", "img_png"),
                    os.path.join("data_dump", "output", "img_png"),
                    os.path.join(".", "data_dump", "output", "img_png"),
                ]
                
                for alt_path in alternative_paths:
                    abs_alt_path = os.path.abspath(alt_path)
                    self.logger.debug(f"Trying alternative path: {abs_alt_path}")
                    if os.path.exists(abs_alt_path):
                        self.logger.info(f"Found alternative path: {abs_alt_path}")
                        # Update the path and try again
                        self.chest_xray_dir = abs_alt_path
                        return self.get_available_chest_xrays()
                
                return []
            
            # Get all PNG files
            image_files = []
            for filename in os.listdir(self.chest_xray_dir):
                if filename.lower().endswith('.png'):
                    image_path = os.path.join(self.chest_xray_dir, filename)
                    image_files.append(image_path)
            
            self.logger.info(f"Found {len(image_files)} chest X-ray images in {self.chest_xray_dir}")
            if len(image_files) > 0:
                self.logger.debug(f"Sample images: {[os.path.basename(f) for f in image_files[:3]]}")
            
            return sorted(image_files)
            
        except Exception as e:
            self.logger.error(f"Error getting available chest X-rays: {e}")
            self.logger.exception("Full traceback:")
            return []

    def get_random_chest_xray(self) -> str:
        """
        Get a random chest X-ray from the available images.
        
        Returns:
            Path to a random chest X-ray image
        """
        try:
            available_images = self.get_available_chest_xrays()
            if not available_images:
                raise ValueError("No chest X-ray images available")
            
            selected_image = random.choice(available_images)
            self.logger.info(f"Selected random chest X-ray: {os.path.basename(selected_image)}")
            return selected_image
            
        except Exception as e:
            self.logger.error(f"Error getting random chest X-ray: {e}")
            raise

    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        try:
            image_path = Path(image_path)
            
            # Validate file exists
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Validate file extension
            valid_extensions = {'.png', '.jpg', '.jpeg', '.dcm', '.dicom'}
            if image_path.suffix.lower() not in valid_extensions:
                raise ValueError(f"Unsupported image format: {image_path.suffix}")
            
            # Load and preprocess image
            if image_path.suffix.lower() in {'.dcm', '.dicom'}:
                # Handle DICOM files (simplified - may need pydicom for full support)
                try:
                    import pydicom
                    ds = pydicom.dcmread(image_path)
                    img_array = ds.pixel_array
                    # Convert to PIL Image
                    img_array = ((img_array - img_array.min()) / 
                               (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                    img = Image.fromarray(img_array).convert('RGB')
                except ImportError:
                    self.logger.warning("pydicom not available, treating DICOM as regular image")
                    img = Image.open(image_path).convert('RGB')
            else:
                # Handle standard image formats
                img = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            img_tensor = self.img_transform(img)
            
            # Add batch dimension and move to device
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            self.logger.info(f"Preprocessed image: {image_path} -> {img_tensor.shape}")
            return img_tensor
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {e}")
            # Return default tensor on error
            return torch.zeros(1, 3, 224, 224).to(self.device)

    def preprocess_bbox_mask(self, bbox_path: Union[str, Path]) -> torch.Tensor:
        try:
            bbox_path = Path(bbox_path)
            
            if not bbox_path.exists():
                raise FileNotFoundError(f"BBox mask file not found: {bbox_path}")
            
            # Load bbox mask
            bbox = Image.open(bbox_path).convert('L')
            
            # Apply transforms
            bbox_tensor = self.bbox_transform(bbox)
            
            # Add batch dimension and move to device
            bbox_tensor = bbox_tensor.unsqueeze(0).to(self.device)
            
            return bbox_tensor
            
        except Exception as e:
            self.logger.error(f"Error preprocessing bbox mask {bbox_path}: {e}")
            # Return default tensor on error
            return torch.zeros(1, 1, 224, 224).to(self.device)

    def preprocess_fixation_sequence(self, fixation_path: Union[str, Path]) -> tuple:
        try:
            fixation_path = Path(fixation_path)
            
            if not fixation_path.exists():
                raise FileNotFoundError(f"Fixation file not found: {fixation_path}")
            
            # Load fixation data
            arr = np.load(fixation_path)
            seq = arr["seq"].astype(np.float32)  # (L,4)
            mask = arr["mask"].astype(bool)      # (L,)
            
            # Safely handle any remaining NaNs
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Convert to tensors and apply safety clipping
            seq = torch.from_numpy(seq).float()
            seq = seq.clamp(-10.0, 10.0)
            mask = torch.from_numpy(mask)
            
            # Add batch dimension and move to device
            seq = seq.unsqueeze(0).to(self.device)
            mask = mask.unsqueeze(0).to(self.device)
            
            # Close numpy file to free memory
            arr.close() if hasattr(arr, 'close') else None
            
            return seq, mask
            
        except Exception as e:
            self.logger.error(f"Error preprocessing fixation sequence {fixation_path}: {e}")
            # Return default tensors on error
            seq = torch.zeros(1, 128, 4).to(self.device)
            mask = torch.zeros(1, 128, dtype=torch.bool).to(self.device)
            return seq, mask

    def load_classification_model(self) -> bool:
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            
            if not os.path.exists(self.model_path):
                # Provide helpful error message with possible locations
                self.logger.error(f"Model file not found: {self.model_path}")
                self.logger.info("Searching for model in possible locations...")
                
                # Search more broadly for model files
                search_patterns = [
                    "new_scripts/output/**/model.safetensors",
                    "output/**/model.safetensors", 
                    "results/**/model.safetensors",
                    "**/model.safetensors"
                ]
                
                import glob
                found_models = []
                for pattern in search_patterns:
                    found_models.extend(glob.glob(pattern, recursive=True))
                
                if found_models:
                    self.logger.info("Found these model files:")
                    for model_file in found_models:
                        self.logger.info(f"  - {model_file}")
                    self.logger.info("You can specify the correct path when initializing MedicalReportGenerator")
                else:
                    self.logger.info("No model.safetensors files found in the project directory")
                    self.logger.info("Please ensure you have trained a model first by running the training script")
                
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Initialize model
            self.model = MultiModalMIMICModel(len(CONDITIONS))
            
            # Load weights
            state_dict = safetensors.torch.load_file(self.model_path, device="cpu")
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                self.logger.warning(f"Missing keys in model state_dict: {len(missing_keys)} keys")
            if unexpected_keys:
                self.logger.warning(f"Unexpected keys in model state_dict: {len(unexpected_keys)} keys")
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Get model size info
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Model loaded successfully: {total_params:,} parameters")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def run_model_inference(self, image_path: str) -> Dict:
        """
        Run model inference on a chest X-ray image.
        
        Args:
            image_path: Path to the chest X-ray image
            
        Returns:
            Dictionary with model predictions and results
        """
        try:
            self.logger.info(f"Running model inference on: {os.path.basename(image_path)}")
            
            # Ensure model is loaded
            if self.model is None:
                if not self.load_classification_model():
                    raise RuntimeError("Failed to load model")
            
            # Preprocess image
            img_tensor = self.preprocess_image(image_path)
            
            # Create dummy multimodal inputs (simplified for real usage)
            dummy_bbox = torch.zeros(1, 1, 224, 224).to(self.device)
            dummy_fix_seq = torch.randn(1, 20, 4).to(self.device)
            dummy_fix_mask = torch.ones(1, 20, dtype=torch.bool).to(self.device)
            dummy_transcript = ["Medical chest X-ray analysis"]
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(
                    img=img_tensor,
                    bbox=dummy_bbox,
                    fix_seq=dummy_fix_seq,
                    fix_mask=dummy_fix_mask,
                    transcript=dummy_transcript
                )
            
            # Process predictions
            logits = outputs["logits"]
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            
            # Create results
            results = {
                "success": True,
                "image_path": str(image_path),
                "condition_predictions": {}
            }
            
            # Map predictions to condition names
            for i, condition in enumerate(CONDITIONS):
                results["condition_predictions"][condition] = {
                    "probability": float(probabilities[0, i].item()),
                    "predicted": bool(predictions[0, i].item()),
                    "logit": float(logits[0, i].item())
                }
            
            self.logger.info(f"Model inference completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Model inference failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_path": str(image_path)
            }

    def test_model_inference(self) -> bool:
        if self.model is None:
            self.logger.error("Model not loaded. Call load_classification_model() first.")
            return False
        
        try:
            # Create dummy inputs
            batch_size = 1
            dummy_img = torch.randn(batch_size, 3, 224, 224).to(self.device)
            dummy_bbox = torch.randn(batch_size, 1, 224, 224).to(self.device)
            dummy_fix_seq = torch.randn(batch_size, 10, 4).to(self.device)
            dummy_fix_mask = torch.ones(batch_size, 10, dtype=torch.bool).to(self.device)
            dummy_transcript = ["test transcript"]
            
            # Test inference
            with torch.no_grad():
                outputs = self.model(
                    img=dummy_img,
                    bbox=dummy_bbox,
                    fix_seq=dummy_fix_seq,
                    fix_mask=dummy_fix_mask,
                    transcript=dummy_transcript
                )
            
            # Verify output shape
            logits = outputs["logits"]
            if logits.shape == (batch_size, len(CONDITIONS)):
                self.logger.info("Model inference test passed")
                return True
            else:
                self.logger.error(f"Unexpected output shape: {logits.shape}")
                return False
                
        except Exception as e:
            self.logger.error(f"Model inference test failed: {e}")
            return False

    def test_image_preprocessing(self, test_image_path: str = None) -> bool:
        try:
            # Create a dummy test image if none provided
            if test_image_path is None:
                # Create a simple test image
                test_img = Image.new('RGB', (512, 512), color='gray')
                test_image_path = "temp_test_image.png"
                test_img.save(test_image_path)
                cleanup_test_image = True
            else:
                cleanup_test_image = False
            
            # Test image preprocessing
            img_tensor = self.preprocess_image(test_image_path)
            
            # Verify tensor properties
            if (img_tensor.shape == (1, 3, 224, 224) and 
                img_tensor.device.type == self.device and
                img_tensor.dtype == torch.float32):
                self.logger.info("Image preprocessing test passed")
                success = True
            else:
                self.logger.error(f"Image preprocessing failed: shape={img_tensor.shape}, device={img_tensor.device}, dtype={img_tensor.dtype}")
                success = False
            
            # Cleanup test image if we created it
            if cleanup_test_image and os.path.exists(test_image_path):
                os.remove(test_image_path)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Image preprocessing test failed: {e}")
            return False

    def create_test_image_collection(self) -> bool:
        """Create a collection of test images for development and testing"""
        try:
            os.makedirs(self.test_data_dir, exist_ok=True)
            
            # Create various test images
            test_images = [
                ("normal_chest.png", "RGB", (512, 512), "lightgray"),
                ("pneumonia_sim.png", "RGB", (512, 512), "darkgray"),
                ("small_image.png", "RGB", (128, 128), "blue"),
                ("large_image.png", "RGB", (1024, 1024), "green"),
            ]
            
            created_images = []
            for filename, mode, size, color in test_images:
                filepath = os.path.join(self.test_data_dir, filename)
                if not os.path.exists(filepath):
                    img = Image.new(mode, size, color)
                    img.save(filepath)
                    created_images.append(filename)
                    self.logger.debug(f"Created test image: {filename}")
            
            # Create test metadata file
            metadata = {
                "test_images": [
                    {"filename": "normal_chest.png", "description": "Simulated normal chest X-ray", "expected_conditions": []},
                    {"filename": "pneumonia_sim.png", "description": "Simulated pneumonia case", "expected_conditions": ["Pneumonia"]},
                    {"filename": "small_image.png", "description": "Small resolution test image", "expected_conditions": []},
                    {"filename": "large_image.png", "description": "Large resolution test image", "expected_conditions": []},
                ],
                "created_at": str(np.datetime64('now')),
                "purpose": "Development and testing of medical report generator"
            }
            
            metadata_path = os.path.join(self.test_data_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Test image collection ready: {len(test_images)} images in {self.test_data_dir}")
            if created_images:
                self.logger.info(f"Created new images: {created_images}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating test image collection: {e}")
            return False

    def debug_model_outputs(self, outputs: Dict, input_info: Dict = None) -> Dict:
        """Debug utility to analyze model outputs"""
        debug_info = {
            "output_keys": list(outputs.keys()),
            "shapes": {},
            "dtypes": {},
            "devices": {},
            "statistics": {}
        }
        
        # Analyze each output tensor
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                debug_info["shapes"][key] = list(value.shape)
                debug_info["dtypes"][key] = str(value.dtype)
                debug_info["devices"][key] = str(value.device)
                
                # Calculate statistics
                if value.numel() > 0:
                    debug_info["statistics"][key] = {
                        "mean": float(value.mean().item()),
                        "std": float(value.std().item()),
                        "min": float(value.min().item()),
                        "max": float(value.max().item()),
                        "zero_fraction": float((value == 0).float().mean().item())
                    }
        
        # Log debug information
        self.logger.debug("=== Model Output Debug Info ===")
        for key in debug_info["output_keys"]:
            if key in debug_info["shapes"]:
                self.logger.debug(f"{key}: shape={debug_info['shapes'][key]}, "
                                f"dtype={debug_info['dtypes'][key]}, "
                                f"device={debug_info['devices'][key]}")
                if key in debug_info["statistics"]:
                    stats = debug_info["statistics"][key]
                    self.logger.debug(f"  Stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                                    f"range=[{stats['min']:.4f}, {stats['max']:.4f}], "
                                    f"zero%={stats['zero_fraction']*100:.1f}%")
        
        # Add input information if provided
        if input_info:
            debug_info["input_info"] = input_info
            self.logger.debug(f"Input info: {input_info}")
        
        return debug_info

    def test_end_to_end_inference(self, image_path: str = None) -> Dict:
        """Comprehensive end-to-end inference test"""
        try:
            self.logger.info("Starting end-to-end inference test...")
            
            # Use test image if none provided
            if image_path is None:
                if not os.path.exists(self.test_data_dir):
                    self.create_test_image_collection()
                image_path = os.path.join(self.test_data_dir, "normal_chest.png")
            
            # Ensure model is loaded
            if self.model is None:
                self.logger.info("Loading model for inference test...")
                if not self.load_classification_model():
                    raise RuntimeError("Failed to load model")
            
            # Step 1: Preprocess image
            self.logger.info("Step 1: Preprocessing image...")
            img_tensor = self.preprocess_image(image_path)
            
            # Step 2: Create dummy multimodal inputs
            self.logger.info("Step 2: Preparing multimodal inputs...")
            dummy_bbox = torch.zeros(1, 1, 224, 224).to(self.device)
            dummy_fix_seq = torch.randn(1, 20, 4).to(self.device)  # Random fixation sequence
            dummy_fix_mask = torch.ones(1, 20, dtype=torch.bool).to(self.device)
            dummy_transcript = ["Patient presents with chest discomfort and cough."]
            
            input_info = {
                "image_path": str(image_path),
                "image_shape": list(img_tensor.shape),
                "bbox_shape": list(dummy_bbox.shape),
                "fix_seq_shape": list(dummy_fix_seq.shape),
                "transcript_length": len(dummy_transcript[0])
            }
            
            # Step 3: Model inference
            self.logger.info("Step 3: Running model inference...")
            with torch.no_grad():
                outputs = self.model(
                    img=img_tensor,
                    bbox=dummy_bbox,
                    fix_seq=dummy_fix_seq,
                    fix_mask=dummy_fix_mask,
                    transcript=dummy_transcript
                )
            
            # Step 4: Process predictions
            self.logger.info("Step 4: Processing predictions...")
            logits = outputs["logits"]
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            
            # Step 5: Create results
            results = {
                "success": True,
                "input_info": input_info,
                "raw_logits": logits.cpu().numpy().tolist(),
                "probabilities": probabilities.cpu().numpy().tolist(),
                "binary_predictions": predictions.cpu().numpy().tolist(),
                "condition_predictions": {}
            }
            
            # Map predictions to condition names
            for i, condition in enumerate(CONDITIONS):
                results["condition_predictions"][condition] = {
                    "probability": float(probabilities[0, i].item()),
                    "predicted": bool(predictions[0, i].item()),
                    "logit": float(logits[0, i].item())
                }
            
            # Step 6: Debug analysis
            debug_info = self.debug_model_outputs(outputs, input_info)
            results["debug_info"] = debug_info
            
            # Step 7: Validation checks
            validation_results = self._validate_inference_results(results)
            results["validation"] = validation_results
            
            if validation_results["all_passed"]:
                self.logger.info("End-to-end inference test PASSED")
            else:
                self.logger.warning(f"⚠️ End-to-end inference test had issues: {validation_results['failed_checks']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"End-to-end inference test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "input_info": input_info if 'input_info' in locals() else None
            }

    def _validate_inference_results(self, results: Dict) -> Dict:
        """Validate inference results for correctness"""
        validation = {
            "checks": {},
            "failed_checks": [],
            "all_passed": True
        }
        
        try:
            # Check 1: Probabilities in valid range
            probs = np.array(results["probabilities"][0])
            prob_valid = np.all((probs >= 0) & (probs <= 1))
            validation["checks"]["probabilities_valid_range"] = prob_valid
            if not prob_valid:
                validation["failed_checks"].append("probabilities_valid_range")
                validation["all_passed"] = False
            
            # Check 2: Correct number of conditions
            num_conditions = len(results["condition_predictions"])
            conditions_correct = num_conditions == len(CONDITIONS)
            validation["checks"]["correct_num_conditions"] = conditions_correct
            if not conditions_correct:
                validation["failed_checks"].append("correct_num_conditions")
                validation["all_passed"] = False
            
            # Check 3: All condition names present
            expected_conditions = set(CONDITIONS)
            actual_conditions = set(results["condition_predictions"].keys())
            conditions_match = expected_conditions == actual_conditions
            validation["checks"]["condition_names_match"] = conditions_match
            if not conditions_match:
                validation["failed_checks"].append("condition_names_match")
                validation["all_passed"] = False
            
            # Check 4: Reasonable probability distribution (not all zeros or ones)
            prob_diversity = 0.01 < probs.std() < 0.5
            validation["checks"]["probability_diversity"] = prob_diversity
            if not prob_diversity:
                validation["failed_checks"].append("probability_diversity")
                validation["all_passed"] = False
            
        except Exception as e:
            validation["validation_error"] = str(e)
            validation["all_passed"] = False
        
        return validation

    # ========================
    # Phase 2: Keyword Extraction System
    # ========================
    
    def extract_keywords_from_predictions(self, condition_predictions: Dict) -> Dict:
        """
        Extract keywords from model predictions using simplified confidence-based approach.
        Only claims what the model was actually trained to detect.
        
        Args:
            condition_predictions: Dictionary with condition names as keys and 
                                 prediction dictionaries as values containing
                                 'probability' and 'predicted' fields
        
        Returns:
            Dictionary with condition names as keys and lists of extracted keywords
        """
        try:
            extracted_keywords = {}
            
            self.logger.debug("Extracting keywords from predictions...")
            
            for condition, pred_info in condition_predictions.items():
                if condition not in self.condition_keywords:
                    self.logger.warning(f"No keyword mapping found for condition: {condition}")
                    continue
                
                probability = pred_info.get('probability', 0.0)
                condition_kw = self.condition_keywords[condition]
                
                # Determine confidence level and extract appropriate keywords
                if probability >= self.confidence_thresholds['high']:
                    # High confidence: positive finding
                    keywords = condition_kw['high_confidence']
                    confidence_level = 'high'
                elif probability >= self.confidence_thresholds['medium']:
                    # Medium confidence: possible finding
                    keywords = condition_kw['medium_confidence']
                    confidence_level = 'medium'
                elif probability >= self.confidence_thresholds['low']:
                    # Low confidence: likely negative, but include for completeness
                    keywords = condition_kw['low_confidence']
                    confidence_level = 'low'
                else:
                    # Below low threshold: skip entirely
                    continue
                
                if keywords:
                    extracted_keywords[condition] = list(keywords)  # Make a copy
                    self.logger.debug(f"{condition} (p={probability:.3f}, {confidence_level}): {len(keywords)} keywords extracted")
            
            # Special handling for "No Finding" - only if it's the dominant prediction
            if self._should_include_no_finding(condition_predictions):
                if 'No Finding' in condition_predictions:
                    no_finding_prob = condition_predictions['No Finding'].get('probability', 0.0)
                    no_finding_kw = self.condition_keywords['No Finding']
                    
                    if no_finding_prob >= self.confidence_thresholds['high']:
                        extracted_keywords['No Finding'] = list(no_finding_kw['high_confidence'])
                    elif no_finding_prob >= self.confidence_thresholds['medium']:
                        extracted_keywords['No Finding'] = list(no_finding_kw['medium_confidence'])
                    else:
                        extracted_keywords['No Finding'] = list(no_finding_kw['low_confidence'])
            
            self.logger.info(f"Keyword extraction completed: {len(extracted_keywords)} conditions with keywords")
            return extracted_keywords
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords from predictions: {e}")
            return {}
    
    def _determine_severity_level(self, probability: float) -> str:
        """Determine severity level based on prediction probability"""
        if probability >= 0.8:
            return 'severe'
        elif probability >= 0.6:
            return 'moderate'
        else:
            return 'mild'
    
    def _should_include_no_finding(self, condition_predictions: Dict) -> bool:
        """
        Determine if 'No Finding' should be included based on overall predictions.
        Include 'No Finding' only if it has the highest probability and 
        no other significant pathology is detected.
        """
        try:
            no_finding_prob = condition_predictions.get('No Finding', {}).get('probability', 0.0)
            
            # Check if No Finding has highest probability
            max_prob = max(pred.get('probability', 0.0) for pred in condition_predictions.values())
            
            if no_finding_prob != max_prob:
                return False
            
            # Check if any pathology has high confidence
            pathology_conditions = [c for c in condition_predictions.keys() if c != 'No Finding']
            high_conf_pathology = any(
                condition_predictions[c].get('probability', 0.0) >= self.confidence_thresholds['high']
                for c in pathology_conditions
            )
            
            return not high_conf_pathology
            
        except Exception as e:
            self.logger.error(f"Error determining No Finding inclusion: {e}")
            return False
    
    def get_keyword_summary(self, extracted_keywords: Dict) -> Dict:
        """
        Create a summary of extracted keywords organized by confidence level.
        
        Args:
            extracted_keywords: Output from extract_keywords_from_predictions
            
        Returns:
            Dictionary with categorized keyword summary
        """
        try:
            summary = {
                'primary_findings': [],
                'anatomical_locations': [],  # Empty since we removed anatomical keywords
                'severity_indicators': [],   # Empty since we removed severity keywords  
                'descriptive_terms': [],     # Keywords organized by confidence instead
                'total_keywords': 0,
                'conditions_detected': list(extracted_keywords.keys())
            }
            
            # New structure: organize by confidence level rather than keyword type
            high_confidence_findings = []
            medium_confidence_findings = []
            low_confidence_findings = []
            
            for condition, keywords in extracted_keywords.items():
                if condition not in self.condition_keywords:
                    continue
                
                condition_kw = self.condition_keywords[condition]
                
                # Categorize keywords by confidence level
                for keyword in keywords:
                    if keyword in condition_kw['high_confidence']:
                        high_confidence_findings.append(keyword)
                        summary['primary_findings'].append(keyword)  # High confidence = primary
                    elif keyword in condition_kw['medium_confidence']:
                        medium_confidence_findings.append(keyword)
                    elif keyword in condition_kw['low_confidence']:
                        low_confidence_findings.append(keyword)
                
                summary['total_keywords'] += len(keywords)
            
            # Store findings by confidence level
            summary['high_confidence_findings'] = list(set(high_confidence_findings))
            summary['medium_confidence_findings'] = list(set(medium_confidence_findings))
            summary['low_confidence_findings'] = list(set(low_confidence_findings))
            
            # Remove duplicates from existing categories
            summary['primary_findings'] = list(set(summary['primary_findings']))
            
            self.logger.debug(f"Keyword summary: {summary['total_keywords']} total keywords across {len(summary['conditions_detected'])} conditions")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating keyword summary: {e}")
            return {
                'primary_findings': [],
                'anatomical_locations': [],
                'severity_indicators': [],
                'descriptive_terms': [],
                'total_keywords': 0,
                'conditions_detected': [],
                'high_confidence_findings': [],
                'medium_confidence_findings': [],
                'low_confidence_findings': []
            }
    
    def test_keyword_extraction_system(self) -> bool:
        """Test the keyword extraction system with various scenarios"""
        try:
            self.logger.info("Testing keyword extraction system...")
            
            # Test Case 1: High confidence pneumonia
            test_predictions_1 = {
                'Pneumonia': {'probability': 0.85, 'predicted': True},
                'No Finding': {'probability': 0.15, 'predicted': False}
            }
            keywords_1 = self.extract_keywords_from_predictions(test_predictions_1)
            
            if 'Pneumonia' not in keywords_1 or len(keywords_1['Pneumonia']) == 0:
                self.logger.error("Test 1 failed: No keywords for high confidence pneumonia")
                return False
            
            # Test Case 2: Multiple conditions
            test_predictions_2 = {
                'Cardiomegaly': {'probability': 0.6, 'predicted': True},
                'Pleural Effusion': {'probability': 0.4, 'predicted': False},
                'No Finding': {'probability': 0.1, 'predicted': False}
            }
            keywords_2 = self.extract_keywords_from_predictions(test_predictions_2)
            
            if len(keywords_2) != 2:  # Should have keywords for both conditions above threshold
                self.logger.error(f"Test 2 failed: Expected 2 conditions with keywords, got {len(keywords_2)}")
                return False
            
            # Test Case 3: Normal case (No Finding dominant)
            test_predictions_3 = {
                'No Finding': {'probability': 0.9, 'predicted': True},
                'Pneumonia': {'probability': 0.1, 'predicted': False}
            }
            keywords_3 = self.extract_keywords_from_predictions(test_predictions_3)
            
            if 'No Finding' not in keywords_3:
                self.logger.error("Test 3 failed: No Finding should be included for normal case")
                return False
            
            # Test keyword summary
            summary = self.get_keyword_summary(keywords_1)
            if not isinstance(summary, dict) or 'total_keywords' not in summary:
                self.logger.error("Keyword summary test failed")
                return False
            
            self.logger.info("Keyword extraction system tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Keyword extraction system test failed: {e}")
            return False

    # ========================
    # Phase 3: Attention Visualization System
    # ========================
    
    def generate_grad_cam_attention(self, image_tensor: torch.Tensor, 
                                   target_condition: str = None) -> torch.Tensor:
        """
        Generate Grad-CAM attention map for Vision Transformer.
        
        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224)
            target_condition: Specific condition to generate attention for (optional)
        
        Returns:
            Attention heatmap tensor (224, 224)
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_classification_model() first.")
            
            # Set model to training mode for gradients but temporarily
            original_training_mode = self.model.training
            self.model.train()  # Need training mode for RNN gradients
            
            image_tensor = image_tensor.to(self.device)
            image_tensor.requires_grad_(True)
            
            # Create dummy multimodal inputs
            dummy_bbox = torch.zeros(1, 1, 224, 224).to(self.device)
            dummy_fix_seq = torch.randn(1, 10, 4).to(self.device)
            dummy_fix_mask = torch.ones(1, 10, dtype=torch.bool).to(self.device)
            dummy_transcript = [""]
            
            # Forward pass
            outputs = self.model(
                img=image_tensor,
                bbox=dummy_bbox,
                fix_seq=dummy_fix_seq,
                fix_mask=dummy_fix_mask,
                transcript=dummy_transcript
            )
            
            logits = outputs["logits"]
            
            # Determine target class for gradient
            if target_condition and target_condition in CONDITIONS:
                target_idx = CONDITIONS.index(target_condition)
                target_score = logits[0, target_idx]
            else:
                # Use highest probability class
                target_score = logits.max()
            
            # Backward pass
            self.model.zero_grad()
            target_score.backward(retain_graph=True)
            
            # Get gradients from the image
            gradients = image_tensor.grad
            
            # Create attention map from gradients
            # Average across color channels and take absolute value
            attention_map = torch.mean(torch.abs(gradients.squeeze(0)), dim=0)
            
            # Normalize to [0, 1]
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
            
            # Restore original training mode
            self.model.train(original_training_mode)
            
            self.logger.debug(f"Generated Grad-CAM attention map: {attention_map.shape}")
            return attention_map.detach().cpu()
            
        except Exception as e:
            self.logger.error(f"Error generating Grad-CAM attention: {e}")
            # Restore original training mode
            if 'original_training_mode' in locals():
                self.model.train(original_training_mode)
            # Return zero attention map on error
            return torch.zeros(224, 224)
    
    def generate_enhanced_attention(self, image_tensor: torch.Tensor, 
                                   target_condition: str = None) -> torch.Tensor:
        """
        Generate enhanced attention map using multiple techniques.
        
        Args:
            image_tensor: Preprocessed image tensor
            target_condition: Specific condition for attention generation
            
        Returns:
            Enhanced attention heatmap
        """
        try:
            # Generate basic Grad-CAM
            grad_cam_attention = self.generate_grad_cam_attention(image_tensor, target_condition)
            
            # Apply Gaussian smoothing for better visualization
            attention_np = grad_cam_attention.detach().cpu().numpy()
            smoothed_attention = gaussian_filter(attention_np, sigma=1.5)
            
            # Enhance contrast
            smoothed_attention = np.power(smoothed_attention, 0.7)
            
            # Normalize again
            smoothed_attention = (smoothed_attention - smoothed_attention.min()) / (
                smoothed_attention.max() - smoothed_attention.min() + 1e-8)
            
            return torch.from_numpy(smoothed_attention)
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced attention: {e}")
            return torch.zeros(224, 224)
    
    def get_relevant_anatomical_regions(self, condition_predictions: Dict) -> Dict:
        """
        Get relevant anatomical regions based on predicted conditions using static mapping.
        
        Args:
            condition_predictions: Dictionary with condition names and prediction info
            
        Returns:
            Dictionary with relevant regions and their info
        """
        try:
            relevant_regions = {}
            
            # Get conditions with significant predictions
            for condition, pred_info in condition_predictions.items():
                probability = pred_info.get('probability', 0.0)
                
                # Include regions for conditions with reasonable probability
                if probability > 0.3 and condition in CONDITION_TO_REGIONS:
                    condition_regions = CONDITION_TO_REGIONS[condition]
                    
                    for region_key in condition_regions:
                        if region_key in ANATOMICAL_REGIONS:
                            region_info = ANATOMICAL_REGIONS[region_key]
                            
                            if region_key not in relevant_regions:
                                relevant_regions[region_key] = {
                                    'name': region_info['name'],
                                    'keywords': region_info['keywords'][:3],  # Limit to top 3 keywords
                                    'bounds': region_info['bounds'],
                                    'related_conditions': []
                                }
                            
                            # Track which conditions are related to this region
                            relevant_regions[region_key]['related_conditions'].append({
                                'condition': condition,
                                'probability': probability
                            })
            
            self.logger.debug(f"Identified {len(relevant_regions)} relevant anatomical regions")
            return relevant_regions
            
        except Exception as e:
            self.logger.error(f"Error getting relevant anatomical regions: {e}")
            return {}

    def analyze_attention_regions(self, attention_map: torch.Tensor, 
                                 threshold: float = None) -> Dict:
        """
        Analyze attention map to identify key anatomical regions.
        
        Args:
            attention_map: Attention heatmap (224, 224)
            threshold: Minimum attention intensity to consider
            
        Returns:
            Dictionary with attention analysis results
        """
        try:
            if threshold is None:
                threshold = self.attention_threshold
            
            attention_np = attention_map.detach().cpu().numpy()
            height, width = attention_np.shape
            
            # Analyze each anatomical region
            region_analysis = {}
            
            for region_id, region_info in self.anatomical_regions.items():
                x_min, y_min, x_max, y_max = region_info['bounds']
                
                # Convert to pixel coordinates
                pixel_x_min = int(x_min * width)
                pixel_y_min = int(y_min * height)
                pixel_x_max = int(x_max * width)
                pixel_y_max = int(y_max * height)
                
                # Extract region from attention map
                region_attention = attention_np[pixel_y_min:pixel_y_max, pixel_x_min:pixel_x_max]
                
                if region_attention.size > 0:
                    # Calculate statistics for this region
                    max_attention = float(region_attention.max())
                    mean_attention = float(region_attention.mean())
                    attention_area = float((region_attention > threshold).mean())
                    
                    region_analysis[region_id] = {
                        'name': region_info['name'],
                        'max_attention': max_attention,
                        'mean_attention': mean_attention,
                        'attention_coverage': attention_area,
                        'is_significant': max_attention > threshold,
                        'keywords': region_info['keywords']
                    }
            
            # Find top regions by attention
            sorted_regions = sorted(
                [(rid, data) for rid, data in region_analysis.items()],
                key=lambda x: x[1]['max_attention'],
                reverse=True
            )
            
            analysis_result = {
                'regions': region_analysis,
                'top_regions': [rid for rid, _ in sorted_regions[:3]],
                'significant_regions': [rid for rid, data in region_analysis.items() 
                                      if data['is_significant']],
                'overall_attention_intensity': float(attention_np.mean()),
                'max_attention_location': {
                    'y': int(np.unravel_index(attention_np.argmax(), attention_np.shape)[0]),
                    'x': int(np.unravel_index(attention_np.argmax(), attention_np.shape)[1])
                }
            }
            
            self.logger.debug(f"Attention analysis: {len(analysis_result['significant_regions'])} significant regions")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing attention regions: {e}")
            return {
                'regions': {},
                'top_regions': [],
                'significant_regions': [],
                'overall_attention_intensity': 0.0,
                'max_attention_location': {'y': 0, 'x': 0}
            }
    
    def extract_spatial_keywords(self, attention_analysis: Dict) -> List[str]:
        """
        Extract spatial keywords based on attention analysis.
        
        Args:
            attention_analysis: Output from analyze_attention_regions
            
        Returns:
            List of spatial keywords describing attention focus
        """
        try:
            spatial_keywords = []
            
            # Add keywords from significant regions
            for region_id in attention_analysis.get('significant_regions', []):
                region_data = attention_analysis['regions'].get(region_id, {})
                
                # Add region-specific keywords based on attention intensity
                max_attention = region_data.get('max_attention', 0)
                region_keywords = region_data.get('keywords', [])
                
                if max_attention > 0.8:
                    # High attention - add all keywords
                    spatial_keywords.extend(region_keywords)
                    spatial_keywords.append(f"focal attention in {region_data.get('name', 'unknown region')}")
                elif max_attention > 0.6:
                    # Moderate attention - add primary keywords
                    spatial_keywords.extend(region_keywords[:2])
                    spatial_keywords.append(f"increased attention in {region_data.get('name', 'unknown region')}")
                else:
                    # Lower attention - add one keyword
                    if region_keywords:
                        spatial_keywords.append(region_keywords[0])
            
            # Add descriptive terms based on attention pattern
            overall_intensity = attention_analysis.get('overall_attention_intensity', 0)
            num_significant = len(attention_analysis.get('significant_regions', []))
            
            if num_significant > 3:
                spatial_keywords.append("diffuse attention pattern")
            elif num_significant == 1:
                spatial_keywords.append("focal attention pattern")
            elif num_significant > 1:
                spatial_keywords.append("multifocal attention pattern")
            
            if overall_intensity > 0.7:
                spatial_keywords.append("high model confidence")
            elif overall_intensity > 0.4:
                spatial_keywords.append("moderate model confidence")
            
            # Remove duplicates while preserving order
            unique_keywords = []
            seen = set()
            for keyword in spatial_keywords:
                if keyword not in seen:
                    unique_keywords.append(keyword)
                    seen.add(keyword)
            
            self.logger.debug(f"Extracted {len(unique_keywords)} spatial keywords")
            return unique_keywords
            
        except Exception as e:
            self.logger.error(f"Error extracting spatial keywords: {e}")
            return []
    
    def visualize_attention_map(self, image_tensor: torch.Tensor, 
                               attention_map: torch.Tensor,
                               save_path: str = None) -> np.ndarray:
        """
        Create a visualization of the attention map overlaid on the original image.
        
        Args:
            image_tensor: Original image tensor (1, 3, 224, 224)
            attention_map: Attention heatmap (224, 224)
            save_path: Optional path to save the visualization
            
        Returns:
            Visualization as numpy array
        """
        try:
            # Convert image tensor to numpy (denormalize from [-1, 1] to [0, 1])
            img_np = image_tensor.detach().squeeze(0).cpu().numpy()
            img_np = (img_np + 1.0) / 2.0  # Denormalize from [-1, 1] to [0, 1]
            img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
            
            # Convert to grayscale for better overlay
            img_gray = np.mean(img_np, axis=2)
            
            # Prepare attention map
            attention_np = attention_map.detach().cpu().numpy()
            
            # Create colormap for attention
            colormap = cm.get_cmap('jet')
            attention_colored = colormap(attention_np)[:, :, :3]  # Remove alpha channel
            
            # Blend image and attention
            alpha = 0.4  # Transparency of attention overlay
            blended = img_gray[:, :, np.newaxis] * (1 - alpha) + attention_colored * alpha
            blended = np.clip(blended, 0, 1)
            
            # Add region boundaries for reference
            vis_with_regions = self._add_region_boundaries(blended)
            
            # Save if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.figure(figsize=(10, 10))
                plt.imshow(vis_with_regions)
                plt.title("Attention Map Visualization")
                plt.axis('off')
                plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), shrink=0.8)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                self.logger.debug(f"Attention visualization saved to {save_path}")
            
            return (vis_with_regions * 255).astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error creating attention visualization: {e}")
            # Return blank image on error
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def _add_region_boundaries(self, image: np.ndarray) -> np.ndarray:
        """Add anatomical region boundaries to visualization"""
        try:
            height, width = image.shape[:2]
            vis_image = image.copy()
            
            # Draw region boundaries
            for region_info in self.anatomical_regions.values():
                x_min, y_min, x_max, y_max = region_info['bounds']
                
                # Convert to pixel coordinates
                pixel_x_min = int(x_min * width)
                pixel_y_min = int(y_min * height)
                pixel_x_max = int(x_max * width)
                pixel_y_max = int(y_max * height)
                
                # Draw thin boundary lines
                vis_image[pixel_y_min:pixel_y_min+1, pixel_x_min:pixel_x_max] = [0.8, 0.8, 0.8]  # Top
                vis_image[pixel_y_max-1:pixel_y_max, pixel_x_min:pixel_x_max] = [0.8, 0.8, 0.8]  # Bottom
                vis_image[pixel_y_min:pixel_y_max, pixel_x_min:pixel_x_min+1] = [0.8, 0.8, 0.8]  # Left
                vis_image[pixel_y_min:pixel_y_max, pixel_x_max-1:pixel_x_max] = [0.8, 0.8, 0.8]  # Right
            
            return vis_image
            
        except Exception as e:
            self.logger.error(f"Error adding region boundaries: {e}")
            return image
    
    def test_attention_visualization_system(self) -> bool:
        """Test the attention visualization system with various scenarios"""
        try:
            self.logger.info("Testing attention visualization system...")
            
            if self.model is None:
                self.logger.error("Model not loaded for attention testing")
                return False
            
            # Test 1: Generate attention map
            test_image_path = "test_data/pneumonia_sim.png"
            img_tensor = self.preprocess_image(test_image_path)
            
            attention_map = self.generate_enhanced_attention(img_tensor, target_condition="Pneumonia")
            if attention_map.shape != (224, 224):
                self.logger.error("Test 1 failed: Incorrect attention map shape")
                return False
            
            # Test 2: Analyze attention regions
            attention_analysis = self.analyze_attention_regions(attention_map)
            if not isinstance(attention_analysis, dict) or 'regions' not in attention_analysis:
                self.logger.error("Test 2 failed: Attention analysis format incorrect")
                return False
            
            # Test 3: Extract spatial keywords
            spatial_keywords = self.extract_spatial_keywords(attention_analysis)
            if not isinstance(spatial_keywords, list):
                self.logger.error("Test 3 failed: Spatial keywords not a list")
                return False
            
            # Test 4: Create visualization
            visualization = self.visualize_attention_map(img_tensor, attention_map)
            if visualization.shape != (224, 224, 3):
                self.logger.error("Test 4 failed: Visualization shape incorrect")
                return False
            
            self.logger.info("Attention visualization system tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Attention visualization system test failed: {e}")
            return False

    # ========================
    # Phase 4: LLM Integration System
    # ========================
    
    def test_llm_connection(self) -> bool:
        """
        Test connection to Gemini 2.5 Pro API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info("Testing Gemini 2.5 Pro connection...")
            
            if not self.gemini_api_key:
                self.logger.error("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
                return False
            
            # Simple test request for Gemini
            test_payload = {
                "contents": [{
                    "parts": [{
                        "text": "Hello! Please respond with just 'Working' to confirm connection."
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "topK": 1,
                    "topP": 0.8,
                    "maxOutputTokens": 50
                }
            }
            
            url = f"{self.llm_config['base_url']}?key={self.gemini_api_key}"
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(
                url,
                headers=headers,
                json=test_payload,
                timeout=self.llm_config['timeout']
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    # Check if we have content (even if parts is missing due to MAX_TOKENS)
                    if 'content' in candidate:
                        # If we got a response (even truncated), connection is working
                        self.llm_available = True
                        self.llm_connection_tested = True
                        self.logger.info("Gemini 2.5 Pro connection test successful")
                        return True
            
            self.logger.error(f"Gemini connection test failed: {response.status_code} - {response.text}")
            return False
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Gemini connection error: {e}")
            self.llm_available = False
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error testing Gemini connection: {e}")
            return False
    
    def encode_image_for_llm(self, image_tensor: torch.Tensor) -> str:
        """
        Encode image tensor as base64 string for LLM transmission.
        
        Args:
            image_tensor: Image tensor (1, 3, 224, 224)
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Convert tensor to PIL Image
            img_np = image_tensor.detach().squeeze(0).cpu().numpy()
            img_np = (img_np + 1.0) / 2.0  # Denormalize from [-1, 1] to [0, 1]
            img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
            img_np = (img_np * 255).astype(np.uint8)
            
            img_pil = Image.fromarray(img_np)
            
            # Convert to base64
            buffer = BytesIO()
            img_pil.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            self.logger.error(f"Error encoding image for LLM: {e}")
            return ""
    
    def create_medical_report_prompt(self, 
                                   condition_predictions: Dict, 
                                   prediction_keywords: Dict,
                                   spatial_keywords: List[str],
                                   attention_analysis: Dict,
                                   relevant_anatomical_regions: Dict = None,  # NEW
                                   template: str = 'standard',
                                   patient_info: Dict = None,
                                   include_image: bool = False) -> str:
        """
        Create an optimized prompt for clinical-grade medical report generation.
        Uses advanced prompt engineering to solve attention-pathology confusion
        and implement confidence-based reporting.
        
        Args:
            condition_predictions: Model predictions for conditions
            prediction_keywords: Keywords extracted from predictions
            spatial_keywords: Keywords from attention analysis (visualization only)
            attention_analysis: Detailed attention analysis results (visualization only)
            template: Report template to use ('standard', 'detailed', 'concise')
            patient_info: Optional patient information
            
        Returns:
            Formatted prompt string for LLM
        """
        try:
            template_config = self.report_templates.get(template, self.report_templates['standard'])
            
            # Determine reporting approach based on all significant findings
            reporting_style = self._determine_reporting_style_multi_condition(condition_predictions)
            
            # System instruction with optimized clinical guidelines
            image_note = "\n\n🖼️ IMPORTANT: You have been provided with the actual chest X-ray image. Use visual analysis of the image to correlate with and validate the AI predictions below." if include_image else ""
            
            system_instruction = f"""You are an expert radiologist with 20+ years of experience. Generate a concise, accurate chest X-ray report that combines visual analysis with AI predictions.{image_note}

✅ Your report combines direct visual analysis of the chest X-ray with AI model predictions.
✅ Use clear radiological terminology and anatomical specificity based on **visual findings and model predictions**.

### 🚨 REPORTING GUIDELINES:

1. **VISUAL-AI CORRELATION**
   • Examine the provided chest X-ray image directly
   • Correlate visual findings with AI predictions 
   • Prioritize what you see in the image over AI predictions when they conflict
   • Use AI predictions to guide attention to specific areas

2. **CONFIDENCE-BASED REPORTING**
   • >70% = Report directly and confidently
   • 50–70% = Use appropriate clinical uncertainty
   • <50% = Do not report

3. **INCLUDE DEVICE FINDINGS**
   • Always describe any visible medical device (e.g. tubes, catheters, lines), even if incidental
   • Mention if the **tip** is not visible or fully imaged
   • Report device positioning and termination when visible

3. **USE PROVIDED TERMINOLOGY**
   • Prefer using **exact phrases** from `CLINICAL KEYWORDS` to improve alignment with ground truth
   • When high-confidence keywords are provided, incorporate them verbatim when clinically appropriate

4. **AVOID OVER-HEDGING**
   • Do not say "subtle findings cannot be excluded" unless prediction confidence is mixed (50–70%)
   • If the study is normal and high confidence, use definitive phrases: "No focal consolidation, pleural effusion, or pneumothorax."
   • Be decisive when model confidence is high (>70%)

5. **STYLE & STRUCTURE**
   • Match expert radiologist tone
   • Avoid unnecessary hedging or speculation
   • Each section (FINDINGS, IMPRESSION) should be continuous text (no bullet points)
   • Include non-pathological findings such as tubes, lines, or structural anomalies

6. **ANATOMICAL SPECIFICITY**
   • Use precise anatomical terms when supported by high-confidence predictions
   • Reference specific lung zones, cardiac contours, and bony structures as appropriate
   • Always mention any visible medical device, line, or tube if present

REPORTING STYLE: {reporting_style}
"""

            # Clinical data section - SEPARATED from attention data
            clinical_data = f"""
=== CLINICAL ANALYSIS DATA ===

🔬 MODEL PREDICTIONS (Clinical Decision Basis):
{self._format_clinical_predictions(condition_predictions)}

📋 CLINICAL KEYWORDS (Condition-Based):
{self._format_clinical_keywords(prediction_keywords, condition_predictions)}

🗺️ RELEVANT ANATOMICAL REGIONS (Condition-Based):
{self._format_relevant_anatomical_regions(relevant_anatomical_regions, condition_predictions)}
"""

            # Patient context if available
            if patient_info:
                clinical_data += f"\n👤 PATIENT INFORMATION:\n"
                for key, value in patient_info.items():
                    clinical_data += f"- {key.replace('_', ' ').title()}: {value}\n"

            # Task instruction with optimized guidance
            task_instruction = f"""
=== REPORTING TASK ===
Generate a {template_config['style']} with sections: {', '.join(template_config['sections']).upper()}

🎯 SPECIFIC INSTRUCTIONS FOR THIS CASE:
{self._get_case_specific_instructions(condition_predictions, template)}

### 📏 FORMATTING INSTRUCTIONS:
• Structure:
  ```
  FINDINGS: 
  [continuous paragraph]  

  IMPRESSION:  
  [continuous paragraph]
  ```

### ✳️ INPUT STRUCTURE:
• `CHEST X-RAY IMAGE`: Primary source - examine visually for all findings
• `CLINICAL KEYWORDS`: Use exact phrases when clinically appropriate to maximize alignment
• `MODEL PREDICTIONS`: Secondary guide - use to focus attention and validate visual findings
• `RELEVANT ANATOMICAL REGIONS`: Reference these locations when describing findings

### 🎯 OPTIMIZATION GOALS:
• **Maximize lexical similarity** to expert reference reports
• **Use provided terminology verbatim** when possible
• **Include device findings** (tubes, catheters, lines) even if incidental
• **Be anatomically specific** when high-confidence predictions support it

📋 EXAMPLE REPORT FORMATS:

Example 1 (Device Present):
FINDINGS: 
Feeding tube extends into the upper abdomen, the tip is not imaged. Lung volumes are normal. Mediastinal contours and heart size within normal limits. No consolidation or pleural effusion. No pneumothorax. No acute osseous abnormality.

IMPRESSION: 
No acute cardiopulmonary process.

Example 2 (Multiple Findings):
FINDINGS: 
PA and lateral views of the chest demonstrate well-expanded lungs. In comparison to the prior study, there is interval obscuration of the right heart border and the medial right hemidiaphragm. Correlation with the lateral view suggests that this is likely due to interval development of small bilateral pleural effusions. Underlying consolidation is not excluded. No pneumothorax. Cardiomediastinal silhouette is otherwise stable.

IMPRESSION: 
Interval development of small bilateral pleural effusions. Underlying consolidation not excluded.

Example 3 (Normal Study):
FINDINGS: 
The lungs are hyperinflated reflective of COPD. Apparent increased opacity projecting over the right lung apex correlates with posterior right fifth rib fracture with callus. Streaky bibasilar opacities likely reflect atelectasis. No focal consolidation to suggest pneumonia. No pleural effusion or pneumothorax. The heart is normal in size, and the mediastinal contours are normal.

IMPRESSION: 
No acute cardiopulmonary process. Focal opacity in the retrocardiac region.

⚠️ **REMEMBER**: Do NOT mention attention maps, saliency, heatmaps, or explainability data. Use model predictions and provided keywords only.

### ✳️ GOAL:
Maximize lexical and semantic similarity to the expert reference report. Prioritize clinical specificity and exact terminology alignment.

CHEST X-RAY REPORT:
"""
            
            # Combine sections
            full_prompt = system_instruction + clinical_data + task_instruction
            
            self.logger.debug(f"Created optimized medical report prompt ({len(full_prompt)} characters)")
            return full_prompt
            
        except Exception as e:
            self.logger.error(f"Error creating medical report prompt: {e}")
            return "Error creating prompt. Please provide chest X-ray analysis for medical report generation."

    def _get_primary_finding(self, condition_predictions: Dict) -> tuple:
        """Determine the primary finding and its confidence level."""
        # Find highest confidence prediction
        max_prob = 0
        primary_condition = "No Finding"
        
        for condition, pred_info in condition_predictions.items():
            prob = pred_info.get('probability', 0)
            if prob > max_prob:
                max_prob = prob
                primary_condition = condition
                
        return primary_condition, max_prob

    def _determine_reporting_style_multi_condition(self, condition_predictions: Dict) -> str:
        """Determine appropriate reporting style based on all significant findings."""
        
        # Get significant findings (≥60% confidence)
        significant_findings = [
            (condition, pred_info['probability']) 
            for condition, pred_info in condition_predictions.items()
            if pred_info.get('probability', 0) >= 0.55
        ]
        
        # Sort by confidence
        significant_findings = sorted(significant_findings, key=lambda x: x[1], reverse=True)[:3]
        
        # Check for normal study
        no_finding_prob = condition_predictions.get('No Finding', {}).get('probability', 0)
        if no_finding_prob > 0.70 and len([f for f in significant_findings if f[0] != 'No Finding']) == 0:
            return "SIMPLE & DIRECT (Normal Study - Radiologist-style brevity required)"
        
        # Multi-condition assessment
        if len(significant_findings) >= 2:
            high_conf_count = sum(1 for _, prob in significant_findings if prob > 0.70)
            if high_conf_count >= 2:
                return "MULTI-PATHOLOGY HIGH CONFIDENCE (Multiple significant findings - comprehensive analysis required)"
            else:
                return "MULTI-PATHOLOGY MIXED CONFIDENCE (Multiple findings with varying confidence levels)"
        
        # Single condition
        elif len(significant_findings) == 1:
            _, confidence = significant_findings[0]
            if confidence > 0.70:
                return "FOCUSED PATHOLOGY (High-confidence single finding - focus on medical significance)"
            else:
                return "MODERATE PATHOLOGY (Single moderate-confidence finding - appropriate uncertainty)"
        
        # Low confidence across all conditions
        else:
            return "CONSERVATIVE ASSESSMENT (Low confidence across all findings - emphasize limitations)"

    def _format_clinical_predictions(self, condition_predictions: Dict) -> str:
        """Format only clinically significant predictions."""
        formatted = ""
        for condition, pred_info in condition_predictions.items():
            prob = pred_info.get('probability', 0)
            predicted = pred_info.get('predicted', False)
            
            if prob > 0.30 or predicted:  # Only show clinically relevant predictions
                confidence_level = "HIGH" if prob > 0.70 else "MODERATE" if prob > 0.50 else "LOW"
                formatted += f"- {condition}: {prob:.1%} probability ({confidence_level} confidence)\n"
                
        return formatted.strip()

    def _format_clinical_keywords(self, prediction_keywords: Dict, condition_predictions: Dict) -> str:
        """Format keywords only for significant clinical findings with emphasis on exact usage."""
        formatted = ""
        high_confidence_phrases = []
        
        for condition, keywords in prediction_keywords.items():
            prob = condition_predictions.get(condition, {}).get('probability', 0)
            if keywords and prob > 0.30:
                confidence_level = "HIGH" if prob > 0.70 else "MODERATE" if prob > 0.50 else "LOW"
                
                # Prioritize high-confidence keywords for verbatim usage
                if prob > 0.70:
                    high_confidence_phrases.extend(keywords[:2])  # Top 2 for high confidence
                    formatted += f"- {condition} ({confidence_level}): {', '.join(keywords[:3])} [USE VERBATIM]\n"
                else:
                    formatted += f"- {condition} ({confidence_level}): {', '.join(keywords[:3])}\n"
        
        if high_confidence_phrases:
            formatted += f"\n🎯 HIGH-CONFIDENCE PHRASES TO USE VERBATIM: {', '.join(set(high_confidence_phrases[:5]))}\n"
                
        return formatted.strip() if formatted else "- Based on model predictions only"

    def _format_relevant_anatomical_regions(self, relevant_regions: Dict, condition_predictions: Dict) -> str:
        """Format relevant anatomical regions based on predicted conditions with 60% threshold."""
        if not relevant_regions:
            return "- Anatomical regions determined by static condition-to-region mapping"
        
        # Get significant conditions (≥60% confidence)
        significant_conditions = {
            condition: pred_info['probability'] 
            for condition, pred_info in condition_predictions.items()
            if pred_info.get('probability', 0) >= 0.60
        }
        
        if not significant_conditions:
            return "- No significant findings above 60% confidence for anatomical localization"
        
        formatted = ""
        
        # Group by condition for better clinical presentation
        for condition, probability in sorted(significant_conditions.items(), key=lambda x: x[1], reverse=True):
            relevant_anatomical_regions = []
            for region_key, region_info in relevant_regions.items():
                related_conditions = region_info.get('related_conditions', [])
                for rc in related_conditions:
                    if rc['condition'] == condition and rc['probability'] >= 0.60:
                        relevant_anatomical_regions.append(region_info['name'])
                        break
            
            if relevant_anatomical_regions:
                confidence_level = "HIGH" if probability > 0.70 else "MODERATE"
                formatted += f"- {condition} ({probability:.1%}, {confidence_level}): {', '.join(relevant_anatomical_regions)}\n"
        
        return formatted.strip() if formatted else "- No specific anatomical regions identified for significant findings"

    def _get_case_specific_instructions(self, condition_predictions: Dict, template: str) -> str:
        """Provide specific instructions based on the case characteristics and all significant findings."""
        
        # Get all significant findings (≥60% confidence)
        significant_findings = [
            (condition, pred_info['probability']) 
            for condition, pred_info in condition_predictions.items()
            if pred_info.get('probability', 0) >= 0.60
        ]
        
        # Sort by confidence and take top 3
        significant_findings = sorted(significant_findings, key=lambda x: x[1], reverse=True)[:3]
        
        # Check for No Finding dominance
        no_finding_prob = condition_predictions.get('No Finding', {}).get('probability', 0)
        is_normal_study = (no_finding_prob > 0.70 and 
                          len([f for f in significant_findings if f[0] != 'No Finding']) == 0)
        
        if is_normal_study:
            return """- This is a NORMAL study with high confidence
- Use simple, direct language with definitive statements
- Include any visible medical devices or support equipment
- Use exact phrases: "No focal consolidation, pleural effusion, or pneumothorax"
- Keep findings section concise but anatomically specific
- Impression should be brief and definitive: "No acute cardiopulmonary process"
- Avoid hedging language like "cannot be excluded" """
        
        elif len(significant_findings) == 0:
            return """- Low confidence findings across all conditions
- Emphasize limitations of the analysis
- Suggest clinical correlation
- Consider additional imaging if warranted
- Use conservative language throughout"""
        
        elif len(significant_findings) == 1:
            condition, confidence = significant_findings[0]
            confidence_level = "HIGH" if confidence > 0.70 else "MODERATE"
            return f"""- This shows {condition} with {confidence_level} confidence ({confidence:.1%})
- Focus specifically on this pathological finding using PROVIDED KEYWORDS verbatim
- Use relevant anatomical regions to describe location precisely
- Include any visible medical devices (tubes, catheters, lines) even if incidental
- Use exact terminology from CLINICAL KEYWORDS when available
- Avoid over-hedging if confidence is high (>70%)
- Describe the medical significance clearly"""
        
        else:  # Multiple significant findings
            conditions_text = ', '.join([f"{cond} ({prob:.1%})" for cond, prob in significant_findings])
            return f"""- This case shows MULTIPLE significant findings: {conditions_text}
- Report ALL significant findings above 60% confidence using PROVIDED KEYWORDS verbatim
- Use relevant anatomical regions to specify locations for each finding precisely
- Include any visible medical devices (tubes, catheters, lines) even if incidental
- Use exact terminology from CLINICAL KEYWORDS when available for each condition
- Describe the medical significance of each finding without over-hedging
- Consider relationships between findings (e.g., atelectasis with pleural effusion)
- Structure as: FINDINGS (describe each with specific terminology), IMPRESSION (synthesize findings)"""
    
    def call_llm_for_report(self, prompt: str, include_image: bool = False, 
                           image_tensor: torch.Tensor = None) -> Dict:
        """
        Call Gemini 2.5 Pro to generate a medical report.
        
        Args:
            prompt: Formatted prompt for report generation
            include_image: Whether to include the image in the request
            image_tensor: Image tensor if including image
            
        Returns:
            Dictionary with response information
        """
        try:
            if not self.llm_connection_tested:
                if not self.test_llm_connection():
                    return {
                        'success': False,
                        'error': 'Gemini connection not available',
                        'report': self._generate_fallback_report(),
                        'used_fallback': True
                    }
            
            if not self.gemini_api_key:
                return {
                    'success': False,
                    'error': 'Gemini API key not available',
                    'report': self._generate_fallback_report(),
                    'used_fallback': True
                }
            
            # Prepare message content with image support
            if include_image and image_tensor is not None:
                # Encode image as base64
                image_base64 = self.encode_image_for_llm(image_tensor)
                if image_base64:
                    self.logger.info("Including image in Gemini request")
                else:
                    self.logger.warning("Failed to encode image, proceeding with text-only")
                    include_image = False
            else:
                include_image = False
            
            # Prepare request payload for Gemini
            if include_image and 'image_base64' in locals():
                # Include both text and image
                payload = {
                    "contents": [{
                        "parts": [
                            {
                                "text": prompt
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": image_base64
                                }
                            }
                        ]
                    }],
                    "generationConfig": {
                        "temperature": self.llm_config['temperature'],
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": self.llm_config['max_tokens'],
                        "candidateCount": 1
                    },
                    "safetySettings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH", 
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE"
                        }
                    ]
                }
            else:
                # Text only
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": self.llm_config['temperature'],
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": self.llm_config['max_tokens'],
                        "candidateCount": 1
                    },
                    "safetySettings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH", 
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE"
                        }
                    ]
                }
            
            url = f"{self.llm_config['base_url']}?key={self.gemini_api_key}"
            headers = {"Content-Type": "application/json"}
            
            # Helper to strip non-ASCII characters (e.g., emojis) for safe console logging
            def _strip_non_ascii(s: str) -> str:
                return s.encode('ascii', 'ignore').decode()
            
            # Debug: Log the prompt being sent
            self.logger.info(_strip_non_ascii(f"DEBUG: Prompt length: {len(prompt)} chars"))
            self.logger.info(_strip_non_ascii(f"DEBUG: First 200 chars of prompt: {prompt[:200]}..."))
            
            # Make request with retries
            for attempt in range(self.llm_config['max_retries']):
                try:
                    self.logger.debug(f"Gemini request attempt {attempt + 1}/{self.llm_config['max_retries']}")
                    
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.llm_config['timeout']
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if 'candidates' in result and len(result['candidates']) > 0:
                            candidate = result['candidates'][0]
                            
                            if 'content' in candidate:
                                # Check if we have parts with content
                                if 'parts' in candidate['content'] and len(candidate['content']['parts']) > 0:
                                    report_text = candidate['content']['parts'][0]['text']
                                    
                                    return {
                                        'success': True,
                                        'report': report_text,
                                        'raw_response': result,
                                        'used_fallback': False,
                                        'attempt': attempt + 1
                                    }
                                else:
                                    # Content exists but no parts - likely MAX_TOKENS hit
                                    self.logger.error(f"Gemini response has no content parts. Finish reason: {candidate.get('finishReason', 'unknown')}")
                                    return {
                                        'success': False,
                                        'error': f"Response has no content parts. Finish reason: {candidate.get('finishReason', 'unknown')}",
                                        'report': self._generate_fallback_report(),
                                        'used_fallback': True
                                    }
                            elif 'finishReason' in candidate and candidate['finishReason'] == 'MAX_TOKENS':
                                self.logger.error("MAX_TOKENS hit - response was too long")
                                return {
                                    'success': False,
                                    'error': 'Response exceeded maximum tokens',
                                    'report': self._generate_fallback_report(),
                                    'used_fallback': True
                                }
                    
                    elif response.status_code == 429:  # Rate limit
                        wait_time = self.llm_config['retry_delay'] * (2 ** attempt)
                        self.logger.warning(f"Rate limit hit (attempt {attempt + 1}), waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    
                    self.logger.warning(f"Gemini request failed (attempt {attempt + 1}): {response.status_code} - {response.text}")
                    
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"Gemini request error (attempt {attempt + 1}): {e}")
                
                # Wait before retry
                if attempt < self.llm_config['max_retries'] - 1:
                    time.sleep(self.llm_config['retry_delay'])
            
            # All attempts failed
            self.logger.error("All Gemini request attempts failed")
            return {
                'success': False,
                'error': 'All Gemini request attempts failed',
                'report': self._generate_fallback_report(),
                'used_fallback': True
            }
            
        except Exception as e:
            self.logger.error(f"Error calling Gemini for report: {e}")
            return {
                'success': False,
                'error': str(e),
                'report': self._generate_fallback_report(),
                'used_fallback': True
            }
    
    def _generate_fallback_report(self) -> str:
        """
        Generate a fallback report when LLM is not available.
        
        Returns:
            Basic medical report template
        """
        return """CHEST X-RAY REPORT

FINDINGS:
Chest X-ray analysis has been completed using automated multimodal AI analysis.
Detailed findings are available in the system logs.

IMPRESSION:
Automated analysis completed. Please refer to the detailed predictions and 
keywords for specific findings. Clinical correlation recommended.

RECOMMENDATIONS:
1. Review automated analysis results with attending radiologist
2. Clinical correlation with patient symptoms and history
3. Follow-up imaging as clinically indicated

NOTE: This report was generated using fallback mode due to LLM unavailability.
Please ensure proper clinical review of all automated findings.

[Report generated by Medical Report Generator - Fallback Mode]
"""
    
    def format_medical_report(self, raw_report: str, metadata: Dict = None) -> Dict:
        """
        Format and validate the medical report output.
        
        Args:
            raw_report: Raw report text from LLM
            metadata: Additional metadata to include
            
        Returns:
            Formatted report dictionary
        """
        try:
            # Basic formatting and cleanup
            formatted_text = raw_report.strip()
            
            # Split into sections (basic parsing)
            sections = {}
            current_section = "report"
            current_content = []
            
            lines = formatted_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line is a section header
                line_upper = line.upper()
                if any(header in line_upper for header in ['FINDINGS:', 'IMPRESSION:', 'RECOMMENDATIONS:', 'CLINICAL HISTORY:', 'TECHNIQUE:']):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    
                    # Start new section
                    current_section = line_upper.replace(':', '').lower()
                    current_content = []
                else:
                    current_content.append(line)
            
            # Save last section
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Create formatted report
            formatted_report = {
                'report_text': formatted_text,
                'sections': sections,
                'metadata': {
                    'generated_at': str(np.datetime64('now')),
                    'generator': 'Medical Report Generator v1.0',
                    'model_used': self.llm_config['model_name'],
                    'word_count': len(formatted_text.split()),
                    'character_count': len(formatted_text)
                }
            }
            
            # Add additional metadata if provided
            if metadata:
                formatted_report['metadata'].update(metadata)
            
            self.logger.debug(f"Formatted medical report: {len(sections)} sections, {formatted_report['metadata']['word_count']} words")
            return formatted_report
            
        except Exception as e:
            self.logger.error(f"Error formatting medical report: {e}")
            return {
                'report_text': raw_report,
                'sections': {'report': raw_report},
                'metadata': {
                    'generated_at': str(np.datetime64('now')),
                    'generator': 'Medical Report Generator v1.0',
                    'error': str(e)
                }
            }
    
    def generate_complete_medical_report(self, 
                                       image_path: str,
                                       template: str = 'standard',
                                       include_image: bool = False,
                                       patient_info: Dict = None) -> Dict:
        """
        Generate a complete medical report from image analysis.
        
        Args:
            image_path: Path to the chest X-ray image
            template: Report template to use
            include_image: Whether to include image in LLM request
            patient_info: Optional patient information
            
        Returns:
            Complete report analysis dictionary
        """
        try:
            self.logger.info(f"Generating complete medical report for: {image_path}")
            
            # Step 1: Load and preprocess image
            img_tensor = self.preprocess_image(image_path)
            
            # Step 2: Run model inference
            inference_results = self.test_end_to_end_inference(image_path)
            if not inference_results["success"]:
                raise RuntimeError("Model inference failed")
            
            # Step 3: Extract prediction keywords
            prediction_keywords = self.extract_keywords_from_predictions(inference_results["condition_predictions"])
            
            # Step 4: Get relevant anatomical regions (NEW - static mapping)
            relevant_regions = self.get_relevant_anatomical_regions(inference_results["condition_predictions"])
            
            # Step 5: Generate attention analysis (existing - but now focused on relevant regions)
            top_condition = max(
                inference_results["condition_predictions"].items(),
                key=lambda x: x[1]['probability']
            )[0]
            
            attention_map = self.generate_enhanced_attention(img_tensor, target_condition=top_condition)
            attention_analysis = self.analyze_attention_regions(attention_map)
            spatial_keywords = self.extract_spatial_keywords(attention_analysis)
            
            # Step 6: Create LLM prompt (UPDATED - include relevant regions)
            prompt = self.create_medical_report_prompt(
                condition_predictions=inference_results["condition_predictions"],
                prediction_keywords=prediction_keywords,
                spatial_keywords=spatial_keywords,
                attention_analysis=attention_analysis,
                relevant_anatomical_regions=relevant_regions,  # NEW
                template=template,
                patient_info=patient_info
            )
            
            # Step 7: Generate report via LLM
            llm_response = self.call_llm_for_report(
                prompt=prompt,
                include_image=include_image,
                image_tensor=img_tensor if include_image else None
            )
            
            # Step 8: Format final report
            formatted_report = self.format_medical_report(
                raw_report=llm_response['report'],
                metadata={
                    'image_path': str(image_path),
                    'template_used': template,
                    'llm_success': llm_response['success'],
                    'used_fallback': llm_response.get('used_fallback', False),
                    'top_predicted_condition': top_condition,
                    'total_keywords': len(prediction_keywords) + len(spatial_keywords),
                    'attention_regions': len(attention_analysis.get('significant_regions', [])),
                    'relevant_anatomical_regions': len(relevant_regions)  # NEW
                }
            )
            
            # Step 9: Compile complete analysis
            complete_analysis = {
                'success': True,
                'image_path': str(image_path),
                'model_predictions': inference_results["condition_predictions"],
                'prediction_keywords': prediction_keywords,
                'relevant_anatomical_regions': relevant_regions,  # NEW
                'spatial_keywords': spatial_keywords,
                'attention_analysis': attention_analysis,
                'llm_response': llm_response,
                'medical_report': formatted_report,
                'processing_metadata': {
                    'phases_completed': ['inference', 'keywords', 'anatomical_regions', 'attention', 'llm_generation'],  # UPDATED
                    'template_used': template,
                    'total_processing_time': None  # Could add timing if needed
                }
            }
            
            self.logger.info("Complete medical report generation successful")
            return complete_analysis
            
        except Exception as e:
            self.logger.error(f"Error generating complete medical report: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_path': str(image_path) if 'image_path' in locals() else None,
                'medical_report': {
                    'report_text': self._generate_fallback_report(),
                    'metadata': {'error': str(e)}
                }
            }
    
    def test_llm_integration_system(self) -> bool:
        """Test the complete LLM integration system"""
        try:
            self.logger.info("Testing LLM integration system...")
            
            # Test 1: LLM Connection
            if not self.test_llm_connection():
                self.logger.warning("LLM connection test failed - testing with fallback mode")
            
            # Test 2: Prompt Creation
            dummy_predictions = {
                'Pneumonia': {'probability': 0.8, 'predicted': True},
                'No Finding': {'probability': 0.2, 'predicted': False}
            }
            dummy_keywords = {'Pneumonia': ['pneumonia', 'consolidation']}
            dummy_spatial = ['right lower lobe', 'focal attention']
            dummy_attention = {
                'overall_attention_intensity': 0.7,
                'significant_regions': ['lower_right'],
                'regions': {
                    'lower_right': {'name': 'right lower lung field', 'max_attention': 0.9}
                }
            }
            
            prompt = self.create_medical_report_prompt(
                dummy_predictions, dummy_keywords, dummy_spatial, dummy_attention
            )
            
            if len(prompt) < 100:
                self.logger.error("Test 2 failed: Prompt too short")
                return False
            
            # Test 3: Report Generation (with fallback)
            llm_response = self.call_llm_for_report(prompt)
            if not isinstance(llm_response, dict) or 'report' not in llm_response:
                self.logger.error("Test 3 failed: Invalid LLM response format")
                return False
            
            # Test 4: Report Formatting
            formatted = self.format_medical_report(llm_response['report'])
            if not isinstance(formatted, dict) or 'report_text' not in formatted:
                self.logger.error("Test 4 failed: Report formatting failed")
                return False
            
            self.logger.info("LLM integration system tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"LLM integration system test failed: {e}")
            return False

    # ========================
    # Phase 5: Complete System Integration
    # ========================
    
    def generate_comprehensive_analysis(self, 
                                       image_path: str,
                                       template: str = 'standard',
                                       include_image_in_llm: bool = False,
                                       patient_info: Dict = None,
                                       save_visualizations: bool = False,
                                       progress_callback: callable = None) -> Dict:
        """
        Complete end-to-end medical analysis pipeline with progress tracking.
        
        Args:
            image_path: Path to the chest X-ray image
            template: Report template ('standard', 'detailed', 'concise')
            include_image_in_llm: Whether to send image to LLM
            patient_info: Optional patient information
            save_visualizations: Whether to save attention visualizations
            progress_callback: Optional callback for progress updates
            
        Returns:
            Comprehensive analysis dictionary with all components
        """
        def update_progress(step: str, progress: float, message: str = ""):
            if progress_callback:
                progress_callback(step, progress, message)
            self.logger.info(f"[{progress:.0f}%] {step}: {message}")
        
        try:
            update_progress("Initialization", 0, "Starting comprehensive analysis")
            
            # Initialize result structure
            analysis_result = {
                'success': False,
                'image_path': str(image_path),
                'template_used': template,
                'processing_steps': {},
                'performance_metrics': {},
                'error_log': []
            }
            
            start_time = time.time()
            
            # Step 1: Image Processing (10%)
            update_progress("Image Processing", 10, "Loading and preprocessing image")
            step_start = time.time()
            
            try:
                img_tensor = self.preprocess_image(image_path)
                analysis_result['processing_steps']['image_processing'] = {
                    'success': True,
                    'shape': list(img_tensor.shape),
                    'device': str(img_tensor.device),
                    'processing_time': time.time() - step_start
                }
                update_progress("Image Processing", 15, "Image preprocessing completed")
            except Exception as e:
                error_msg = f"Image processing failed: {str(e)}"
                analysis_result['error_log'].append(error_msg)
                self.logger.error(error_msg)
                return analysis_result
            
            # Step 2: Model Inference (25%)
            update_progress("Model Inference", 25, "Running multimodal model inference")
            step_start = time.time()
            
            try:
                inference_results = self.run_model_inference(image_path)
                if not inference_results["success"]:
                    raise RuntimeError("Model inference failed")
                
                analysis_result['model_predictions'] = inference_results["condition_predictions"]
                analysis_result['processing_steps']['model_inference'] = {
                    'success': True,
                    'num_conditions': len(inference_results["condition_predictions"]),
                    'processing_time': time.time() - step_start
                }
                update_progress("Model Inference", 35, "Model inference completed")
            except Exception as e:
                error_msg = f"Model inference failed: {str(e)}"
                analysis_result['error_log'].append(error_msg)
                self.logger.error(error_msg)
                return analysis_result
            
            # Step 3: Keyword Extraction (45%)
            update_progress("Keyword Extraction", 45, "Extracting medical keywords")
            step_start = time.time()
            
            try:
                prediction_keywords = self.extract_keywords_from_predictions(
                    analysis_result['model_predictions']
                )
                keyword_summary = self.get_keyword_summary(prediction_keywords)
                
                analysis_result['prediction_keywords'] = prediction_keywords
                analysis_result['keyword_summary'] = keyword_summary
                analysis_result['processing_steps']['keyword_extraction'] = {
                    'success': True,
                    'total_keywords': keyword_summary['total_keywords'],
                    'conditions_with_keywords': len(prediction_keywords),
                    'processing_time': time.time() - step_start
                }
                update_progress("Keyword Extraction", 55, f"Extracted {keyword_summary['total_keywords']} keywords")
            except Exception as e:
                error_msg = f"Keyword extraction failed: {str(e)}"
                analysis_result['error_log'].append(error_msg)
                self.logger.error(error_msg)
                # Continue with empty keywords
                analysis_result['prediction_keywords'] = {}
                analysis_result['keyword_summary'] = {'total_keywords': 0}
            
            # Step 4: Anatomical Region Mapping (55%)
            update_progress("Anatomical Region Mapping", 55, "Identifying relevant anatomical regions")
            step_start = time.time()
            
            try:
                relevant_regions = self.get_relevant_anatomical_regions(analysis_result['model_predictions'])
                
                analysis_result['relevant_anatomical_regions'] = relevant_regions
                analysis_result['processing_steps']['anatomical_region_mapping'] = {
                    'success': True,
                    'relevant_regions_count': len(relevant_regions),
                    'processing_time': time.time() - step_start
                }
                update_progress("Anatomical Region Mapping", 65, f"Identified {len(relevant_regions)} relevant regions")
            except Exception as e:
                error_msg = f"Anatomical region mapping failed: {str(e)}"
                analysis_result['error_log'].append(error_msg)
                self.logger.error(error_msg)
                # Continue with empty regions
                analysis_result['relevant_anatomical_regions'] = {}

            # Step 5: Attention Analysis (75%)
            update_progress("Attention Analysis", 75, "Generating attention visualization")
            step_start = time.time()
            
            try:
                # Determine target condition for attention
                top_condition = max(
                    analysis_result['model_predictions'].items(),
                    key=lambda x: x[1]['probability']
                )[0]
                
                attention_map = self.generate_enhanced_attention(img_tensor, target_condition=top_condition)
                attention_analysis = self.analyze_attention_regions(attention_map)
                spatial_keywords = self.extract_spatial_keywords(attention_analysis)
                
                # Save visualization if requested
                if save_visualizations:
                    vis_dir = os.path.join("test_data", "attention_visualizations")
                    os.makedirs(vis_dir, exist_ok=True)
                    vis_path = os.path.join(vis_dir, f"attention_{Path(image_path).stem}.png")
                    self.visualize_attention_map(img_tensor, attention_map, save_path=vis_path)
                    attention_analysis['visualization_saved'] = vis_path
                
                analysis_result['attention_analysis'] = attention_analysis
                analysis_result['attention_analysis']['attention_map'] = attention_map.detach().cpu().numpy()  # Add raw attention map
                analysis_result['spatial_keywords'] = spatial_keywords
                analysis_result['processing_steps']['attention_analysis'] = {
                    'success': True,
                    'significant_regions': len(attention_analysis['significant_regions']),
                    'spatial_keywords': len(spatial_keywords),
                    'target_condition': top_condition,
                    'processing_time': time.time() - step_start
                }
                update_progress("Attention Analysis", 85, f"Analyzed {len(attention_analysis['significant_regions'])} regions")
            except Exception as e:
                error_msg = f"Attention analysis failed: {str(e)}"
                analysis_result['error_log'].append(error_msg)
                self.logger.error(error_msg)
                # Continue with empty attention
                analysis_result['attention_analysis'] = {'significant_regions': []}
                analysis_result['spatial_keywords'] = []
            
            # Step 6: LLM Report Generation (90%)
            update_progress("Report Generation", 90, "Generating medical report")
            step_start = time.time()
                        
            try:
                # Create comprehensive prompt
                prompt = self.create_medical_report_prompt(
                    condition_predictions=analysis_result['model_predictions'],
                    prediction_keywords=analysis_result['prediction_keywords'],
                    spatial_keywords=analysis_result['spatial_keywords'],
                    attention_analysis=analysis_result['attention_analysis'],
                    relevant_anatomical_regions=analysis_result['relevant_anatomical_regions'],  # NEW
                    template=template,
                    patient_info=patient_info,
                    include_image=include_image_in_llm
                )
                
                # Generate report
                llm_response = self.call_llm_for_report(
                    prompt=prompt,
                    include_image=include_image_in_llm,
                    image_tensor=img_tensor if include_image_in_llm else None
                )
                
                # Format final report
                formatted_report = self.format_medical_report(
                    raw_report=llm_response['report'],
                    metadata={
                        'image_path': str(image_path),
                        'template_used': template,
                        'llm_success': llm_response['success'],
                                            'used_fallback': llm_response.get('used_fallback', False),
                    'total_keywords': analysis_result['keyword_summary']['total_keywords'],
                    'attention_regions': len(analysis_result['attention_analysis']['significant_regions']),
                    'relevant_anatomical_regions': len(analysis_result['relevant_anatomical_regions']),  # NEW
                    'patient_info_included': patient_info is not None
                }
                )
                
                analysis_result['llm_response'] = llm_response
                analysis_result['medical_report'] = formatted_report
                analysis_result['processing_steps']['report_generation'] = {
                    'success': llm_response['success'],
                    'used_fallback': llm_response.get('used_fallback', False),
                    'report_length': len(formatted_report['report_text']),
                    'sections_found': len(formatted_report['sections']),
                    'processing_time': time.time() - step_start
                }
                update_progress("Report Generation", 98, "Medical report generated")
            except Exception as e:
                error_msg = f"Report generation failed: {str(e)}"
                analysis_result['error_log'].append(error_msg)
                self.logger.error(error_msg)
                # Generate fallback report
                fallback_report = self._generate_fallback_report()
                analysis_result['medical_report'] = self.format_medical_report(fallback_report)
                analysis_result['processing_steps']['report_generation'] = {
                    'success': False,
                    'used_fallback': True,
                    'error': str(e)
                }
            
            # Step 7: Finalization (100%)
            total_time = time.time() - start_time
            
            # Calculate performance metrics
            analysis_result['performance_metrics'] = {
                'total_processing_time': total_time,
                'steps_completed': len([s for s in analysis_result['processing_steps'].values() if s.get('success', False)]),
                'total_steps': 6,  # UPDATED: now 6 steps
                'errors_encountered': len(analysis_result['error_log']),
                'memory_efficient': self.low_memory,
                'device_used': str(self.device)
            }
            
            # Final validation
            analysis_result['success'] = self._validate_comprehensive_analysis(analysis_result)
            
            update_progress("Complete", 100, f"Analysis completed in {total_time:.1f}s")
            
            # Log summary
            self._log_analysis_summary(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            analysis_result['error_log'].append(f"System error: {str(e)}")
            analysis_result['success'] = False
            return analysis_result
    
    def _validate_comprehensive_analysis(self, analysis_result: Dict) -> bool:
        """Validate that comprehensive analysis completed successfully"""
        try:
            # Check essential components
            required_keys = ['model_predictions', 'medical_report', 'processing_steps']
            for key in required_keys:
                if key not in analysis_result:
                    self.logger.error(f"Missing required component: {key}")
                    return False
            
            # Check that medical report has content
            if len(analysis_result['medical_report']['report_text']) < 50:
                self.logger.error("Medical report too short")
                return False
            
            # Check that at least inference succeeded
            inference_success = analysis_result['processing_steps'].get('model_inference', {}).get('success', False)
            if not inference_success:
                self.logger.error("Model inference did not succeed")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False
    
    def _log_analysis_summary(self, analysis_result: Dict):
        """Log a comprehensive summary of the analysis"""
        try:
            self.logger.info("=== COMPREHENSIVE ANALYSIS SUMMARY ===")
            self.logger.info(f"Image: {analysis_result['image_path']}")
            self.logger.info(f"Success: {analysis_result['success']}")
            self.logger.info(f"Template: {analysis_result['template_used']}")
            
            # Performance metrics
            metrics = analysis_result['performance_metrics']
            self.logger.info(f"Total time: {metrics['total_processing_time']:.2f}s")
            self.logger.info(f"Steps completed: {metrics['steps_completed']}/{metrics['total_steps']}")
            self.logger.info(f"Errors: {metrics['errors_encountered']}")
            
            # Content summary
            if 'model_predictions' in analysis_result:
                top_predictions = sorted(
                    analysis_result['model_predictions'].items(),
                    key=lambda x: x[1]['probability'], reverse=True
                )[:3]
                self.logger.info("Top predictions:")
                for condition, pred in top_predictions:
                    self.logger.info(f"  {condition}: {pred['probability']:.3f}")
            
            if 'keyword_summary' in analysis_result:
                keywords = analysis_result['keyword_summary']['total_keywords']
                self.logger.info(f"Keywords extracted: {keywords}")
            
            if 'relevant_anatomical_regions' in analysis_result:
                relevant_regions = len(analysis_result['relevant_anatomical_regions'])
                self.logger.info(f"Relevant anatomical regions: {relevant_regions}")

            if 'attention_analysis' in analysis_result:
                attention_regions = len(analysis_result['attention_analysis']['significant_regions'])
                self.logger.info(f"Attention regions: {attention_regions}")
            
            if 'medical_report' in analysis_result:
                report_len = len(analysis_result['medical_report']['report_text'])
                sections = len(analysis_result['medical_report']['sections'])
                self.logger.info(f"Report: {report_len} chars, {sections} sections")
            
            self.logger.info("=" * 40)
            
        except Exception as e:
            self.logger.error(f"Error logging summary: {e}")
    
    def batch_analyze_images(self, 
                           image_paths: List[str],
                           template: str = 'standard',
                           max_concurrent: int = 2,
                           save_results: bool = True,
                           output_dir: str = "batch_analysis_results") -> Dict:
        """
        Analyze multiple images in batch with performance optimization.
        
        Args:
            image_paths: List of image paths to analyze
            template: Report template to use
            max_concurrent: Maximum concurrent analyses (for memory management)
            save_results: Whether to save individual results
            output_dir: Directory to save results
            
        Returns:
            Batch analysis results
        """
        try:
            self.logger.info(f"Starting batch analysis of {len(image_paths)} images")
            
            if save_results:
                os.makedirs(output_dir, exist_ok=True)
            
            batch_results = {
                'total_images': len(image_paths),
                'successful_analyses': 0,
                'failed_analyses': 0,
                'results': {},
                'performance_summary': {},
                'error_summary': []
            }
            
            start_time = time.time()
            
            # Process images in chunks for memory efficiency
            chunk_size = max_concurrent
            for i in range(0, len(image_paths), chunk_size):
                chunk_paths = image_paths[i:i+chunk_size]
                
                self.logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(image_paths)-1)//chunk_size + 1}")
                
                for image_path in chunk_paths:
                    try:
                        # Individual analysis
                        analysis = self.generate_comprehensive_analysis(
                            image_path=image_path,
                            template=template,
                            save_visualizations=save_results
                        )
                        
                        if analysis['success']:
                            batch_results['successful_analyses'] += 1
                        else:
                            batch_results['failed_analyses'] += 1
                            batch_results['error_summary'].extend(analysis.get('error_log', []))
                        
                        batch_results['results'][image_path] = analysis
                        
                        # Save individual result if requested
                        if save_results:
                            result_file = os.path.join(
                                output_dir, 
                                f"analysis_{Path(image_path).stem}.json"
                            )
                            with open(result_file, 'w') as f:
                                json.dump(analysis, f, indent=2, default=str)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to analyze {image_path}: {e}")
                        batch_results['failed_analyses'] += 1
                        batch_results['error_summary'].append(f"{image_path}: {str(e)}")
                
                # Memory cleanup between chunks
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Calculate performance summary
            total_time = time.time() - start_time
            batch_results['performance_summary'] = {
                'total_time': total_time,
                'average_time_per_image': total_time / len(image_paths),
                'success_rate': batch_results['successful_analyses'] / len(image_paths) * 100,
                'device_used': str(self.device),
                'memory_mode': 'low' if self.low_memory else 'high'
            }
            
            # Save batch summary - DISABLED (no aggregate file desired)
            # if save_results:
            #     summary_file = os.path.join(output_dir, "batch_summary.json")
            #     with open(summary_file, 'w') as f:
            #         json.dump(batch_results, f, indent=2, default=str)
            
            self.logger.info(f"Batch analysis completed: {batch_results['successful_analyses']}/{len(image_paths)} successful")
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch analysis failed: {e}")
            return {
                'total_images': len(image_paths),
                'successful_analyses': 0,
                'failed_analyses': len(image_paths),
                'error': str(e)
            }
    
    def optimize_performance(self, enable_caching: bool = True, 
                           cleanup_frequency: int = 5) -> Dict:
        """
        Optimize system performance with caching and memory management.
        
        Args:
            enable_caching: Whether to enable attention map caching
            cleanup_frequency: How often to perform memory cleanup
            
        Returns:
            Performance optimization settings
        """
        try:
            optimization_settings = {
                'caching_enabled': enable_caching,
                'cleanup_frequency': cleanup_frequency,
                'memory_optimizations': [],
                'performance_mode': 'optimized'
            }
            
            # Enable attention caching
            if enable_caching:
                if not hasattr(self, 'attention_cache'):
                    self.attention_cache = {}
                optimization_settings['memory_optimizations'].append('attention_caching')
            
            # Set cleanup frequency
            self.cleanup_frequency = cleanup_frequency
            optimization_settings['memory_optimizations'].append('periodic_cleanup')
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                optimization_settings['memory_optimizations'].append('gradient_checkpointing')
            
            # Set optimal batch sizes based on system memory
            if self.low_memory:
                optimization_settings['recommended_batch_size'] = 1
                optimization_settings['max_concurrent_analyses'] = 1
            else:
                optimization_settings['recommended_batch_size'] = 4
                optimization_settings['max_concurrent_analyses'] = 2
            
            self.logger.info("Performance optimization applied")
            self.logger.info(f"Optimizations enabled: {optimization_settings['memory_optimizations']}")
            
            return optimization_settings
            
        except Exception as e:
            self.logger.error(f"Performance optimization failed: {e}")
            return {'error': str(e)}
    
    def test_system_robustness(self, num_iterations: int = 5) -> Dict:
        """
        Test system robustness with multiple runs and edge cases.
        
        Args:
            num_iterations: Number of test iterations
            
        Returns:
            Robustness test results
        """
        try:
            self.logger.info(f"Testing system robustness with {num_iterations} iterations")
            
            robustness_results = {
                'iterations_completed': 0,
                'consistency_check': {},
                'error_handling_test': {},
                'performance_stability': [],
                'edge_case_results': {}
            }
            
            test_image = "test_data/normal_chest.png"
            
            # Test 1: Consistency across multiple runs
            self.logger.info("Testing consistency across multiple runs...")
            consistency_results = []
            
            for i in range(num_iterations):
                try:
                    result = self.generate_comprehensive_analysis(test_image)
                    if result['success']:
                        # Extract key metrics for consistency check
                        metrics = {
                            'top_condition': max(result['model_predictions'].items(), 
                                               key=lambda x: x[1]['probability'])[0],
                            'num_keywords': result['keyword_summary']['total_keywords'],
                            'attention_regions': len(result['attention_analysis']['significant_regions']),
                            'processing_time': result['performance_metrics']['total_processing_time']
                        }
                        consistency_results.append(metrics)
                        robustness_results['iterations_completed'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Iteration {i+1} failed: {e}")
            
            # Analyze consistency
            if consistency_results:
                top_conditions = [r['top_condition'] for r in consistency_results]
                consistency_rate = max(set(top_conditions), key=top_conditions.count)
                robustness_results['consistency_check'] = {
                    'consistent_predictions': top_conditions.count(consistency_rate) / len(top_conditions),
                    'average_keywords': sum(r['num_keywords'] for r in consistency_results) / len(consistency_results),
                    'average_processing_time': sum(r['processing_time'] for r in consistency_results) / len(consistency_results),
                    'most_common_prediction': consistency_rate
                }
            
            # Test 2: Error handling
            self.logger.info("Testing error handling...")
            error_test_cases = [
                ("nonexistent_image.png", "File not found"),
                ("", "Empty path"),
                (None, "None input")
            ]
            
            error_handling_results = {}
            for test_input, expected_error in error_test_cases:
                try:
                    result = self.generate_comprehensive_analysis(test_input)
                    error_handling_results[expected_error] = {
                        'handled_gracefully': not result['success'],
                        'error_logged': len(result.get('error_log', [])) > 0
                    }
                except Exception as e:
                    error_handling_results[expected_error] = {
                        'handled_gracefully': False,
                        'exception_raised': str(e)
                    }
            
            robustness_results['error_handling_test'] = error_handling_results
            
            # Test 3: Edge cases
            self.logger.info("Testing edge cases...")
            edge_cases = {}
            
            # Test with different templates
            for template in ['standard', 'detailed', 'concise']:
                try:
                    result = self.generate_comprehensive_analysis(test_image, template=template)
                    edge_cases[f'template_{template}'] = result['success']
                except Exception as e:
                    edge_cases[f'template_{template}'] = False
            
            robustness_results['edge_case_results'] = edge_cases
            
            # Calculate overall robustness score
            total_tests = (num_iterations + len(error_test_cases) + len(edge_cases))
            successful_tests = (
                robustness_results['iterations_completed'] + 
                sum(1 for r in error_handling_results.values() if r.get('handled_gracefully', False)) +
                sum(1 for success in edge_cases.values() if success)
            )
            
            robustness_results['overall_robustness_score'] = successful_tests / total_tests * 100
            
            self.logger.info(f"Robustness testing completed. Score: {robustness_results['overall_robustness_score']:.1f}%")
            return robustness_results
            
        except Exception as e:
            self.logger.error(f"Robustness testing failed: {e}")
            return {'error': str(e), 'iterations_completed': 0}
    
    def get_system_status(self) -> Dict:
        """
        Get comprehensive system status and health check.
        
        Returns:
            System status dictionary
        """
        try:
            status = {
                'timestamp': str(datetime.now()),
                'system_health': {},
                'component_status': {},
                'performance_info': {},
                'configuration': {}
            }
            
            # System health
            status['system_health'] = {
                'model_loaded': self.model is not None,
                'device_available': torch.cuda.is_available() if self.device == 'cuda' else True,
                'memory_mode': 'low_memory' if self.low_memory else 'standard',
                'llm_connection': self.test_llm_connection()
            }
            
            # Component status
            status['component_status'] = {
                'image_preprocessing': True,  # Always available
                'model_inference': self.model is not None,
                'keyword_extraction': len(self.condition_keywords) > 0,
                'attention_visualization': hasattr(self, 'anatomical_regions'),
                'llm_integration': hasattr(self, 'llm_config'),
                'batch_processing': True  # Always available
            }
            
            # Performance info
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                status['performance_info']['gpu_memory_gb'] = gpu_memory
            
            system_memory = psutil.virtual_memory().total / (1024**3)
            status['performance_info']['system_memory_gb'] = system_memory
            status['performance_info']['device'] = str(self.device)
            
            # Configuration
            status['configuration'] = {
                'conditions_supported': len(CONDITIONS),
                'keyword_mappings': len(self.condition_keywords),
                'anatomical_regions': len(self.anatomical_regions),
                'report_templates': list(self.report_templates.keys()),
                'llm_model': self.llm_config['model_name'],
                'attention_caching': hasattr(self, 'attention_cache')
            }
            
            # Overall health score - properly count boolean values
            health_checks = list(status['system_health'].values()) + list(status['component_status'].values())
            health_count = sum(1 for check in health_checks if check is True)
            health_score = (health_count / len(health_checks)) * 100
            status['overall_health_score'] = health_score
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                'error': str(e),
                'timestamp': str(datetime.now()),
                'overall_health_score': 0.0
            }
