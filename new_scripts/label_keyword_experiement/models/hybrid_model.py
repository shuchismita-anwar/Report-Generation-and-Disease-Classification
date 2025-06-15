"""
Enhanced Hybrid Model for Disease Classification + Keyword Prediction
===================================================================

This module extends the existing MultiModalMIMICModel to support both:
1. Disease classification (8 conditions) - existing functionality
2. Medical keyword prediction (global + region-specific) - new functionality

The model integrates:
- Existing multimodal architecture (ViT + BBox + Fixation + Text)
- New keyword prediction heads (GlobalKeywordHead + RegionKeywordHead)
- Multi-task loss function combining disease + keyword objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModel

# Import our custom components
from .keyword_heads import GlobalKeywordHead, RegionKeywordHead, CombinedKeywordHead
from .loss_functions import MultiTaskLoss

logger = logging.getLogger(__name__)

class HybridMedicalModel(nn.Module):
    """
    Enhanced multimodal model for disease classification + keyword prediction.
    
    Extends the original MultiModalMIMICModel with keyword prediction capabilities
    while maintaining backward compatibility for disease classification.
    """
    
    def __init__(
        self,
        num_conditions: int = 8,
        vocab_size: int = 35,  # From our vocabulary builder
        num_regions: int = 17,
        hidden_dim: int = 768,
        contrastive_temperature: float = 0.1,
        loss_type: str = "focal",
        low_memory: bool = False,
        enable_keywords: bool = True,
        vit_model_name: str = "google/vit-base-patch16-224-in21k",
        text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    ):
        super().__init__()
        
        self.num_conditions = num_conditions
        self.vocab_size = vocab_size
        self.num_regions = num_regions
        self.hidden_dim = hidden_dim if not low_memory else 512
        self.contrastive_temperature = contrastive_temperature
        self.loss_type = loss_type
        self.low_memory = low_memory
        self.enable_keywords = enable_keywords
        
        logger.info(f"Initializing HybridMedicalModel:")
        logger.info(f"  - Conditions: {num_conditions}")
        logger.info(f"  - Vocabulary size: {vocab_size}")
        logger.info(f"  - Regions: {num_regions}")
        logger.info(f"  - Hidden dim: {self.hidden_dim}")
        logger.info(f"  - Keywords enabled: {enable_keywords}")
        logger.info(f"  - Low memory mode: {low_memory}")
        
        # Add compatibility with Transformers trainer
        self._keys_to_ignore_on_save = None
        
        # Initialize the core multimodal components
        self._init_multimodal_components(vit_model_name, text_model_name)
        
        # Initialize keyword prediction components (if enabled)
        if self.enable_keywords:
            self._init_keyword_components()
        
        # Initialize loss functions
        self._init_loss_functions()
        
        # Better weight initialization
        self._initialize_weights()
        
        logger.info("HybridMedicalModel initialization complete")
    
    def _init_multimodal_components(self, vit_model_name: str, text_model_name: str):
        """Initialize the core multimodal architecture components"""
        
        # --- Image encoder (ViT) ---
        self.vit = AutoModel.from_pretrained(vit_model_name)
        img_feature_dim = self.vit.config.hidden_size
        
        # Image projection
        if self.low_memory:
            self.img_proj = nn.Sequential(
                nn.Linear(img_feature_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 512)
            )
        else:
            self.img_proj = nn.Sequential(
                nn.Linear(img_feature_dim, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.Dropout(0.1)
            )
        
        # --- BBox encoder ---
        if self.low_memory:
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
        
        # --- Fixation encoder ---
        if self.low_memory:
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
        
        # --- Text encoder ---
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_feature_dim = self.text_encoder.config.hidden_size
        
        if self.low_memory:
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
        
        # --- Fusion layers ---
        if self.low_memory:
            self.fusion = nn.Sequential(
                nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
                nn.LayerNorm(self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            )
            self.fusion_attention = None
        else:
            # Attention-based fusion
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
        
        # --- Disease classification heads ---
        if self.low_memory:
            self.global_classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_conditions)
            )
            self.condition_specific_heads = None
        else:
            # Enhanced classifier with multiple heads
            self.condition_specific_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(768, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 1)
                ) for _ in range(self.num_conditions)
            ])
            
            self.global_classifier = nn.Sequential(
                nn.Linear(768, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_conditions)
            )
    
    def _init_keyword_components(self):
        """Initialize keyword prediction components"""
        if not self.enable_keywords:
            return
        
        logger.info("Initializing keyword prediction components...")
        
        # Global keyword prediction head
        self.global_keyword_head = GlobalKeywordHead(
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size,
            dropout=0.15 if not self.low_memory else 0.1
        )
        
        # Region-specific keyword prediction head
        self.region_keyword_head = RegionKeywordHead(
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size,
            num_regions=self.num_regions,
            dropout=0.15 if not self.low_memory else 0.1
        )
        
        # Combined keyword head for coordination
        self.combined_keyword_head = CombinedKeywordHead(
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size,
            num_regions=self.num_regions,
            dropout=0.15 if not self.low_memory else 0.1
        )
        
        logger.info(f"Keyword components initialized:")
        logger.info(f"  - Global head: {self.hidden_dim} -> {self.vocab_size}")
        logger.info(f"  - Region head: {self.hidden_dim} -> {self.num_regions} x {self.vocab_size}")
    
    def _init_loss_functions(self):
        """Initialize loss functions"""
        # Disease classification loss (existing)
        self.disease_loss_fn = None  # Will be set via set_loss_function
        
        # Multi-task loss (new)
        if self.enable_keywords:
            self.multi_task_loss = MultiTaskLoss(
                loss_weights={
                    'disease': 1.0,
                    'keyword_global': 0.5,
                    'keyword_region': 0.3,
                    'consistency': 0.2
                }
            )
            logger.info("Multi-task loss initialized with keyword prediction")
        else:
            self.multi_task_loss = None
            logger.info("Single-task mode: disease classification only")
    
    def _initialize_weights(self):
        """Better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def set_loss_function(self, pos_weights, loss_type: str = "focal"):
        """Set disease classification loss function"""
        from .loss_functions import FocalLoss
        
        if loss_type == "focal":
            self.disease_loss_fn = FocalLoss(
                alpha=1.10, 
                gamma=2.15, 
                pos_weight=pos_weights, 
                label_smoothing=0.08
            )
        else:
            self.disease_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
        logger.info(f"Disease loss function set: {loss_type}")
    
    def load_chexpert_weights(self, chexpert_model_path: str):
        """Load pre-trained weights from CheXpert model"""
        import os
        import safetensors
        
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
                elif key.startswith("img_proj.") and not self.low_memory:
                    filtered_state_dict[key] = value
            
            # Load the filtered weights
            missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
            logger.info(f"Loaded CheXpert weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            
        except Exception as e:
            logger.error(f"Error loading CheXpert weights: {e}")
            logger.warning("Continuing without pre-trained weights")
    
    def info_nce_loss(self, features1, features2, temperature=None, eps=1e-6):
        """Calculates the InfoNCE loss for contrastive learning"""
        temp = temperature if temperature is not None else self.contrastive_temperature
        
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features1, features2.T) / temp
        
        # Create labels (diagonal should be positive pairs)
        batch_size = features1.size(0)
        labels = torch.arange(batch_size, device=features1.device)
        
        # Compute InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
    
    def forward(
        self, 
        img: torch.Tensor,
        bbox: torch.Tensor,
        fix_seq: torch.Tensor,
        fix_mask: torch.Tensor,
        transcript: List[str],
        labels: Optional[torch.Tensor] = None,
        global_keywords: Optional[torch.Tensor] = None,
        region_keywords: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
        contrastive_weight: float = 0.1
    ) -> Dict[str, Any]:
        """
        Forward pass with support for both disease classification and keyword prediction
        
        Args:
            img: Image tensor [batch_size, 3, 224, 224]
            bbox: BBox mask tensor [batch_size, 1, 224, 224]
            fix_seq: Fixation sequence [batch_size, seq_len, 4]
            fix_mask: Fixation mask [batch_size, seq_len]
            transcript: List of transcript strings
            labels: Disease labels [batch_size, num_conditions] (optional)
            global_keywords: Global keyword labels [batch_size, vocab_size] (optional)
            region_keywords: Region keyword labels [batch_size, num_regions, vocab_size] (optional)
            region_mask: Region availability mask [batch_size, num_regions] (optional)
            contrastive_weight: Weight for contrastive loss
            
        Returns:
            Dictionary containing logits, losses, and other outputs
        """
        batch_size = img.size(0)
        device = img.device
        
        # Memory management
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # --- Encode all modalities ---
        
        # Image encoding
        img_outputs = self.vit(pixel_values=img)
        if hasattr(img_outputs, 'pooler_output') and img_outputs.pooler_output is not None:
            img_feat = img_outputs.pooler_output
        else:
            img_feat = img_outputs.last_hidden_state[:, 0, :]  # CLS token
        img_proj = self.img_proj(img_feat)
        
        # Clean up
        del img_outputs, img_feat
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # BBox encoding
        bbox_feat = self.bbox_encoder(bbox).flatten(1)
        bbox_proj = self.bbox_proj(bbox_feat)
        del bbox_feat
        
        # Fixation encoding
        fix_lens = fix_mask.sum(dim=1).cpu()
        
        if torch.all(fix_lens == 0):
            if self.low_memory:
                fix_feat_gru = torch.zeros(batch_size, 256, device=device)
            else:
                fix_feat_gru = torch.zeros(batch_size, 1536, device=device)
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
            
            del fix_emb, packed_seq, h_n
        
        fix_proj = self.fix_proj(fix_feat_gru)
        del fix_feat_gru
        
        # Text encoding
        if isinstance(transcript, list):
            try:
                max_length = 256 if self.low_memory else 512
                transcript_tokens = self.tokenizer(
                    transcript, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length
                ).to(device)
                
                text_outputs = self.text_encoder(**transcript_tokens)
                text_feat = text_outputs.last_hidden_state[:, 0, :]  # CLS token
                text_proj = self.text_proj(text_feat)
                
                del transcript_tokens, text_outputs, text_feat
                
            except Exception as e:
                logger.warning(f"Error processing transcript: {e}")
                text_proj = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            text_proj = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Memory cleanup before fusion
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # --- Multimodal fusion ---
        combined = torch.cat([img_proj, bbox_proj, fix_proj, text_proj], dim=1)
        fused = self.fusion(combined)
        del combined
        
        if not self.low_memory and self.fusion_attention is not None:
            # Additional attention refinement
            stacked_features = torch.stack([img_proj, bbox_proj, fix_proj, text_proj], dim=1)
            attended_features, attn_weights = self.fusion_attention(stacked_features, stacked_features, stacked_features)
            attended_combined = attended_features.mean(dim=1)
            final_fused = self.fusion_norm(fused + attended_combined)
            
            del stacked_features, attended_features, attn_weights, attended_combined
        else:
            final_fused = fused
        
        del fused
        
        # --- Disease classification ---
        global_logits = self.global_classifier(final_fused)
        
        if not self.low_memory and self.condition_specific_heads is not None:
            condition_logits = []
            for head in self.condition_specific_heads:
                cond_logit = head(final_fused)
                condition_logits.append(cond_logit)
            
            condition_logits = torch.cat(condition_logits, dim=1)
            
            # Ensemble predictions
            alpha = 0.7
            disease_logits = alpha * global_logits + (1 - alpha) * condition_logits
            del condition_logits
        else:
            disease_logits = global_logits
        
        # --- Keyword prediction (if enabled) ---
        keyword_outputs = {}
        if self.enable_keywords:
            keyword_results = self.combined_keyword_head(
                features=final_fused,
                region_mask=region_mask
            )
            keyword_outputs.update(keyword_results)
        
        # --- Loss computation ---
        total_loss = None
        loss_components = {}
        
        if labels is not None:
            if self.enable_keywords and (global_keywords is not None or region_keywords is not None):
                # Multi-task loss
                predictions_dict = {
                    'disease_logits': disease_logits,
                    'global_keyword_logits': keyword_outputs.get('global_logits'),
                    'region_keyword_logits': keyword_outputs.get('region_logits')
                }
                
                targets_dict = {
                    'disease_labels': labels,
                    'global_keyword_labels': global_keywords,
                    'region_keyword_labels': region_keywords,
                    'region_mask': region_mask
                }
                
                loss_components = self.multi_task_loss(
                    predictions=predictions_dict,
                    targets=targets_dict,
                    return_individual_losses=True
                )
                total_loss = loss_components['total']
            else:
                # Disease-only loss
                if self.disease_loss_fn is None:
                    self.disease_loss_fn = nn.BCEWithLogitsLoss()
                    logger.warning("Using fallback BCEWithLogitsLoss!")
                
                disease_loss = self.disease_loss_fn(disease_logits, labels)
                total_loss = disease_loss
                loss_components['disease'] = disease_loss
            
            # Add contrastive loss if applicable
            if img_proj.shape[0] > 1 and not self.low_memory:
                loss_img_fix = self.info_nce_loss(img_proj, fix_proj)
                loss_img_text = self.info_nce_loss(img_proj, text_proj)
                contrastive_loss = (loss_img_fix + loss_img_text) / 2.0
                
                if total_loss is not None:
                    total_loss = total_loss + contrastive_weight * contrastive_loss
                loss_components['contrastive'] = contrastive_loss
        
        # Clean up projection tensors
        del img_proj, bbox_proj, fix_proj, text_proj, final_fused
        
        # Final memory cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Prepare output
        output = {
            'disease_logits': disease_logits,
            'attn_map': torch.zeros_like(bbox),  # Placeholder
        }
        
        # Add keyword outputs if enabled
        if self.enable_keywords:
            output.update(keyword_outputs)
        
        # Add loss information
        if total_loss is not None:
            output['loss'] = total_loss
            output['loss_components'] = loss_components
        
        return output
    
    def predict_keywords(
        self,
        features: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Predict keywords from fused features
        
        Args:
            features: Fused multimodal features [batch_size, hidden_dim]
            region_mask: Region availability mask [batch_size, num_regions]
            top_k: Number of top keywords to return
            threshold: Threshold for keyword prediction
            
        Returns:
            Dictionary with keyword predictions
        """
        if not self.enable_keywords:
            logger.warning("Keywords not enabled in this model")
            return {}
        
        return self.combined_keyword_head.predict_keywords(
            features=features,
            region_mask=region_mask,
            top_k=top_k,
            threshold=threshold
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information"""
        return {
            'model_type': 'HybridMedicalModel',
            'num_conditions': self.num_conditions,
            'vocab_size': self.vocab_size,
            'num_regions': self.num_regions,
            'hidden_dim': self.hidden_dim,
            'enable_keywords': self.enable_keywords,
            'low_memory': self.low_memory,
            'loss_type': self.loss_type,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }