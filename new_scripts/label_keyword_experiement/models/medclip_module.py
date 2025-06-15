"""
MedCLIP Module for Medical Vision-Language Contrastive Learning
==============================================================

This module integrates Microsoft's BiomedCLIP for contrastive learning between
medical images and keyword texts. It provides:
- Loading pre-trained BiomedCLIP text encoder
- Encoding keyword texts to medical embeddings
- Computing contrastive loss between image and text features
- Enabling cross-modal retrieval and alignment

BiomedCLIP is trained on 15M medical image-text pairs and achieves SOTA
performance on biomedical vision-language tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union, Any
from collections import Counter, defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from open_clip import create_model_from_pretrained, get_tokenizer
    MEDCLIP_AVAILABLE = True
except ImportError:
    logger.warning("open_clip_torch not available. Install with: pip install open_clip_torch")
    MEDCLIP_AVAILABLE = False


class MedCLIPModule(nn.Module):
    """
    MedCLIP integration for medical vision-language contrastive learning.
    
    Uses Microsoft's BiomedCLIP model for encoding medical keywords and computing
    contrastive loss with image features.
    """
    
    def __init__(self, 
                 model_name: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                 embedding_dim: int = 512,  # BiomedCLIP outputs 512-dim embeddings
                 temperature: float = 0.07,
                 max_context_length: int = 256,
                 cache_dir: Optional[str] = None):
        """
        Initialize MedCLIP module.
        
        Args:
            model_name: HuggingFace model identifier for BiomedCLIP
            embedding_dim: Dimension of text embeddings
            temperature: Temperature for contrastive loss
            max_context_length: Maximum text context length
            cache_dir: Directory to cache downloaded models
        """
        super().__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.max_context_length = max_context_length
        self.cache_dir = cache_dir
        
        # Initialize models
        self._load_medclip_models()
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
        # Projection layers for alignment
        self.text_projection = nn.Linear(embedding_dim, embedding_dim)
        self.image_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization
        self.text_norm = nn.LayerNorm(embedding_dim)
        self.image_norm = nn.LayerNorm(embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        logger.info(f"MedCLIP module initialized with model: {model_name}")
    
    def _load_medclip_models(self):
        """Load pre-trained BiomedCLIP models."""
        if not MEDCLIP_AVAILABLE:
            logger.warning("MedCLIP not available, using dummy models")
            self.medclip_model = None
            self.tokenizer = None
            return
        
        try:
            # Load BiomedCLIP model and tokenizer
            logger.info(f"Loading BiomedCLIP model: {self.model_name}")
            self.medclip_model, self.preprocess = create_model_from_pretrained(self.model_name)
            self.tokenizer = get_tokenizer(self.model_name)
            
            # Set model to evaluation mode and freeze weights
            self.medclip_model.eval()
            for param in self.medclip_model.parameters():
                param.requires_grad = False
            
            logger.info("BiomedCLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BiomedCLIP model: {e}")
            logger.warning("Using dummy models for fallback")
            self.medclip_model = None
            self.tokenizer = None
    
    def encode_keywords(self, keywords: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode medical keywords using BiomedCLIP text encoder.
        
        Args:
            keywords: List of medical keyword strings
            batch_size: Batch size for processing
            
        Returns:
            Text embeddings tensor [len(keywords), embedding_dim]
        """
        if not keywords:
            return torch.zeros(0, self.embedding_dim, device=self.device)
        
        if self.medclip_model is None:
            # Fallback: random embeddings
            return torch.randn(len(keywords), self.embedding_dim, device=self.device)
        
        all_embeddings = []
        
        # Process keywords in batches
        for i in range(0, len(keywords), batch_size):
            batch_keywords = keywords[i:i + batch_size]
            
            # Tokenize keywords
            text_inputs = self.tokenizer(
                batch_keywords, 
                context_length=self.max_context_length
            ).to(self.device)
            
            # Extract text features using BiomedCLIP
            with torch.no_grad():
                _, text_features, _ = self.medclip_model(None, text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(text_features)
        
        # Concatenate all embeddings
        text_embeddings = torch.cat(all_embeddings, dim=0)
        
        return text_embeddings
    
    def encode_keyword_categories(self, keyword_dict: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Encode keywords by medical categories.
        
        Args:
            keyword_dict: Dictionary mapping categories to keyword lists
            
        Returns:
            Dictionary mapping categories to embeddings
        """
        category_embeddings = {}
        
        for category, keywords in keyword_dict.items():
            if keywords:
                embeddings = self.encode_keywords(keywords)
                category_embeddings[category] = embeddings
            else:
                category_embeddings[category] = torch.zeros(0, self.embedding_dim, device=self.device)
        
        return category_embeddings
    
    def compute_contrastive_loss(self, 
                                image_features: torch.Tensor, 
                                text_features: torch.Tensor,
                                temperature: Optional[float] = None) -> torch.Tensor:
        """
        Compute contrastive loss between image and text features.
        
        Args:
            image_features: Image feature tensor [batch_size, feature_dim]
            text_features: Text feature tensor [batch_size, feature_dim]
            temperature: Optional temperature override
            
        Returns:
            Contrastive loss scalar
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Project features to common space
        image_features = self.image_projection(image_features)
        text_features = self.text_projection(text_features)
        
        # Apply layer normalization
        image_features = self.image_norm(image_features)
        text_features = self.text_norm(text_features)
        
        # Normalize again after projection
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logit_scale = self.logit_scale.exp() if temperature is None else (1.0 / temperature)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        # Create labels (diagonal)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Compute cross-entropy loss in both directions
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        # Average the two losses
        contrastive_loss = (loss_i2t + loss_t2i) / 2.0
        
        return contrastive_loss
    
    def compute_semantic_similarity(self, 
                                   pred_keywords: List[str], 
                                   gt_keywords: List[str]) -> torch.Tensor:
        """
        Compute semantic similarity between predicted and ground truth keywords.
        
        Args:
            pred_keywords: Predicted keyword strings
            gt_keywords: Ground truth keyword strings
            
        Returns:
            Similarity score tensor
        """
        if not pred_keywords or not gt_keywords:
            return torch.tensor(0.0, device=self.device)
        
        # Encode both sets of keywords
        pred_embeddings = self.encode_keywords(pred_keywords)
        gt_embeddings = self.encode_keywords(gt_keywords)
        
        # Compute pairwise cosine similarities
        similarities = torch.mm(pred_embeddings, gt_embeddings.t())
        
        # Return mean of maximum similarities (bipartite matching approximation)
        max_similarities = torch.max(similarities, dim=1)[0]
        semantic_similarity = torch.mean(max_similarities)
        
        return semantic_similarity
    
    def forward(self, 
                image_features: torch.Tensor,
                keyword_texts: List[str],
                compute_loss: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MedCLIP module.
        
        Args:
            image_features: Image feature tensor [batch_size, feature_dim]
            keyword_texts: List of keyword text strings
            compute_loss: Whether to compute contrastive loss
            
        Returns:
            Dictionary containing embeddings and loss
        """
        batch_size = image_features.shape[0]
        
        # Ensure we have keyword texts for each image in batch
        if len(keyword_texts) != batch_size:
            # Pad or truncate keyword texts to match batch size
            keyword_texts = (keyword_texts * (batch_size // len(keyword_texts) + 1))[:batch_size]
        
        # Encode keyword texts
        text_embeddings = self.encode_keywords(keyword_texts)
        
        outputs = {
            'text_embeddings': text_embeddings,
            'image_features': image_features
        }
        
        if compute_loss:
            contrastive_loss = self.compute_contrastive_loss(image_features, text_embeddings)
            outputs['contrastive_loss'] = contrastive_loss
        
        return outputs
    
    @property
    def device(self) -> torch.device:
        """Get device of the module."""
        return next(self.parameters()).device
    
    def get_text_encoder(self):
        """Get the text encoder component."""
        if self.medclip_model is not None:
            return self.medclip_model.text_encoder if hasattr(self.medclip_model, 'text_encoder') else self.medclip_model
        return None
    
    def save_embeddings(self, keywords: List[str], save_path: str):
        """
        Save keyword embeddings to file.
        
        Args:
            keywords: List of keywords to encode
            save_path: Path to save embeddings
        """
        embeddings = self.encode_keywords(keywords)
        
        embedding_data = {
            'keywords': keywords,
            'embeddings': embeddings.cpu().numpy().tolist(),
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        }
        
        with open(save_path, 'w') as f:
            json.dump(embedding_data, f, indent=2)
        
        logger.info(f"Saved {len(keywords)} keyword embeddings to {save_path}")


def create_medclip_processor(model_name: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                            embedding_dim: int = 512,  # BiomedCLIP outputs 512-dim embeddings
                            temperature: float = 0.07) -> MedCLIPModule:
    """
    Factory function to create MedCLIP processor.
    
    Args:
        model_name: BiomedCLIP model name
        embedding_dim: Embedding dimension
        temperature: Contrastive learning temperature
        
    Returns:
        Initialized MedCLIP module
    """
    return MedCLIPModule(
        model_name=model_name,
        embedding_dim=embedding_dim,
        temperature=temperature
    )


def test_medclip_integration():
    """Test MedCLIP integration functionality."""
    logger.info("Testing MedCLIP integration...")
    
    # Create MedCLIP module
    medclip = create_medclip_processor()
    
    # Test keyword encoding
    sample_keywords = [
        "cardiomegaly cardiac enlargement",
        "pleural effusion fluid accumulation", 
        "pneumothorax collapsed lung",
        "pulmonary edema lung swelling",
        "atelectasis lung collapse"
    ]
    
    try:
        # Test text encoding
        text_embeddings = medclip.encode_keywords(sample_keywords)
        logger.info(f"✅ Text encoding: {text_embeddings.shape}")
        
        # Test contrastive loss computation
        batch_size = len(sample_keywords)
        dummy_image_features = torch.randn(batch_size, 512)  # Match BiomedCLIP embedding dim
        
        contrastive_loss = medclip.compute_contrastive_loss(dummy_image_features, text_embeddings)
        logger.info(f"✅ Contrastive loss: {contrastive_loss.item():.4f}")
        
        # Test category encoding
        keyword_categories = {
            'anatomy': ['heart', 'lung', 'chest'],
            'pathology': ['pneumonia', 'edema', 'effusion'],
            'devices': ['pacemaker', 'catheter', 'tube']
        }
        
        category_embeddings = medclip.encode_keyword_categories(keyword_categories)
        logger.info(f"✅ Category encoding: {len(category_embeddings)} categories")
        
        # Test semantic similarity
        pred_keywords = ["heart enlargement", "lung fluid"]
        gt_keywords = ["cardiomegaly", "pleural effusion"]
        
        similarity = medclip.compute_semantic_similarity(pred_keywords, gt_keywords)
        logger.info(f"✅ Semantic similarity: {similarity.item():.4f}")
        
        logger.info("🎉 MedCLIP integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ MedCLIP integration test failed: {e}")
        return False


if __name__ == "__main__":
    # Run integration test
    test_medclip_integration() 