"""
Inference Predictor for Hybrid Disease + Keyword Prediction
==========================================================

Main inference interface for making predictions on single images.
Loads trained hybrid model and returns both disease and keyword predictions.
"""

import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

# Import model components
import sys
sys.path.append('..')
from models.hybrid_model import HybridMedicalModel
from config.model_config import *
from config.data_config import *
from inference.keyword_decoder import KeywordDecoder

class Predictor:
    """
    Main inference interface for hybrid disease + keyword prediction.
    
    Handles:
    - Loading trained model
    - Processing single image inputs
    - Decoding predictions to human-readable format
    - Returning structured predictions
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 vocab_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize predictor with trained model and vocabulary.
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary JSON file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Setup paths
        if model_path is None:
            model_path = "output/models/hybrid_model_final.safetensors"
        if vocab_path is None:
            vocab_path = "output/vocabulary/simple_vocab.json"
            
        self.model_path = model_path
        self.vocab_path = vocab_path
        
        # Initialize components
        self.model = None
        self.vocabulary = None
        self.keyword_decoder = None
        
        # Disease labels from config
        self.disease_labels = DISEASE_CONFIG["conditions"]
        
        # Load components
        self._load_vocabulary()
        self._load_model()
        self._setup_keyword_decoder()
        
        logging.info(f"Predictor initialized on device: {self.device}")
        
    def _load_vocabulary(self):
        """Load vocabulary for keyword decoding."""
        try:
            with open(self.vocab_path, 'r') as f:
                self.vocabulary = json.load(f)
            logging.info(f"Loaded vocabulary with {len(self.vocabulary['word2idx'])} words")
        except FileNotFoundError:
            logging.warning(f"Vocabulary file not found at {self.vocab_path}")
            # Create dummy vocabulary for testing
            self.vocabulary = {
                "word2idx": {"<PAD>": 0, "<UNK>": 1},
                "idx2word": {"0": "<PAD>", "1": "<UNK>"},
                "word_frequencies": {},
                "category_frequencies": {}
            }
            
    def _load_model(self):
        """Load trained hybrid model."""
        try:
            # Initialize model architecture
            self.model = HybridMedicalModel(
                num_conditions=DISEASE_CONFIG["num_conditions"],
                vocab_size=len(self.vocabulary["word2idx"]),
                num_regions=REGION_CONFIG["num_regions"],
                hidden_dim=FUSION_CONFIG["fusion_hidden_dim"]
            )
            
            # Load checkpoint if it exists
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logging.info(f"Loaded model from {self.model_path}")
            else:
                logging.warning(f"Model checkpoint not found at {self.model_path}")
                logging.info("Using randomly initialized model for testing")
                
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
            
    def _setup_keyword_decoder(self):
        """Setup keyword decoder for converting logits to text."""
        self.keyword_decoder = KeywordDecoder(
            vocabulary=self.vocabulary,
            device=self.device
        )
        
    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess single image for inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image tensor [1, 3, 224, 224]
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Resize to model input size
            image = image.resize((IMAGE_CONFIG["image_size"], IMAGE_CONFIG["image_size"]))
            
            # Convert to tensor and normalize
            image_tensor = torch.tensor(np.array(image), dtype=torch.float32)
            image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # [3, 224, 224]
            
            # Normalize using config values
            mean = torch.tensor(IMAGE_CONFIG["mean"]).view(3, 1, 1)
            std = torch.tensor(IMAGE_CONFIG["std"]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            # Add batch dimension
            return image_tensor.unsqueeze(0)  # [1, 3, 224, 224]
            
        except Exception as e:
            logging.error(f"Error preprocessing image {image_path}: {e}")
            raise
            
    def _create_dummy_inputs(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """
        Create dummy inputs for testing when real data is not available.
        
        Args:
            batch_size: Batch size for dummy inputs
            
        Returns:
            Dictionary of dummy input tensors
        """
        return {
            'img': torch.randn(batch_size, 3, 224, 224, device=self.device),
            'bbox': torch.randn(batch_size, 1, 224, 224, device=self.device),
            'fix_seq': torch.randn(batch_size, 64, 4, device=self.device),
            'fix_mask': torch.ones(batch_size, 64, device=self.device, dtype=torch.bool),
            'transcript': ["dummy medical transcript text"] * batch_size,
            'region_mask': torch.ones(batch_size, 17, device=self.device, dtype=torch.bool)
        }
        
    def predict(self, 
                image_path: str,
                bbox_path: Optional[str] = None,
                fixation_path: Optional[str] = None,
                transcript: Optional[str] = None,
                top_k_keywords: int = 10) -> Dict:
        """
        Make predictions on a single image.
        
        Args:
            image_path: Path to input image
            bbox_path: Path to bbox mask (optional)
            fixation_path: Path to fixation sequence (optional)
            transcript: Transcript text (optional)
            top_k_keywords: Number of top keywords to return
            
        Returns:
            Dictionary with disease and keyword predictions
        """
        try:
            with torch.no_grad():
                # Preprocess image
                image_tensor = self._preprocess_image(image_path).to(self.device)
                
                # For now, create dummy inputs for missing modalities
                # In full implementation, these would be properly loaded
                batch_inputs = {
                    'img': image_tensor,
                    'bbox': torch.zeros(1, 1, 224, 224, device=self.device),
                    'fix_seq': torch.zeros(1, 64, 4, device=self.device),
                    'fix_mask': torch.ones(1, 64, device=self.device, dtype=torch.bool),
                    'transcript': [transcript or "normal chest x-ray"],
                    'region_mask': torch.ones(1, 17, device=self.device, dtype=torch.bool)
                }
                
                # Forward pass through model
                outputs = self.model(
                    img=batch_inputs['img'],
                    bbox=batch_inputs['bbox'],
                    fix_seq=batch_inputs['fix_seq'],
                    fix_mask=batch_inputs['fix_mask'],
                    transcript=batch_inputs['transcript'],
                    region_mask=batch_inputs['region_mask']
                )
                
                # Extract predictions
                disease_logits = outputs['disease_logits'][0]  # [8]
                global_keyword_logits = outputs['global_logits'][0]  # [vocab_size]
                region_keyword_logits = outputs['region_logits'][0]  # [17, vocab_size]
                
                # Convert to predictions
                predictions = self._decode_predictions(
                    disease_logits=disease_logits,
                    global_keyword_logits=global_keyword_logits,
                    region_keyword_logits=region_keyword_logits,
                    top_k_keywords=top_k_keywords
                )
                
                return predictions
                
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise
            
    def _decode_predictions(self,
                          disease_logits: torch.Tensor,
                          global_keyword_logits: torch.Tensor,
                          region_keyword_logits: torch.Tensor,
                          top_k_keywords: int = 10) -> Dict:
        """
        Decode model outputs to human-readable predictions.
        
        Args:
            disease_logits: Disease prediction logits [8]
            global_keyword_logits: Global keyword logits [vocab_size]
            region_keyword_logits: Region keyword logits [17, vocab_size]
            top_k_keywords: Number of top keywords to return
            
        Returns:
            Structured prediction dictionary
        """
        # Disease predictions
        disease_probs = torch.sigmoid(disease_logits).cpu().numpy()
        disease_predictions = []
        for i, (label, prob) in enumerate(zip(self.disease_labels, disease_probs)):
            disease_predictions.append({
                "condition": label,
                "probability": float(prob),
                "predicted": bool(prob > 0.5)
            })
            
        # Global keyword predictions
        global_keywords = self.keyword_decoder.decode_global_keywords(
            global_keyword_logits, top_k=top_k_keywords
        )
        
        # Region keyword predictions
        region_keywords = self.keyword_decoder.decode_region_keywords(
            region_keyword_logits, top_k=top_k_keywords
        )
        
        return {
            "diseases": disease_predictions,
            "global_keywords": global_keywords,
            "region_keywords": region_keywords,
            "summary": {
                "predicted_diseases": [d["condition"] for d in disease_predictions if d["predicted"]],
                "top_keywords": [kw["keyword"] for kw in global_keywords[:5]],
                "confidence_score": float(np.mean([d["probability"] for d in disease_predictions]))
            }
        }
        
    def predict_batch(self, 
                     image_paths: List[str],
                     **kwargs) -> List[Dict]:
        """
        Make predictions on a batch of images.
        
        Args:
            image_paths: List of image paths
            **kwargs: Additional arguments passed to predict()
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        for image_path in image_paths:
            try:
                pred = self.predict(image_path, **kwargs)
                predictions.append(pred)
            except Exception as e:
                logging.error(f"Error predicting {image_path}: {e}")
                predictions.append({"error": str(e)})
                
        return predictions
        
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"error": "Model not loaded"}
            
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_architecture": "HybridMedicalModel",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "vocabulary_size": len(self.vocabulary["word2idx"]) if self.vocabulary else 0,
            "disease_conditions": self.disease_labels,
            "device": str(self.device),
            "model_path": self.model_path,
            "vocab_path": self.vocab_path
        } 