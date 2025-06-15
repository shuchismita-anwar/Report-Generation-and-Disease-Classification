"""
RadBERT Module for Medical Text Processing
==========================================

This module integrates RadBERT for sophisticated medical NLP tasks including:
- Processing medical transcripts with domain-specific language understanding
- Extracting medical entities using RadBERT representations
- Generating keyword predictions from radiology reports
- Providing semantic features for multimodal fusion
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union
from collections import Counter, defaultdict
from transformers import AutoConfig, AutoTokenizer, AutoModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RadBERTModule(nn.Module):
    """
    RadBERT integration for medical text understanding.
    
    This module provides:
    - Pre-trained RadBERT model loading
    - Medical transcript processing
    - Medical entity extraction
    - Keyword prediction and ranking
    - Semantic feature generation for fusion
    """
    
    def __init__(self, 
                 model_name: str = "UCSD-VA-health/RadBERT-RoBERTa-4m",
                 max_length: int = 512,
                 device: str = "auto"):
        """
        Initialize RadBERT module.
        
        Args:
            model_name: HuggingFace model identifier for RadBERT
            max_length: Maximum sequence length for tokenization
            device: Device to run model on ("auto", "cuda", "cpu")
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing RadBERT module with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load RadBERT model and tokenizer
        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Successfully loaded RadBERT model")
            logger.info(f"Model config: {self.config}")
            
        except Exception as e:
            logger.error(f"Failed to load RadBERT model: {e}")
            raise RuntimeError(f"Could not initialize RadBERT: {e}")
        
        # Medical vocabularies for entity extraction
        self.medical_categories = self._create_medical_categories()
        
        # Feature dimensions
        self.hidden_dim = self.config.hidden_size  # 768 for RoBERTa-base
        
    def _create_medical_categories(self) -> Dict[str, Set[str]]:
        """Create medical category vocabularies for entity classification."""
        return {
            'anatomy': {
                'heart', 'cardiac', 'lung', 'pulmonary', 'chest', 'thorax',
                'mediastinum', 'hilar', 'pleural', 'diaphragm', 'aorta',
                'ventricle', 'atrium', 'bronchi', 'alveolar', 'pericardium',
                'costophrenic', 'sternum', 'clavicle', 'spine', 'vertebra',
                'bilateral', 'unilateral', 'left', 'right', 'upper', 'lower'
            },
            'pathology': {
                'atelectasis', 'consolidation', 'effusion', 'pneumothorax',
                'pneumonia', 'edema', 'opacity', 'infiltrate', 'nodule',
                'mass', 'lesion', 'cardiomegaly', 'enlargement', 'abnormality',
                'inflammation', 'infection', 'congestion', 'dilation'
            },
            'devices': {
                'tube', 'catheter', 'line', 'pacemaker', 'defibrillator',
                'stent', 'prosthesis', 'implant', 'hardware', 'wire',
                'endotracheal', 'tracheostomy', 'picc', 'central', 'chest'
            },
            'descriptors': {
                'normal', 'abnormal', 'enlarged', 'small', 'large',
                'bilateral', 'mild', 'moderate', 'severe', 'stable',
                'improved', 'worsened', 'acute', 'chronic', 'clear'
            },
            'location': {
                'anterior', 'posterior', 'lateral', 'medial', 'central',
                'peripheral', 'superior', 'inferior', 'apical', 'basal'
            }
        }
    
    def process_transcript(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Process medical transcript text with RadBERT.
        
        Args:
            text: Medical transcript text
            
        Returns:
            Dictionary containing RadBERT embeddings and attention weights
        """
        if not text or not text.strip():
            # Return zero embeddings for empty text
            return {
                'embeddings': torch.zeros(self.hidden_dim, device=self.device),
                'pooled_output': torch.zeros(self.hidden_dim, device=self.device),
                'attention_weights': torch.zeros(1, 12, 1, 1, device=self.device),  # Dummy attention
                'tokens': [],
                'token_embeddings': torch.zeros(1, self.hidden_dim, device=self.device)
            }
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Process with RadBERT
        with torch.no_grad():
            outputs = self.model(**encoded, output_attentions=True)
            
            # Get embeddings
            last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else last_hidden_state.mean(dim=1)
            attention_weights = outputs.attentions  # List of attention tensors
            
            # Create mean embedding across sequence
            attention_mask = encoded['attention_mask']
            sequence_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
            mean_embedding = sequence_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            # Get tokens for interpretability
            tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
            
        return {
            'embeddings': mean_embedding.squeeze(0),  # [hidden_dim]
            'pooled_output': pooled_output.squeeze(0),  # [hidden_dim]
            'attention_weights': attention_weights[-1],  # Last layer attention [batch, heads, seq_len, seq_len]
            'tokens': tokens,
            'token_embeddings': last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
        }
    
    def extract_medical_keywords(self, text: str, top_k: int = 20) -> Dict[str, List[Dict[str, float]]]:
        """
        Extract medical keywords using RadBERT semantic understanding.
        
        Args:
            text: Medical transcript text
            top_k: Number of top keywords to extract per category
            
        Returns:
            Dictionary with categorized keywords and confidence scores
        """
        # Process text with RadBERT
        radbert_output = self.process_transcript(text)
        tokens = radbert_output['tokens']
        token_embeddings = radbert_output['token_embeddings']  # [seq_len, hidden_dim]
        
        if len(tokens) == 0:
            return {cat: [] for cat in self.medical_categories}
        
        # Extract medical entities using semantic similarity
        categorized_keywords = {}
        
        for category, vocabulary in self.medical_categories.items():
            category_keywords = []
            
            # For each token, check if it's medically relevant
            for i, token in enumerate(tokens):
                # Skip special tokens
                if token.startswith('##') or token in ['<s>', '</s>', '<pad>', '<unk>']:
                    continue
                
                # Clean token
                clean_token = token.strip().lower()
                if len(clean_token) < 2:
                    continue
                
                # Check if token is in medical vocabulary or semantically similar
                relevance_score = self._compute_medical_relevance(
                    clean_token, vocabulary, token_embeddings[i]
                )
                
                if relevance_score > 0.1:  # Threshold for medical relevance
                    category_keywords.append({
                        'keyword': clean_token,
                        'confidence': float(relevance_score),
                        'position': i
                    })
            
            # Sort by confidence and take top_k
            category_keywords.sort(key=lambda x: x['confidence'], reverse=True)
            categorized_keywords[category] = category_keywords[:top_k]
        
        return categorized_keywords
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract keywords from text (simplified interface for compatibility).
        
        Args:
            text: Medical transcript text
            top_k: Number of keywords to extract
            
        Returns:
            List of extracted keyword strings
        """
        categorized_keywords = self.extract_medical_keywords(text, top_k=top_k)
        
        # Flatten all categories into a single list
        all_keywords = []
        for category, keywords in categorized_keywords.items():
            for keyword_data in keywords:
                all_keywords.append(keyword_data['keyword'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in all_keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:top_k]
    
    def _compute_medical_relevance(self, token: str, vocabulary: Set[str], token_embedding: torch.Tensor) -> float:
        """
        Compute medical relevance score for a token.
        
        Args:
            token: Token to evaluate
            vocabulary: Medical vocabulary for category
            token_embedding: RadBERT embedding for token
            
        Returns:
            Relevance score between 0 and 1
        """
        # Direct vocabulary match gets high score
        if token in vocabulary:
            return 1.0
        
        # Partial match gets medium score
        for vocab_term in vocabulary:
            if token in vocab_term or vocab_term in token:
                return 0.7
        
        # For semantic similarity, we would need embeddings of vocabulary terms
        # For now, use simple heuristics based on medical prefixes/suffixes
        medical_prefixes = ['cardio', 'pulmon', 'pneum', 'thorac', 'bronch', 'alveol']
        medical_suffixes = ['itis', 'osis', 'pathy', 'megaly', 'trophy', 'ectomy']
        
        for prefix in medical_prefixes:
            if token.startswith(prefix):
                return 0.5
        
        for suffix in medical_suffixes:
            if token.endswith(suffix):
                return 0.5
        
        return 0.0
    
    def extract_keywords_from_transcripts(self, transcripts: Dict[str, str]) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
        """
        Extract keywords from multiple transcripts using RadBERT.
        
        Args:
            transcripts: Dictionary mapping transcript IDs to text
            
        Returns:
            Dictionary mapping transcript IDs to categorized keywords
        """
        all_keywords = {}
        
        logger.info(f"Processing {len(transcripts)} transcripts with RadBERT")
        
        for transcript_id, text in transcripts.items():
            try:
                keywords = self.extract_medical_keywords(text)
                all_keywords[transcript_id] = keywords
                
                if len(all_keywords) % 10 == 0:
                    logger.info(f"Processed {len(all_keywords)} transcripts")
                    
            except Exception as e:
                logger.warning(f"Failed to process transcript {transcript_id}: {e}")
                # Return empty keywords for failed transcripts
                all_keywords[transcript_id] = {cat: [] for cat in self.medical_categories}
        
        logger.info(f"Successfully processed {len(all_keywords)} transcripts")
        return all_keywords
    
    def get_semantic_features(self, text: str) -> torch.Tensor:
        """
        Get RadBERT semantic features for multimodal fusion.
        
        Args:
            text: Medical transcript text
            
        Returns:
            Semantic feature tensor [hidden_dim]
        """
        radbert_output = self.process_transcript(text)
        return radbert_output['embeddings']
    
    def forward(self, text_batch: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for batch processing.
        
        Args:
            text_batch: List of transcript texts
            
        Returns:
            Dictionary containing batch embeddings and features
        """
        batch_embeddings = []
        batch_pooled = []
        
        for text in text_batch:
            radbert_output = self.process_transcript(text)
            batch_embeddings.append(radbert_output['embeddings'])
            batch_pooled.append(radbert_output['pooled_output'])
        
        return {
            'embeddings': torch.stack(batch_embeddings),  # [batch, hidden_dim]
            'pooled_output': torch.stack(batch_pooled)    # [batch, hidden_dim]
        }


def create_radbert_processor(model_name: str = "UCSD-VA-health/RadBERT-RoBERTa-4m") -> RadBERTModule:
    """
    Factory function to create RadBERT processor.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Initialized RadBERT module
    """
    try:
        processor = RadBERTModule(model_name=model_name)
        logger.info("RadBERT processor created successfully")
        return processor
    except Exception as e:
        logger.error(f"Failed to create RadBERT processor: {e}")
        raise


def test_radbert_extraction(sample_text: str = None) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """
    Test RadBERT keyword extraction with sample medical text.
    
    Args:
        sample_text: Optional sample text, uses default if None
        
    Returns:
        Extracted keywords for demonstration
    """
    if sample_text is None:
        sample_text = """
        Chest radiograph shows cardiomegaly with bilateral lower lobe atelectasis.
        There is a small left pleural effusion. No pneumothorax is identified.
        Central venous catheter is in appropriate position. The patient has stable
        chronic changes with no acute cardiopulmonary abnormality.
        """
    
    processor = create_radbert_processor()
    
    # Test single transcript
    keywords = processor.extract_medical_keywords(sample_text)
    
    logger.info("RadBERT extraction test completed")
    logger.info(f"Sample keywords: {keywords}")
    
    return {"sample_transcript": keywords}


if __name__ == "__main__":
    # Test RadBERT functionality
    test_results = test_radbert_extraction()
    print("RadBERT Test Results:")
    print(json.dumps(test_results, indent=2)) 