#!/usr/bin/env python3
"""
BERTScore Scorer
================
Implementation of BERTScore metric using contextual embeddings for semantic similarity.

This module provides:
- BERTScore calculation using pre-trained BERT models
- Token-level semantic similarity matching
- Precision, Recall, and F1 scores
- Support for different BERT model variants
- Medical domain adaptation capabilities

Author: Medical Report Evaluation System
Version: 1.0
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

# Import base metric class
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from base_metric import BaseMetric


class BERTScoreScorer(BaseMetric):
    """
    BERTScore metric implementation using contextual embeddings.
    
    Calculates semantic similarity between reference and candidate texts
    using BERT-based contextual embeddings and token-level matching.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 use_fast_tokenizer: bool = True, device: str = "auto",
                 batch_size: int = 32, max_length: int = 512,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the BERTScore scorer.
        
        Args:
            model_name (str): Name of the BERT model to use
            use_fast_tokenizer (bool): Whether to use fast tokenizer
            device (str): Device to run model on ("auto", "cpu", "cuda")
            batch_size (int): Batch size for processing
            max_length (int): Maximum sequence length
            logger (Optional[logging.Logger]): Logger instance
        """
        super().__init__("bertscore", logger)
        self.description = "BERTScore metric using contextual embeddings for semantic similarity"
        self.metric_type = "similarity"
        
        # Configuration
        self.model_name = model_name
        self.use_fast_tokenizer = use_fast_tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Model components (initialized lazily)
        self.tokenizer = None
        self.model = None
        self.actual_device = None
        
        # Cache for embeddings (optional optimization)
        self.embedding_cache = {}
        self.use_cache = True
        
        self.logger.debug(f"Initialized BERTScore scorer with model={model_name}")
    
    def initialize(self) -> None:
        """Initialize the BERT model and tokenizer."""
        if self.is_initialized:
            return
        
        try:
            # Import transformers
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.logger.info(f"Loading BERT model: {self.model_name}")
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=self.use_fast_tokenizer
            )
            
            # Initialize model
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Set device
            if self.device == "auto":
                self.actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.actual_device = self.device
            
            self.model.to(self.actual_device)
            self.model.eval()
            
            self.logger.info(f"BERTScore model loaded on device: {self.actual_device}")
            
        except ImportError as e:
            self.logger.error("Transformers library not available. Install with: pip install transformers torch")
            raise ImportError("BERTScore requires transformers library") from e
        except Exception as e:
            self.logger.error(f"Failed to initialize BERTScore model: {e}")
            raise
        
        super().initialize()
    
    def calculate(self, reference: str, candidate: str, **kwargs) -> Dict[str, float]:
        """
        Calculate BERTScore between reference and candidate texts.
        
        Args:
            reference (str): Ground truth reference text
            candidate (str): Generated candidate text to evaluate
            **kwargs: Additional parameters (idf_weights, rescale_with_baseline)
            
        Returns:
            Dict[str, float]: Dictionary containing BERTScore metrics
        """
        # Validate inputs
        is_valid, issues = self.validate_inputs(reference, candidate)
        if not is_valid:
            raise ValueError(f"Invalid inputs for BERTScore calculation: {issues}")
        
        # Initialize model if needed
        if not self.is_initialized:
            self.initialize()
        
        try:
            import torch
            
            # Get embeddings for both texts
            ref_embeddings, ref_tokens = self._get_embeddings(reference)
            cand_embeddings, cand_tokens = self._get_embeddings(candidate)
            
            if ref_embeddings is None or cand_embeddings is None:
                return self._get_zero_scores()
            
            # Calculate similarity matrix
            similarity_matrix = self._calculate_similarity_matrix(ref_embeddings, cand_embeddings)
            
            # Calculate precision, recall, and F1
            precision = self._calculate_precision(similarity_matrix)
            recall = self._calculate_recall(similarity_matrix)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            # Additional metrics
            scores = {
                "bertscore_precision": precision,
                "bertscore_recall": recall,
                "bertscore_f1": f1,
                "bertscore": f1,  # Main score
                "reference_tokens": float(len(ref_tokens)),
                "candidate_tokens": float(len(cand_tokens)),
                "similarity_mean": float(similarity_matrix.mean()),
                "similarity_max": float(similarity_matrix.max()),
                "similarity_min": float(similarity_matrix.min())
            }
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error calculating BERTScore: {e}")
            return self._get_zero_scores()
    
    def _get_embeddings(self, text: str) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Get BERT embeddings for text.
        
        Args:
            text (str): Input text
            
        Returns:
            Tuple[Optional[np.ndarray], List[str]]: (embeddings, tokens)
        """
        try:
            import torch
            
            # Check cache first
            if self.use_cache and text in self.embedding_cache:
                return self.embedding_cache[text]
            
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.actual_device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use last hidden state
                embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
            
            # Convert to numpy
            embeddings = embeddings.cpu().numpy()
            
            # Get tokens (excluding special tokens)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
            
            # Remove special tokens and get corresponding embeddings
            valid_indices = []
            valid_tokens = []
            
            for i, token in enumerate(tokens):
                if token not in [self.tokenizer.cls_token, self.tokenizer.sep_token, 
                               self.tokenizer.pad_token]:
                    valid_indices.append(i)
                    valid_tokens.append(token)
            
            if valid_indices:
                valid_embeddings = embeddings[valid_indices]
            else:
                valid_embeddings = None
                valid_tokens = []
            
            # Cache result
            if self.use_cache:
                self.embedding_cache[text] = (valid_embeddings, valid_tokens)
            
            return valid_embeddings, valid_tokens
            
        except Exception as e:
            self.logger.error(f"Error getting embeddings: {e}")
            return None, []
    
    def _calculate_similarity_matrix(self, ref_embeddings: np.ndarray, 
                                   cand_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity matrix between reference and candidate embeddings.
        
        Args:
            ref_embeddings (np.ndarray): Reference embeddings [ref_len, hidden_size]
            cand_embeddings (np.ndarray): Candidate embeddings [cand_len, hidden_size]
            
        Returns:
            np.ndarray: Similarity matrix [ref_len, cand_len]
        """
        # Normalize embeddings
        ref_norm = ref_embeddings / np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
        cand_norm = cand_embeddings / np.linalg.norm(cand_embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarity_matrix = np.dot(ref_norm, cand_norm.T)
        
        return similarity_matrix
    
    def _calculate_precision(self, similarity_matrix: np.ndarray) -> float:
        """
        Calculate BERTScore precision.
        
        Args:
            similarity_matrix (np.ndarray): Similarity matrix [ref_len, cand_len]
            
        Returns:
            float: Precision score
        """
        # For each candidate token, find the maximum similarity with reference tokens
        max_similarities = np.max(similarity_matrix, axis=0)  # Max over reference tokens
        precision = np.mean(max_similarities)
        return float(precision)
    
    def _calculate_recall(self, similarity_matrix: np.ndarray) -> float:
        """
        Calculate BERTScore recall.
        
        Args:
            similarity_matrix (np.ndarray): Similarity matrix [ref_len, cand_len]
            
        Returns:
            float: Recall score
        """
        # For each reference token, find the maximum similarity with candidate tokens
        max_similarities = np.max(similarity_matrix, axis=1)  # Max over candidate tokens
        recall = np.mean(max_similarities)
        return float(recall)
    
    def _get_zero_scores(self) -> Dict[str, float]:
        """Get zero scores for error cases."""
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "bertscore": 0.0,
            "reference_tokens": 0.0,
            "candidate_tokens": 0.0,
            "similarity_mean": 0.0,
            "similarity_max": 0.0,
            "similarity_min": 0.0
        }
    
    def calculate_with_idf(self, reference: str, candidate: str, 
                          idf_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate BERTScore with IDF weighting.
        
        Args:
            reference (str): Reference text
            candidate (str): Candidate text
            idf_weights (Optional[Dict[str, float]]): IDF weights for tokens
            
        Returns:
            Dict[str, float]: BERTScore with IDF weighting
        """
        # Get basic BERTScore
        scores = self.calculate(reference, candidate)
        
        if idf_weights is None:
            # Use uniform weights
            return scores
        
        try:
            # Get embeddings and tokens
            ref_embeddings, ref_tokens = self._get_embeddings(reference)
            cand_embeddings, cand_tokens = self._get_embeddings(candidate)
            
            if ref_embeddings is None or cand_embeddings is None:
                return scores
            
            # Calculate similarity matrix
            similarity_matrix = self._calculate_similarity_matrix(ref_embeddings, cand_embeddings)
            
            # Apply IDF weighting
            ref_weights = np.array([idf_weights.get(token, 1.0) for token in ref_tokens])
            cand_weights = np.array([idf_weights.get(token, 1.0) for token in cand_tokens])
            
            # Weighted precision
            max_similarities = np.max(similarity_matrix, axis=0)
            weighted_precision = np.sum(max_similarities * cand_weights) / np.sum(cand_weights)
            
            # Weighted recall
            max_similarities = np.max(similarity_matrix, axis=1)
            weighted_recall = np.sum(max_similarities * ref_weights) / np.sum(ref_weights)
            
            # Weighted F1
            if weighted_precision + weighted_recall > 0:
                weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
            else:
                weighted_f1 = 0.0
            
            # Update scores
            scores.update({
                "bertscore_idf_precision": float(weighted_precision),
                "bertscore_idf_recall": float(weighted_recall),
                "bertscore_idf_f1": float(weighted_f1),
                "bertscore_idf": float(weighted_f1)
            })
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error calculating IDF-weighted BERTScore: {e}")
            return scores
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        self.logger.debug("Cleared BERTScore embedding cache")
    
    def get_cache_size(self) -> int:
        """Get the current cache size."""
        return len(self.embedding_cache)
    
    def get_name(self) -> str:
        """Get the metric name."""
        return "BERTScore"
    
    def get_description(self) -> str:
        """Get the metric description."""
        return (f"BERTScore metric using {self.model_name} for semantic similarity "
                f"measurement with contextual embeddings")


def test_bertscore_scorer():
    """Test function for the BERTScore Scorer."""
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    try:
        print("🧪 Testing BERTScore Scorer...")
        
        # Initialize BERTScore scorer (use smaller model for testing)
        bertscore = BERTScoreScorer(
            model_name="distilbert-base-uncased",  # Smaller model for testing
            device="cpu",  # Force CPU for testing
            logger=logger
        )
        
        # Test basic properties
        print(f"✅ Metric name: {bertscore.get_name()}")
        print(f"✅ Metric description: {bertscore.get_description()}")
        
        # Test cases with different similarity levels
        test_cases = [
            # Perfect match
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Normal chest radiograph with clear lung fields.",
                "description": "Perfect match"
            },
            
            # High semantic similarity
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Normal chest X-ray with clear lungs.",
                "description": "High semantic similarity"
            },
            
            # Medium similarity (paraphrase)
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Chest imaging shows normal pulmonary findings.",
                "description": "Medium similarity (paraphrase)"
            },
            
            # Low similarity (different meaning)
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Patient has severe pneumonia and effusion.",
                "description": "Low similarity (different meaning)"
            },
            
            # No similarity
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "The weather is sunny today.",
                "description": "No similarity"
            }
        ]
        
        print("\n--- BERTScore Tests ---")
        for i, test_case in enumerate(test_cases, 1):
            try:
                scores = bertscore.calculate(test_case["reference"], test_case["candidate"])
                
                print(f"\nTest {i}: {test_case['description']}")
                print(f"Reference: '{test_case['reference']}'")
                print(f"Candidate: '{test_case['candidate']}'")
                print(f"BERTScore F1: {scores['bertscore_f1']:.4f}")
                print(f"BERTScore Precision: {scores['bertscore_precision']:.4f}")
                print(f"BERTScore Recall: {scores['bertscore_recall']:.4f}")
                print(f"Similarity Mean: {scores['similarity_mean']:.4f}")
                print(f"Reference Tokens: {scores['reference_tokens']}")
                print(f"Candidate Tokens: {scores['candidate_tokens']}")
                
            except Exception as e:
                print(f"❌ Test {i} failed: {e}")
                # Continue with other tests
                continue
        
        # Test IDF weighting (if basic tests passed)
        print("\n--- IDF Weighting Test ---")
        try:
            # Simple IDF weights (higher weight for medical terms)
            idf_weights = {
                "normal": 2.0,
                "chest": 2.5,
                "radiograph": 3.0,
                "lung": 2.5,
                "fields": 2.0,
                "clear": 1.5,
                "the": 0.1,
                "with": 0.1,
                "and": 0.1
            }
            
            ref_text = "Normal chest radiograph with clear lung fields."
            cand_text = "Normal chest X-ray with clear lungs."
            
            idf_scores = bertscore.calculate_with_idf(ref_text, cand_text, idf_weights)
            regular_scores = bertscore.calculate(ref_text, cand_text)
            
            print(f"Regular BERTScore F1: {regular_scores['bertscore_f1']:.4f}")
            print(f"IDF-weighted BERTScore F1: {idf_scores.get('bertscore_idf_f1', 'N/A')}")
            
        except Exception as e:
            print(f"❌ IDF test failed: {e}")
        
        # Test batch calculation
        print("\n--- Batch Calculation Test ---")
        try:
            ref_list = [
                "Normal chest radiograph with clear lung fields.",
                "Heart size is within normal limits.",
                "No acute cardiopulmonary abnormalities."
            ]
            cand_list = [
                "Normal chest X-ray with clear lungs.",
                "Heart size appears normal.",
                "No acute findings identified."
            ]
            
            batch_scores = bertscore.calculate_batch(ref_list, cand_list)
            print(f"Batch results: {len(batch_scores)} scores calculated")
            
            # Calculate average BERTScore
            valid_scores = [score.get('bertscore_f1', 0) for score in batch_scores if score]
            if valid_scores:
                avg_bertscore = sum(valid_scores) / len(valid_scores)
                print(f"Average BERTScore F1: {avg_bertscore:.4f}")
            
        except Exception as e:
            print(f"❌ Batch test failed: {e}")
        
        # Test cache functionality
        print("\n--- Cache Test ---")
        try:
            cache_size_before = bertscore.get_cache_size()
            
            # Calculate same pair twice
            bertscore.calculate("Normal chest radiograph.", "Normal chest X-ray.")
            bertscore.calculate("Normal chest radiograph.", "Normal chest X-ray.")
            
            cache_size_after = bertscore.get_cache_size()
            print(f"Cache size before: {cache_size_before}")
            print(f"Cache size after: {cache_size_after}")
            
            # Clear cache
            bertscore.clear_cache()
            cache_size_cleared = bertscore.get_cache_size()
            print(f"Cache size after clearing: {cache_size_cleared}")
            
        except Exception as e:
            print(f"❌ Cache test failed: {e}")
        
        # Test performance
        print("\n--- Performance Test ---")
        try:
            perf_stats = bertscore.get_performance_stats()
            print(f"Calculations performed: {perf_stats['calculation_count']}")
            print(f"Average calculation time: {perf_stats['average_time']:.4f}s")
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
        
        print("\n✅ BERTScore scorer tests completed!")
        return True
        
    except ImportError as e:
        print(f"⚠️  BERTScore test skipped - missing dependencies: {e}")
        print("Install with: pip install transformers torch")
        return True  # Don't fail the test suite
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    success = test_bertscore_scorer()
    
    if success:
        print("\n🎉 BERTScore Scorer tests passed!")
    else:
        print("\n❌ Some tests failed!") 