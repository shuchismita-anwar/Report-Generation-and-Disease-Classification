#!/usr/bin/env python3
"""
CIDEr Scorer
============
Implementation of CIDEr (Consensus-based Image Description Evaluation) metric.

This module provides:
- CIDEr score calculation with TF-IDF weighting
- N-gram based consensus evaluation
- Support for multiple reference texts
- Medical text preprocessing
- Stemming and synonym support

Author: Medical Report Evaluation System
Version: 1.0
"""

import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union
import logging

# Import base metric class
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from base_metric import BaseMetric


class CIDErScorer(BaseMetric):
    """
    CIDEr (Consensus-based Image Description Evaluation) metric implementation.
    
    Calculates CIDEr scores using TF-IDF weighted n-gram matching
    to measure consensus between candidate and reference descriptions.
    """
    
    def __init__(self, max_n: int = 4, sigma: float = 6.0, 
                 use_stemming: bool = False, remove_stopwords: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the CIDEr scorer.
        
        Args:
            max_n (int): Maximum n-gram order to consider (default: 4)
            sigma (float): Standard deviation for Gaussian penalty (default: 6.0)
            use_stemming (bool): Whether to apply stemming
            remove_stopwords (bool): Whether to remove stopwords
            logger (Optional[logging.Logger]): Logger instance
        """
        super().__init__("cider", logger)
        self.description = "CIDEr (Consensus-based Image Description Evaluation) metric"
        self.metric_type = "similarity"
        
        # Configuration
        self.max_n = max_n
        self.sigma = sigma
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        
        # Document frequency storage for TF-IDF calculation
        self.document_frequency = defaultdict(int)
        self.total_documents = 0
        self.is_corpus_initialized = False
        
        # Stopwords (basic medical-aware list)
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'there', 'this', 'these', 'they',
            'have', 'had', 'been', 'being', 'do', 'does', 'did', 'can', 'could',
            'should', 'would', 'may', 'might', 'must', 'shall', 'will', 'am'
        }
        
        # Initialize stemmer if needed
        self.stemmer = None
        if self.use_stemming:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
                self.logger.debug("Initialized Porter stemmer for CIDEr")
            except ImportError:
                self.logger.warning("NLTK not available, stemming disabled")
                self.use_stemming = False
        
        self.logger.debug(f"Initialized CIDEr scorer with max_n={max_n}, sigma={sigma}")
    
    def calculate(self, reference: str, candidate: str, **kwargs) -> Dict[str, float]:
        """
        Calculate CIDEr score between reference and candidate texts.
        
        Args:
            reference (str): Ground truth reference text
            candidate (str): Generated candidate text to evaluate
            **kwargs: Additional parameters (multiple_references)
            
        Returns:
            Dict[str, float]: Dictionary containing CIDEr scores
        """
        # Validate inputs
        is_valid, issues = self.validate_inputs(reference, candidate)
        if not is_valid:
            raise ValueError(f"Invalid inputs for CIDEr calculation: {issues}")
        
        # Handle multiple references if provided
        multiple_references = kwargs.get('multiple_references', [])
        if multiple_references:
            references = [reference] + multiple_references
        else:
            references = [reference]
        
        # Preprocess texts
        ref_tokens_list = [self._tokenize_and_preprocess(ref) for ref in references]
        cand_tokens = self._tokenize_and_preprocess(candidate)
        
        if not any(ref_tokens_list) or not cand_tokens:
            return self._get_zero_scores()
        
        # Calculate CIDEr score
        cider_score = self._calculate_cider_score(ref_tokens_list, cand_tokens)
        
        # Calculate individual n-gram scores
        ngram_scores = {}
        for n in range(1, self.max_n + 1):
            ngram_score = self._calculate_ngram_cider(ref_tokens_list, cand_tokens, n)
            ngram_scores[f"cider_{n}"] = ngram_score
        
        # Additional metrics
        scores = {
            "cider": cider_score,
            "cider_mean": cider_score,  # Alias for compatibility
            **ngram_scores,
            "reference_length": float(len(ref_tokens_list[0]) if ref_tokens_list[0] else 0),
            "candidate_length": float(len(cand_tokens)),
            "num_references": float(len(references)),
            "length_ratio": len(cand_tokens) / len(ref_tokens_list[0]) if ref_tokens_list[0] else 0.0
        }
        
        return scores
    
    def _tokenize_and_preprocess(self, text: str) -> List[str]:
        """
        Tokenize and preprocess text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of preprocessed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Handle medical abbreviations
        text = re.sub(r'([a-z])\.([a-z])', r'\1. \2', text)
        
        # Tokenize
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Apply stemming if enabled
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """
        Extract n-grams from a list of tokens.
        
        Args:
            tokens (List[str]): List of tokens
            n (int): N-gram order
            
        Returns:
            List[Tuple[str, ...]]: List of n-grams
        """
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _calculate_cider_score(self, ref_tokens_list: List[List[str]], 
                              cand_tokens: List[str]) -> float:
        """
        Calculate overall CIDEr score.
        
        Args:
            ref_tokens_list (List[List[str]]): List of reference token lists
            cand_tokens (List[str]): Candidate tokens
            
        Returns:
            float: CIDEr score
        """
        total_score = 0.0
        
        # Calculate score for each n-gram order
        for n in range(1, self.max_n + 1):
            ngram_score = self._calculate_ngram_cider(ref_tokens_list, cand_tokens, n)
            total_score += ngram_score
        
        # Average over n-gram orders
        cider_score = total_score / self.max_n
        
        return cider_score
    
    def _calculate_ngram_cider(self, ref_tokens_list: List[List[str]], 
                              cand_tokens: List[str], n: int) -> float:
        """
        Calculate CIDEr score for specific n-gram order.
        
        Args:
            ref_tokens_list (List[List[str]]): List of reference token lists
            cand_tokens (List[str]): Candidate tokens
            n (int): N-gram order
            
        Returns:
            float: N-gram CIDEr score
        """
        # Get candidate n-grams
        cand_ngrams = self._get_ngrams(cand_tokens, n)
        if not cand_ngrams:
            return 0.0
        
        # Count candidate n-grams
        cand_counts = Counter(cand_ngrams)
        
        # Calculate TF for candidate
        cand_tf = {}
        for ngram, count in cand_counts.items():
            cand_tf[ngram] = count / len(cand_ngrams)
        
        # Calculate average reference vector
        ref_tf_list = []
        for ref_tokens in ref_tokens_list:
            if not ref_tokens:
                continue
                
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            if not ref_ngrams:
                continue
                
            ref_counts = Counter(ref_ngrams)
            ref_tf = {}
            for ngram, count in ref_counts.items():
                ref_tf[ngram] = count / len(ref_ngrams)
            
            ref_tf_list.append(ref_tf)
        
        if not ref_tf_list:
            return 0.0
        
        # Calculate average reference TF
        all_ngrams = set()
        for ref_tf in ref_tf_list:
            all_ngrams.update(ref_tf.keys())
        all_ngrams.update(cand_tf.keys())
        
        avg_ref_tf = {}
        for ngram in all_ngrams:
            tf_values = [ref_tf.get(ngram, 0.0) for ref_tf in ref_tf_list]
            avg_ref_tf[ngram] = sum(tf_values) / len(ref_tf_list)
        
        # Calculate IDF weights (simplified - using document frequency from corpus)
        idf_weights = self._calculate_idf_weights(all_ngrams, n)
        
        # Calculate cosine similarity with TF-IDF weighting
        numerator = 0.0
        cand_norm = 0.0
        ref_norm = 0.0
        
        for ngram in all_ngrams:
            cand_tfidf = cand_tf.get(ngram, 0.0) * idf_weights.get(ngram, 1.0)
            ref_tfidf = avg_ref_tf.get(ngram, 0.0) * idf_weights.get(ngram, 1.0)
            
            numerator += cand_tfidf * ref_tfidf
            cand_norm += cand_tfidf ** 2
            ref_norm += ref_tfidf ** 2
        
        # Calculate cosine similarity
        if cand_norm > 0 and ref_norm > 0:
            similarity = numerator / (math.sqrt(cand_norm) * math.sqrt(ref_norm))
        else:
            similarity = 0.0
        
        # Apply length penalty (Gaussian penalty based on length difference)
        length_penalty = self._calculate_length_penalty(ref_tokens_list, cand_tokens)
        
        # Final n-gram CIDEr score
        ngram_cider = similarity * length_penalty
        
        return ngram_cider
    
    def _calculate_idf_weights(self, ngrams: set, n: int) -> Dict[Tuple[str, ...], float]:
        """
        Calculate IDF weights for n-grams.
        
        Args:
            ngrams (set): Set of n-grams
            n (int): N-gram order
            
        Returns:
            Dict[Tuple[str, ...], float]: IDF weights for each n-gram
        """
        idf_weights = {}
        
        # If corpus is not initialized, use uniform weights
        if not self.is_corpus_initialized or self.total_documents == 0:
            for ngram in ngrams:
                idf_weights[ngram] = 1.0
            return idf_weights
        
        # Calculate IDF for each n-gram
        for ngram in ngrams:
            doc_freq = self.document_frequency.get(ngram, 1)  # Add-one smoothing
            idf = math.log(self.total_documents / doc_freq)
            idf_weights[ngram] = max(idf, 0.1)  # Minimum IDF threshold
        
        return idf_weights
    
    def _calculate_length_penalty(self, ref_tokens_list: List[List[str]], 
                                 cand_tokens: List[str]) -> float:
        """
        Calculate length penalty using Gaussian function.
        
        Args:
            ref_tokens_list (List[List[str]]): List of reference token lists
            cand_tokens (List[str]): Candidate tokens
            
        Returns:
            float: Length penalty (0 to 1)
        """
        # Calculate average reference length
        ref_lengths = [len(ref_tokens) for ref_tokens in ref_tokens_list if ref_tokens]
        if not ref_lengths:
            return 1.0
        
        avg_ref_length = sum(ref_lengths) / len(ref_lengths)
        cand_length = len(cand_tokens)
        
        # Gaussian penalty based on length difference
        length_diff = abs(cand_length - avg_ref_length)
        penalty = math.exp(-(length_diff ** 2) / (2 * self.sigma ** 2))
        
        return penalty
    
    def initialize_corpus(self, corpus_texts: List[str]) -> None:
        """
        Initialize corpus statistics for IDF calculation.
        
        Args:
            corpus_texts (List[str]): List of corpus texts for IDF calculation
        """
        self.logger.info(f"Initializing CIDEr corpus with {len(corpus_texts)} documents")
        
        self.document_frequency.clear()
        self.total_documents = len(corpus_texts)
        
        # Calculate document frequency for all n-grams
        for text in corpus_texts:
            tokens = self._tokenize_and_preprocess(text)
            
            # Get all n-grams for this document
            doc_ngrams = set()
            for n in range(1, self.max_n + 1):
                ngrams = self._get_ngrams(tokens, n)
                doc_ngrams.update(ngrams)
            
            # Update document frequency
            for ngram in doc_ngrams:
                self.document_frequency[ngram] += 1
        
        self.is_corpus_initialized = True
        self.logger.info(f"Corpus initialized with {len(self.document_frequency)} unique n-grams")
    
    def calculate_corpus_cider(self, reference_list: List[str], 
                              candidate_list: List[str]) -> Dict[str, float]:
        """
        Calculate corpus-level CIDEr score.
        
        Args:
            reference_list (List[str]): List of reference texts
            candidate_list (List[str]): List of candidate texts
            
        Returns:
            Dict[str, float]: Corpus-level CIDEr scores
        """
        if len(reference_list) != len(candidate_list):
            raise ValueError("Reference and candidate lists must have same length")
        
        # Initialize corpus if not done
        if not self.is_corpus_initialized:
            all_texts = reference_list + candidate_list
            self.initialize_corpus(all_texts)
        
        # Calculate individual CIDEr scores
        individual_scores = []
        for ref, cand in zip(reference_list, candidate_list):
            scores = self.calculate(ref, cand)
            individual_scores.append(scores["cider"])
        
        # Calculate corpus statistics
        corpus_cider = sum(individual_scores) / len(individual_scores)
        
        scores = {
            "corpus_cider": corpus_cider,
            "corpus_cider_std": self._calculate_std(individual_scores),
            "corpus_cider_min": min(individual_scores),
            "corpus_cider_max": max(individual_scores),
            "num_pairs": float(len(individual_scores))
        }
        
        return scores
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def _get_zero_scores(self) -> Dict[str, float]:
        """Get zero scores for error cases."""
        scores = {"cider": 0.0, "cider_mean": 0.0}
        
        # Add individual n-gram scores
        for n in range(1, self.max_n + 1):
            scores[f"cider_{n}"] = 0.0
        
        scores.update({
            "reference_length": 0.0,
            "candidate_length": 0.0,
            "num_references": 0.0,
            "length_ratio": 0.0
        })
        
        return scores
    
    def get_name(self) -> str:
        """Get the metric name."""
        return "CIDEr"
    
    def get_description(self) -> str:
        """Get the metric description."""
        return (f"CIDEr (Consensus-based Image Description Evaluation) metric "
                f"with max_n={self.max_n}, sigma={self.sigma}, "
                f"stemming={self.use_stemming}")


def test_cider_scorer():
    """Test function for the CIDEr Scorer."""
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    try:
        print("🧪 Testing CIDEr Scorer...")
        
        # Initialize CIDEr scorer
        cider = CIDErScorer(max_n=4, sigma=6.0, use_stemming=False, logger=logger)
        
        # Test basic properties
        print(f"✅ Metric name: {cider.get_name()}")
        print(f"✅ Metric description: {cider.get_description()}")
        
        # Initialize corpus for IDF calculation
        corpus_texts = [
            "Normal chest radiograph with clear lung fields.",
            "Heart size is within normal limits.",
            "No acute cardiopulmonary abnormalities.",
            "Chest X-ray shows normal findings.",
            "Clear lungs bilaterally.",
            "Normal cardiac silhouette.",
            "No pleural effusion or pneumothorax.",
            "Unremarkable chest imaging.",
            "Normal pulmonary vasculature.",
            "No focal consolidation."
        ]
        
        cider.initialize_corpus(corpus_texts)
        print(f"✅ Corpus initialized with {len(corpus_texts)} documents")
        
        # Test cases with different similarity levels
        test_cases = [
            # Perfect match
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Normal chest radiograph with clear lung fields.",
                "description": "Perfect match"
            },
            
            # High similarity
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Normal chest X-ray with clear lungs.",
                "description": "High similarity"
            },
            
            # Medium similarity
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Chest imaging shows normal findings.",
                "description": "Medium similarity"
            },
            
            # Low similarity
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Patient has pneumonia and effusion.",
                "description": "Low similarity"
            },
            
            # Length difference
            {
                "reference": "Normal chest radiograph.",
                "candidate": "Normal chest radiograph with clear lung fields and normal heart size.",
                "description": "Length difference"
            }
        ]
        
        print("\n--- CIDEr Score Tests ---")
        for i, test_case in enumerate(test_cases, 1):
            scores = cider.calculate(test_case["reference"], test_case["candidate"])
            
            print(f"\nTest {i}: {test_case['description']}")
            print(f"Reference: '{test_case['reference']}'")
            print(f"Candidate: '{test_case['candidate']}'")
            print(f"CIDEr: {scores['cider']:.4f}")
            print(f"CIDEr-1: {scores['cider_1']:.4f}")
            print(f"CIDEr-2: {scores['cider_2']:.4f}")
            print(f"CIDEr-3: {scores['cider_3']:.4f}")
            print(f"CIDEr-4: {scores['cider_4']:.4f}")
            print(f"Length ratio: {scores['length_ratio']:.4f}")
        
        # Test multiple references
        print("\n--- Multiple References Test ---")
        ref_text = "Normal chest radiograph with clear lung fields."
        multiple_refs = [
            "Normal chest X-ray with clear lungs.",
            "Chest imaging shows normal findings.",
            "Clear lung fields bilaterally."
        ]
        cand_text = "Normal chest radiograph with clear pulmonary areas."
        
        multi_scores = cider.calculate(ref_text, cand_text, multiple_references=multiple_refs)
        single_scores = cider.calculate(ref_text, cand_text)
        
        print(f"Single reference CIDEr: {single_scores['cider']:.4f}")
        print(f"Multiple references CIDEr: {multi_scores['cider']:.4f}")
        print(f"Number of references: {multi_scores['num_references']}")
        
        # Test corpus-level CIDEr
        print("\n--- Corpus-level CIDEr Test ---")
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
        
        corpus_scores = cider.calculate_corpus_cider(ref_list, cand_list)
        print(f"Corpus CIDEr: {corpus_scores['corpus_cider']:.4f}")
        print(f"Corpus CIDEr Std: {corpus_scores['corpus_cider_std']:.4f}")
        print(f"Corpus CIDEr Min: {corpus_scores['corpus_cider_min']:.4f}")
        print(f"Corpus CIDEr Max: {corpus_scores['corpus_cider_max']:.4f}")
        
        # Test batch calculation
        print("\n--- Batch Calculation Test ---")
        batch_scores = cider.calculate_batch(ref_list, cand_list)
        print(f"Batch results: {len(batch_scores)} scores calculated")
        
        # Calculate average CIDEr
        avg_cider = sum(score.get('cider', 0) for score in batch_scores) / len(batch_scores)
        print(f"Average CIDEr: {avg_cider:.4f}")
        
        # Test with stemming
        print("\n--- Stemming Test ---")
        cider_stem = CIDErScorer(use_stemming=True, logger=logger)
        cider_stem.initialize_corpus(corpus_texts)
        
        ref_stem = "The examination shows normal findings."
        cand_stem = "The exam showed normal finding."
        
        scores_no_stem = cider.calculate(ref_stem, cand_stem)
        scores_with_stem = cider_stem.calculate(ref_stem, cand_stem)
        
        print(f"Without stemming: {scores_no_stem['cider']:.4f}")
        print(f"With stemming: {scores_with_stem['cider']:.4f}")
        
        # Test edge cases
        print("\n--- Edge Cases Test ---")
        
        # Empty strings
        try:
            empty_scores = cider.calculate("", "")
            print("❌ Empty strings should raise error")
        except ValueError:
            print("✅ Empty strings correctly handled")
        
        # Very short texts
        short_scores = cider.calculate("Normal.", "Normal.")
        print(f"✅ Short text CIDEr: {short_scores['cider']:.4f}")
        
        # Test performance
        print("\n--- Performance Test ---")
        perf_stats = cider.get_performance_stats()
        print(f"Calculations performed: {perf_stats['calculation_count']}")
        print(f"Average calculation time: {perf_stats['average_time']:.4f}s")
        
        print("\n✅ All CIDEr scorer tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    success = test_cider_scorer()
    
    if success:
        print("\n🎉 CIDEr Scorer tests passed!")
    else:
        print("\n❌ Some tests failed!") 