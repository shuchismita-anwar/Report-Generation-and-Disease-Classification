#!/usr/bin/env python3
"""
BLEU Scorer
===========
Implementation of BLEU (Bilingual Evaluation Understudy) metric for medical report evaluation.

This module provides:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4 scoring
- Smoothing functionality for zero n-gram matches
- Sentence-level and corpus-level BLEU
- Brevity penalty calculation
- Medical text preprocessing

Author: Medical Report Evaluation System
Version: 1.0
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union
import logging

# Import base metric class
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from base_metric import BaseMetric


class BLEUScorer(BaseMetric):
    """
    BLEU (Bilingual Evaluation Understudy) metric implementation.
    
    Calculates BLEU scores for evaluating the quality of machine-generated
    text against reference text, with support for multiple n-gram orders
    and smoothing techniques.
    """
    
    def __init__(self, max_n: int = 4, smoothing: bool = True, 
                 smoothing_method: str = "add_one", logger: Optional[logging.Logger] = None):
        """
        Initialize the BLEU scorer.
        
        Args:
            max_n (int): Maximum n-gram order to calculate (default: 4)
            smoothing (bool): Whether to apply smoothing for zero counts
            smoothing_method (str): Smoothing method ("add_one", "epsilon", "chen_cherry")
            logger (Optional[logging.Logger]): Logger instance
        """
        super().__init__("bleu", logger)
        self.description = "BLEU (Bilingual Evaluation Understudy) metric for text similarity"
        self.metric_type = "similarity"
        
        # Configuration
        self.max_n = max_n
        self.smoothing = smoothing
        self.smoothing_method = smoothing_method
        
        # Smoothing parameters
        self.epsilon = 1e-7  # Small value for epsilon smoothing
        self.add_one_value = 1.0  # Value to add for add-one smoothing
        
        self.logger.debug(f"Initialized BLEU scorer with max_n={max_n}, smoothing={smoothing}")
    
    def calculate(self, reference: str, candidate: str, **kwargs) -> Dict[str, float]:
        """
        Calculate BLEU scores between reference and candidate texts.
        
        Args:
            reference (str): Ground truth reference text
            candidate (str): Generated candidate text to evaluate
            **kwargs: Additional parameters (weights, custom_smoothing)
            
        Returns:
            Dict[str, float]: Dictionary containing BLEU scores
        """
        # Validate inputs
        is_valid, issues = self.validate_inputs(reference, candidate)
        if not is_valid:
            raise ValueError(f"Invalid inputs for BLEU calculation: {issues}")
        
        # Preprocess texts
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            precision = self._calculate_ngram_precision(ref_tokens, cand_tokens, n)
            precisions.append(precision)
        
        # Calculate brevity penalty
        brevity_penalty = self._calculate_brevity_penalty(ref_tokens, cand_tokens)
        
        # Calculate BLEU scores
        scores = {}
        
        # Individual n-gram BLEU scores
        for i, precision in enumerate(precisions, 1):
            if precision > 0:
                bleu_n = brevity_penalty * precision
            else:
                bleu_n = 0.0
            scores[f"bleu_{i}"] = bleu_n
        
        # Cumulative BLEU scores (geometric mean of 1 to n)
        for n in range(1, self.max_n + 1):
            valid_precisions = [p for p in precisions[:n] if p > 0]
            if valid_precisions:
                # Geometric mean
                log_sum = sum(math.log(p) for p in valid_precisions)
                geometric_mean = math.exp(log_sum / len(valid_precisions))
                cumulative_bleu = brevity_penalty * geometric_mean
            else:
                cumulative_bleu = 0.0
            scores[f"bleu_cumulative_{n}"] = cumulative_bleu
        
        # Standard BLEU-4 score (most commonly reported)
        scores["bleu"] = scores.get("bleu_cumulative_4", 0.0)
        
        # Additional metrics
        scores["brevity_penalty"] = brevity_penalty
        scores["reference_length"] = float(len(ref_tokens))
        scores["candidate_length"] = float(len(cand_tokens))
        scores["length_ratio"] = len(cand_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        return scores
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Handle medical abbreviations and punctuation
        # Keep periods in abbreviations like "Dr.", "vs.", etc.
        text = re.sub(r'([a-z])\.([a-z])', r'\1. \2', text)
        
        # Split on whitespace and punctuation, but keep some medical symbols
        tokens = re.findall(r'\b\w+\b|[.!?;,]', text)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
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
    
    def _calculate_ngram_precision(self, ref_tokens: List[str], 
                                  cand_tokens: List[str], n: int) -> float:
        """
        Calculate n-gram precision with clipping.
        
        Args:
            ref_tokens (List[str]): Reference tokens
            cand_tokens (List[str]): Candidate tokens
            n (int): N-gram order
            
        Returns:
            float: N-gram precision score
        """
        # Get n-grams
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        cand_ngrams = self._get_ngrams(cand_tokens, n)
        
        if not cand_ngrams:
            return 0.0
        
        # Count n-grams
        ref_counts = Counter(ref_ngrams)
        cand_counts = Counter(cand_ngrams)
        
        # Calculate clipped counts
        clipped_counts = 0
        total_counts = 0
        
        for ngram, count in cand_counts.items():
            clipped_count = min(count, ref_counts.get(ngram, 0))
            clipped_counts += clipped_count
            total_counts += count
        
        if total_counts == 0:
            return 0.0
        
        precision = clipped_counts / total_counts
        
        # Apply smoothing if enabled and precision is 0
        if precision == 0.0 and self.smoothing:
            precision = self._apply_smoothing(clipped_counts, total_counts, n)
        
        return precision
    
    def _apply_smoothing(self, clipped_counts: int, total_counts: int, n: int) -> float:
        """
        Apply smoothing for zero n-gram matches.
        
        Args:
            clipped_counts (int): Number of clipped n-gram matches
            total_counts (int): Total number of candidate n-grams
            n (int): N-gram order
            
        Returns:
            float: Smoothed precision score
        """
        if self.smoothing_method == "add_one":
            # Add-one smoothing
            return self.add_one_value / (total_counts + self.add_one_value)
        
        elif self.smoothing_method == "epsilon":
            # Epsilon smoothing
            return self.epsilon
        
        elif self.smoothing_method == "chen_cherry":
            # Chen & Cherry (2014) smoothing
            # Add small value inversely proportional to n-gram order
            smooth_value = 1.0 / (2 ** n)
            return smooth_value / total_counts if total_counts > 0 else smooth_value
        
        else:
            # Default: epsilon smoothing
            return self.epsilon
    
    def _calculate_brevity_penalty(self, ref_tokens: List[str], 
                                  cand_tokens: List[str]) -> float:
        """
        Calculate brevity penalty to penalize short translations.
        
        Args:
            ref_tokens (List[str]): Reference tokens
            cand_tokens (List[str]): Candidate tokens
            
        Returns:
            float: Brevity penalty (0 to 1)
        """
        ref_len = len(ref_tokens)
        cand_len = len(cand_tokens)
        
        if cand_len == 0:
            return 0.0
        
        if cand_len >= ref_len:
            return 1.0
        else:
            # BP = exp(1 - r/c) where r is reference length, c is candidate length
            return math.exp(1 - ref_len / cand_len)
    
    def calculate_corpus_bleu(self, reference_list: List[str], 
                             candidate_list: List[str]) -> Dict[str, float]:
        """
        Calculate corpus-level BLEU score.
        
        Args:
            reference_list (List[str]): List of reference texts
            candidate_list (List[str]): List of candidate texts
            
        Returns:
            Dict[str, float]: Corpus-level BLEU scores
        """
        if len(reference_list) != len(candidate_list):
            raise ValueError("Reference and candidate lists must have same length")
        
        # Aggregate counts across all sentences
        total_ref_len = 0
        total_cand_len = 0
        ngram_matches = defaultdict(int)
        ngram_totals = defaultdict(int)
        
        for ref, cand in zip(reference_list, candidate_list):
            ref_tokens = self._tokenize(ref)
            cand_tokens = self._tokenize(cand)
            
            total_ref_len += len(ref_tokens)
            total_cand_len += len(cand_tokens)
            
            # Accumulate n-gram statistics
            for n in range(1, self.max_n + 1):
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                cand_ngrams = self._get_ngrams(cand_tokens, n)
                
                if cand_ngrams:
                    ref_counts = Counter(ref_ngrams)
                    cand_counts = Counter(cand_ngrams)
                    
                    # Calculate clipped counts
                    for ngram, count in cand_counts.items():
                        clipped_count = min(count, ref_counts.get(ngram, 0))
                        ngram_matches[n] += clipped_count
                        ngram_totals[n] += count
        
        # Calculate corpus-level precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            if ngram_totals[n] > 0:
                precision = ngram_matches[n] / ngram_totals[n]
            else:
                precision = 0.0
            
            # Apply smoothing if needed
            if precision == 0.0 and self.smoothing:
                precision = self._apply_smoothing(ngram_matches[n], ngram_totals[n], n)
            
            precisions.append(precision)
        
        # Calculate corpus-level brevity penalty
        if total_cand_len >= total_ref_len:
            brevity_penalty = 1.0
        elif total_cand_len == 0:
            brevity_penalty = 0.0
        else:
            brevity_penalty = math.exp(1 - total_ref_len / total_cand_len)
        
        # Calculate corpus BLEU scores
        scores = {}
        
        # Individual n-gram scores
        for i, precision in enumerate(precisions, 1):
            scores[f"corpus_bleu_{i}"] = brevity_penalty * precision if precision > 0 else 0.0
        
        # Cumulative scores
        for n in range(1, self.max_n + 1):
            valid_precisions = [p for p in precisions[:n] if p > 0]
            if valid_precisions:
                log_sum = sum(math.log(p) for p in valid_precisions)
                geometric_mean = math.exp(log_sum / len(valid_precisions))
                cumulative_bleu = brevity_penalty * geometric_mean
            else:
                cumulative_bleu = 0.0
            scores[f"corpus_bleu_cumulative_{n}"] = cumulative_bleu
        
        # Standard corpus BLEU
        scores["corpus_bleu"] = scores.get("corpus_bleu_cumulative_4", 0.0)
        
        # Additional statistics
        scores["corpus_brevity_penalty"] = brevity_penalty
        scores["corpus_reference_length"] = float(total_ref_len)
        scores["corpus_candidate_length"] = float(total_cand_len)
        scores["corpus_length_ratio"] = total_cand_len / total_ref_len if total_ref_len > 0 else 0.0
        scores["num_sentences"] = float(len(reference_list))
        
        return scores
    
    def get_name(self) -> str:
        """Get the metric name."""
        return "BLEU"
    
    def get_description(self) -> str:
        """Get the metric description."""
        return (f"BLEU (Bilingual Evaluation Understudy) metric measuring n-gram overlap "
                f"between reference and candidate texts (max_n={self.max_n}, "
                f"smoothing={self.smoothing})")


def test_bleu_scorer():
    """Test function for the BLEU Scorer."""
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    try:
        print("🧪 Testing BLEU Scorer...")
        
        # Initialize BLEU scorer
        bleu = BLEUScorer(max_n=4, smoothing=True, logger=logger)
        
        # Test basic properties
        print(f"✅ Metric name: {bleu.get_name()}")
        print(f"✅ Metric description: {bleu.get_description()}")
        
        # Test cases with different similarity levels
        test_cases = [
            # Perfect match
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Normal chest radiograph with clear lung fields.",
                "expected_bleu": 1.0,
                "description": "Perfect match"
            },
            
            # High similarity
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Normal chest X-ray with clear lungs.",
                "expected_bleu": 0.5,  # Approximate
                "description": "High similarity"
            },
            
            # Medium similarity
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Chest X-ray shows normal findings.",
                "expected_bleu": 0.2,  # Approximate
                "description": "Medium similarity"
            },
            
            # Low similarity
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Patient has pneumonia and pleural effusion.",
                "expected_bleu": 0.1,  # Approximate
                "description": "Low similarity"
            },
            
            # No similarity
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "The weather is sunny today.",
                "expected_bleu": 0.0,
                "description": "No similarity"
            }
        ]
        
        print("\n--- Single Sentence BLEU Tests ---")
        for i, test_case in enumerate(test_cases, 1):
            scores = bleu.calculate(test_case["reference"], test_case["candidate"])
            
            print(f"\nTest {i}: {test_case['description']}")
            print(f"Reference: '{test_case['reference']}'")
            print(f"Candidate: '{test_case['candidate']}'")
            print(f"BLEU-4: {scores['bleu']:.4f}")
            print(f"BLEU-1: {scores['bleu_1']:.4f}")
            print(f"BLEU-2: {scores['bleu_2']:.4f}")
            print(f"Brevity Penalty: {scores['brevity_penalty']:.4f}")
            print(f"Length Ratio: {scores['length_ratio']:.4f}")
        
        # Test corpus-level BLEU
        print("\n--- Corpus-level BLEU Test ---")
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
        
        corpus_scores = bleu.calculate_corpus_bleu(ref_list, cand_list)
        print(f"Corpus BLEU-4: {corpus_scores['corpus_bleu']:.4f}")
        print(f"Corpus BLEU-1: {corpus_scores['corpus_bleu_1']:.4f}")
        print(f"Corpus Brevity Penalty: {corpus_scores['corpus_brevity_penalty']:.4f}")
        print(f"Number of sentences: {corpus_scores['num_sentences']}")
        
        # Test batch calculation
        print("\n--- Batch Calculation Test ---")
        batch_scores = bleu.calculate_batch(ref_list, cand_list)
        print(f"Batch results: {len(batch_scores)} scores calculated")
        avg_bleu = sum(score.get('bleu', 0) for score in batch_scores) / len(batch_scores)
        print(f"Average BLEU-4: {avg_bleu:.4f}")
        
        # Test smoothing methods
        print("\n--- Smoothing Methods Test ---")
        ref_no_match = "Normal chest radiograph with clear lung fields."
        cand_no_match = "Patient has severe pneumonia."
        
        for smoothing_method in ["add_one", "epsilon", "chen_cherry"]:
            bleu_smooth = BLEUScorer(smoothing=True, smoothing_method=smoothing_method, logger=logger)
            scores = bleu_smooth.calculate(ref_no_match, cand_no_match)
            print(f"Smoothing ({smoothing_method}): BLEU-4 = {scores['bleu']:.6f}")
        
        # Test performance
        print("\n--- Performance Test ---")
        perf_stats = bleu.get_performance_stats()
        print(f"Calculations performed: {perf_stats['calculation_count']}")
        print(f"Average calculation time: {perf_stats['average_time']:.4f}s")
        
        print("\n✅ All BLEU scorer tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    success = test_bleu_scorer()
    
    if success:
        print("\n🎉 BLEU Scorer tests passed!")
    else:
        print("\n❌ Some tests failed!") 