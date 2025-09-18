#!/usr/bin/env python3
"""
ROUGE Scorer
============
Implementation of ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics.

This module provides:
- ROUGE-1 (unigram overlap)
- ROUGE-2 (bigram overlap)
- ROUGE-L (longest common subsequence)
- Precision, Recall, and F1 scores for each variant
- Medical text preprocessing

Author: Medical Report Evaluation System
Version: 1.0
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Set
import logging

# Import base metric class
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from base_metric import BaseMetric


class ROUGEScorer(BaseMetric):
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric implementation.
    
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L scores for evaluating
    the quality of summaries and generated text against reference text.
    """
    
    def __init__(self, use_stemming: bool = False, remove_stopwords: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the ROUGE scorer.
        
        Args:
            use_stemming (bool): Whether to apply stemming to tokens
            remove_stopwords (bool): Whether to remove stopwords
            logger (Optional[logging.Logger]): Logger instance
        """
        super().__init__("rouge", logger)
        self.description = "ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric"
        self.metric_type = "similarity"
        
        # Configuration
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        
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
                self.logger.debug("Initialized Porter stemmer for ROUGE")
            except ImportError:
                self.logger.warning("NLTK not available, stemming disabled")
                self.use_stemming = False
        
        self.logger.debug(f"Initialized ROUGE scorer with stemming={use_stemming}, "
                         f"remove_stopwords={remove_stopwords}")
    
    def calculate(self, reference: str, candidate: str, **kwargs) -> Dict[str, float]:
        """
        Calculate ROUGE scores between reference and candidate texts.
        
        Args:
            reference (str): Ground truth reference text
            candidate (str): Generated candidate text to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, float]: Dictionary containing ROUGE scores
        """
        # Validate inputs
        is_valid, issues = self.validate_inputs(reference, candidate)
        if not is_valid:
            raise ValueError(f"Invalid inputs for ROUGE calculation: {issues}")
        
        # Preprocess texts
        ref_tokens = self._tokenize_and_preprocess(reference)
        cand_tokens = self._tokenize_and_preprocess(candidate)
        
        scores = {}
        
        # Calculate ROUGE-1 (unigram overlap)
        rouge_1 = self._calculate_rouge_n(ref_tokens, cand_tokens, n=1)
        scores.update({
            "rouge_1_precision": rouge_1["precision"],
            "rouge_1_recall": rouge_1["recall"],
            "rouge_1_f1": rouge_1["f1"]
        })
        
        # Calculate ROUGE-2 (bigram overlap)
        rouge_2 = self._calculate_rouge_n(ref_tokens, cand_tokens, n=2)
        scores.update({
            "rouge_2_precision": rouge_2["precision"],
            "rouge_2_recall": rouge_2["recall"],
            "rouge_2_f1": rouge_2["f1"]
        })
        
        # Calculate ROUGE-L (longest common subsequence)
        rouge_l = self._calculate_rouge_l(ref_tokens, cand_tokens)
        scores.update({
            "rouge_l_precision": rouge_l["precision"],
            "rouge_l_recall": rouge_l["recall"],
            "rouge_l_f1": rouge_l["f1"]
        })
        
        # Overall ROUGE score (average of F1 scores)
        f1_scores = [rouge_1["f1"], rouge_2["f1"], rouge_l["f1"]]
        scores["rouge_avg_f1"] = sum(f1_scores) / len(f1_scores)
        
        # Additional metrics
        scores["reference_length"] = float(len(ref_tokens))
        scores["candidate_length"] = float(len(cand_tokens))
        scores["length_ratio"] = len(cand_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
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
        
        # Handle medical abbreviations and punctuation
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
    
    def _calculate_rouge_n(self, ref_tokens: List[str], cand_tokens: List[str], 
                          n: int) -> Dict[str, float]:
        """
        Calculate ROUGE-N scores.
        
        Args:
            ref_tokens (List[str]): Reference tokens
            cand_tokens (List[str]): Candidate tokens
            n (int): N-gram order
            
        Returns:
            Dict[str, float]: Precision, recall, and F1 scores
        """
        # Get n-grams
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        cand_ngrams = self._get_ngrams(cand_tokens, n)
        
        if not ref_ngrams and not cand_ngrams:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if not ref_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        if not cand_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Count n-grams
        ref_counts = Counter(ref_ngrams)
        cand_counts = Counter(cand_ngrams)
        
        # Calculate overlap
        overlap = 0
        for ngram, count in cand_counts.items():
            if ngram in ref_counts:
                overlap += min(count, ref_counts[ngram])
        
        # Calculate precision and recall
        precision = overlap / len(cand_ngrams) if len(cand_ngrams) > 0 else 0.0
        recall = overlap / len(ref_ngrams) if len(ref_ngrams) > 0 else 0.0
        
        # Calculate F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def _calculate_rouge_l(self, ref_tokens: List[str], cand_tokens: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE-L scores based on longest common subsequence.
        
        Args:
            ref_tokens (List[str]): Reference tokens
            cand_tokens (List[str]): Candidate tokens
            
        Returns:
            Dict[str, float]: Precision, recall, and F1 scores
        """
        if not ref_tokens and not cand_tokens:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if not ref_tokens or not cand_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Calculate longest common subsequence length
        lcs_length = self._lcs_length(ref_tokens, cand_tokens)
        
        # Calculate precision and recall
        precision = lcs_length / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
        recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        # Calculate F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Calculate the length of the longest common subsequence.
        
        Args:
            seq1 (List[str]): First sequence
            seq2 (List[str]): Second sequence
            
        Returns:
            int: Length of longest common subsequence
        """
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def calculate_rouge_w(self, reference: str, candidate: str, weight: float = 1.2) -> Dict[str, float]:
        """
        Calculate ROUGE-W (weighted longest common subsequence).
        
        Args:
            reference (str): Reference text
            candidate (str): Candidate text
            weight (float): Weight parameter for consecutive matches
            
        Returns:
            Dict[str, float]: ROUGE-W scores
        """
        ref_tokens = self._tokenize_and_preprocess(reference)
        cand_tokens = self._tokenize_and_preprocess(candidate)
        
        if not ref_tokens or not cand_tokens:
            return {"rouge_w_precision": 0.0, "rouge_w_recall": 0.0, "rouge_w_f1": 0.0}
        
        # Calculate weighted LCS
        wlcs_length = self._weighted_lcs_length(ref_tokens, cand_tokens, weight)
        
        # Normalization factors
        ref_norm = self._weighted_lcs_length(ref_tokens, ref_tokens, weight)
        cand_norm = self._weighted_lcs_length(cand_tokens, cand_tokens, weight)
        
        # Calculate precision and recall
        precision = wlcs_length / cand_norm if cand_norm > 0 else 0.0
        recall = wlcs_length / ref_norm if ref_norm > 0 else 0.0
        
        # Calculate F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            "rouge_w_precision": precision,
            "rouge_w_recall": recall,
            "rouge_w_f1": f1
        }
    
    def _weighted_lcs_length(self, seq1: List[str], seq2: List[str], weight: float) -> float:
        """
        Calculate weighted LCS length with consecutive match bonus.
        
        Args:
            seq1 (List[str]): First sequence
            seq2 (List[str]): Second sequence
            weight (float): Weight for consecutive matches
            
        Returns:
            float: Weighted LCS length
        """
        m, n = len(seq1), len(seq2)
        
        # DP table for LCS length and consecutive match length
        dp = [[[0.0, 0] for _ in range(n + 1)] for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    # Consecutive match
                    consecutive = dp[i - 1][j - 1][1] + 1
                    length = dp[i - 1][j - 1][0] + weight ** consecutive
                    dp[i][j] = [length, consecutive]
                else:
                    # No match, take maximum from adjacent cells
                    if dp[i - 1][j][0] > dp[i][j - 1][0]:
                        dp[i][j] = [dp[i - 1][j][0], 0]
                    else:
                        dp[i][j] = [dp[i][j - 1][0], 0]
        
        return dp[m][n][0]
    
    def get_name(self) -> str:
        """Get the metric name."""
        return "ROUGE"
    
    def get_description(self) -> str:
        """Get the metric description."""
        return (f"ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric "
                f"measuring n-gram and LCS overlap (stemming={self.use_stemming}, "
                f"remove_stopwords={self.remove_stopwords})")


def test_rouge_scorer():
    """Test function for the ROUGE Scorer."""
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    try:
        print("🧪 Testing ROUGE Scorer...")
        
        # Initialize ROUGE scorer
        rouge = ROUGEScorer(use_stemming=False, remove_stopwords=False, logger=logger)
        
        # Test basic properties
        print(f"✅ Metric name: {rouge.get_name()}")
        print(f"✅ Metric description: {rouge.get_description()}")
        
        # Test cases with different similarity levels
        test_cases = [
            # Perfect match
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Normal chest radiograph with clear lung fields.",
                "description": "Perfect match"
            },
            
            # High similarity (synonym substitution)
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Normal chest X-ray with clear lung areas.",
                "description": "High similarity"
            },
            
            # Medium similarity (partial overlap)
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Chest X-ray shows normal findings.",
                "description": "Medium similarity"
            },
            
            # Low similarity (different content)
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Patient has pneumonia and pleural effusion.",
                "description": "Low similarity"
            },
            
            # Different length (shorter candidate)
            {
                "reference": "Normal chest radiograph with clear lung fields and normal heart size.",
                "candidate": "Normal chest radiograph.",
                "description": "Shorter candidate"
            },
            
            # Different length (longer candidate)
            {
                "reference": "Normal chest radiograph.",
                "candidate": "Normal chest radiograph with clear lung fields and normal heart size.",
                "description": "Longer candidate"
            }
        ]
        
        print("\n--- ROUGE Score Tests ---")
        for i, test_case in enumerate(test_cases, 1):
            scores = rouge.calculate(test_case["reference"], test_case["candidate"])
            
            print(f"\nTest {i}: {test_case['description']}")
            print(f"Reference: '{test_case['reference']}'")
            print(f"Candidate: '{test_case['candidate']}'")
            print(f"ROUGE-1 F1: {scores['rouge_1_f1']:.4f}")
            print(f"ROUGE-2 F1: {scores['rouge_2_f1']:.4f}")
            print(f"ROUGE-L F1: {scores['rouge_l_f1']:.4f}")
            print(f"Average F1: {scores['rouge_avg_f1']:.4f}")
            print(f"Length ratio: {scores['length_ratio']:.4f}")
        
        # Test ROUGE-W
        print("\n--- ROUGE-W Test ---")
        ref_text = "Normal chest radiograph with clear lung fields."
        cand_text = "Normal chest X-ray with clear lung areas."
        
        rouge_w_scores = rouge.calculate_rouge_w(ref_text, cand_text)
        print(f"ROUGE-W F1: {rouge_w_scores['rouge_w_f1']:.4f}")
        print(f"ROUGE-W Precision: {rouge_w_scores['rouge_w_precision']:.4f}")
        print(f"ROUGE-W Recall: {rouge_w_scores['rouge_w_recall']:.4f}")
        
        # Test with stopword removal
        print("\n--- Stopword Removal Test ---")
        rouge_no_stop = ROUGEScorer(remove_stopwords=True, logger=logger)
        
        ref_with_stops = "The chest radiograph shows normal lung fields and the heart is normal."
        cand_with_stops = "Chest radiograph shows normal lung fields and heart is normal."
        
        scores_with_stops = rouge.calculate(ref_with_stops, cand_with_stops)
        scores_no_stops = rouge_no_stop.calculate(ref_with_stops, cand_with_stops)
        
        print(f"With stopwords - ROUGE-1 F1: {scores_with_stops['rouge_1_f1']:.4f}")
        print(f"Without stopwords - ROUGE-1 F1: {scores_no_stops['rouge_1_f1']:.4f}")
        
        # Test batch calculation
        print("\n--- Batch Calculation Test ---")
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
        
        batch_scores = rouge.calculate_batch(ref_list, cand_list)
        print(f"Batch results: {len(batch_scores)} scores calculated")
        
        # Calculate average scores
        avg_rouge_1 = sum(score.get('rouge_1_f1', 0) for score in batch_scores) / len(batch_scores)
        avg_rouge_l = sum(score.get('rouge_l_f1', 0) for score in batch_scores) / len(batch_scores)
        
        print(f"Average ROUGE-1 F1: {avg_rouge_1:.4f}")
        print(f"Average ROUGE-L F1: {avg_rouge_l:.4f}")
        
        # Test edge cases
        print("\n--- Edge Cases Test ---")
        
        # Empty strings
        try:
            empty_scores = rouge.calculate("", "")
            print("❌ Empty strings should raise error")
        except ValueError:
            print("✅ Empty strings correctly handled")
        
        # Very short texts
        short_scores = rouge.calculate("Normal.", "Normal.")
        print(f"✅ Short text ROUGE-1 F1: {short_scores['rouge_1_f1']:.4f}")
        
        # Test performance
        print("\n--- Performance Test ---")
        perf_stats = rouge.get_performance_stats()
        print(f"Calculations performed: {perf_stats['calculation_count']}")
        print(f"Average calculation time: {perf_stats['average_time']:.4f}s")
        
        print("\n✅ All ROUGE scorer tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    success = test_rouge_scorer()
    
    if success:
        print("\n🎉 ROUGE Scorer tests passed!")
    else:
        print("\n❌ Some tests failed!") 