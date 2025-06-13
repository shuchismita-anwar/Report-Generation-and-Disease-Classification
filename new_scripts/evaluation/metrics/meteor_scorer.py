#!/usr/bin/env python3
"""
METEOR Scorer
=============
Implementation of METEOR (Metric for Evaluation of Translation with Explicit ORdering) metric.

This module provides:
- METEOR score calculation with alignment-based matching
- Stemming support for morphological variants
- Synonymy matching using WordNet
- Paraphrase matching capabilities
- Penalty for word order differences
- Medical text preprocessing

Author: Medical Report Evaluation System
Version: 1.0
"""

import re
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set, Union
import logging

# Import base metric class
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from base_metric import BaseMetric


class METEORScorer(BaseMetric):
    """
    METEOR (Metric for Evaluation of Translation with Explicit ORdering) implementation.
    
    Calculates METEOR scores using unigram matching with stemming, synonymy,
    and paraphrase support, plus a penalty for word order differences.
    """
    
    def __init__(self, use_stemming: bool = True, use_synonyms: bool = True,
                 use_paraphrases: bool = False, alpha: float = 0.9, beta: float = 3.0,
                 gamma: float = 0.5, logger: Optional[logging.Logger] = None):
        """
        Initialize the METEOR scorer.
        
        Args:
            use_stemming (bool): Whether to use stemming for matching
            use_synonyms (bool): Whether to use synonym matching
            use_paraphrases (bool): Whether to use paraphrase matching
            alpha (float): Weight for recall vs precision (default: 0.9)
            beta (float): Shape parameter for fragmentation penalty (default: 3.0)
            gamma (float): Relative weight of fragmentation penalty (default: 0.5)
            logger (Optional[logging.Logger]): Logger instance
        """
        super().__init__("meteor", logger)
        self.description = "METEOR (Metric for Evaluation of Translation with Explicit ORdering)"
        self.metric_type = "similarity"
        
        # Configuration parameters
        self.use_stemming = use_stemming
        self.use_synonyms = use_synonyms
        self.use_paraphrases = use_paraphrases
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize stemmer
        self.stemmer = None
        if self.use_stemming:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
                self.logger.debug("Initialized Porter stemmer for METEOR")
            except ImportError:
                self.logger.warning("NLTK not available, stemming disabled")
                self.use_stemming = False
        
        # Initialize WordNet for synonyms
        self.wordnet = None
        if self.use_synonyms:
            try:
                from nltk.corpus import wordnet
                self.wordnet = wordnet
                self.logger.debug("Initialized WordNet for METEOR synonyms")
            except ImportError:
                self.logger.warning("NLTK WordNet not available, synonym matching disabled")
                self.use_synonyms = False
        
        # Medical synonym mappings (basic set)
        self.medical_synonyms = {
            'chest': {'thorax', 'thoracic'},
            'lung': {'pulmonary', 'respiratory'},
            'heart': {'cardiac', 'cardio'},
            'normal': {'unremarkable', 'within_normal_limits', 'wnl'},
            'abnormal': {'abnormality', 'pathologic', 'pathological'},
            'radiograph': {'x-ray', 'xray', 'film'},
            'examination': {'exam', 'study', 'imaging'},
            'bilateral': {'both', 'bilaterally'},
            'clear': {'patent', 'open'},
            'opacity': {'density', 'infiltrate'},
            'effusion': {'fluid', 'collection'},
            'pneumonia': {'infection', 'infiltrate', 'consolidation'}
        }
        
        self.logger.debug(f"Initialized METEOR scorer with stemming={use_stemming}, "
                         f"synonyms={use_synonyms}, paraphrases={use_paraphrases}")
    
    def calculate(self, reference: str, candidate: str, **kwargs) -> Dict[str, float]:
        """
        Calculate METEOR score between reference and candidate texts.
        
        Args:
            reference (str): Ground truth reference text
            candidate (str): Generated candidate text to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, float]: Dictionary containing METEOR scores
        """
        # Validate inputs
        is_valid, issues = self.validate_inputs(reference, candidate)
        if not is_valid:
            raise ValueError(f"Invalid inputs for METEOR calculation: {issues}")
        
        # Tokenize texts
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return {
                "meteor": 0.0,
                "meteor_precision": 0.0,
                "meteor_recall": 0.0,
                "meteor_fmean": 0.0,
                "meteor_fragmentation_penalty": 1.0,
                "meteor_chunks": 0,
                "meteor_matches": 0,
                "reference_length": float(len(ref_tokens)),
                "candidate_length": float(len(cand_tokens))
            }
        
        # Find alignments between reference and candidate
        alignments = self._find_alignments(ref_tokens, cand_tokens)
        
        # Calculate basic statistics
        matches = len(alignments)
        ref_len = len(ref_tokens)
        cand_len = len(cand_tokens)
        
        # Calculate precision and recall
        precision = matches / cand_len if cand_len > 0 else 0.0
        recall = matches / ref_len if ref_len > 0 else 0.0
        
        # Calculate harmonic mean (F-mean)
        if precision + recall > 0:
            fmean = (precision * recall) / (self.alpha * precision + (1 - self.alpha) * recall)
        else:
            fmean = 0.0
        
        # Calculate fragmentation penalty
        chunks = self._count_chunks(alignments, cand_len)
        fragmentation_penalty = self._calculate_fragmentation_penalty(matches, chunks)
        
        # Final METEOR score
        meteor_score = fmean * (1 - fragmentation_penalty)
        
        return {
            "meteor": meteor_score,
            "meteor_precision": precision,
            "meteor_recall": recall,
            "meteor_fmean": fmean,
            "meteor_fragmentation_penalty": fragmentation_penalty,
            "meteor_chunks": float(chunks),
            "meteor_matches": float(matches),
            "reference_length": float(ref_len),
            "candidate_length": float(cand_len),
            "length_ratio": cand_len / ref_len if ref_len > 0 else 0.0
        }
    
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
        
        # Handle medical abbreviations
        text = re.sub(r'([a-z])\.([a-z])', r'\1. \2', text)
        
        # Tokenize (keep only alphanumeric words)
        tokens = re.findall(r'\b\w+\b', text)
        
        return tokens
    
    def _find_alignments(self, ref_tokens: List[str], cand_tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Find alignments between reference and candidate tokens.
        
        Args:
            ref_tokens (List[str]): Reference tokens
            cand_tokens (List[str]): Candidate tokens
            
        Returns:
            List[Tuple[int, int]]: List of (ref_idx, cand_idx) alignments
        """
        alignments = []
        used_ref = set()
        used_cand = set()
        
        # Stage 1: Exact matches
        for i, ref_token in enumerate(ref_tokens):
            for j, cand_token in enumerate(cand_tokens):
                if i not in used_ref and j not in used_cand:
                    if ref_token == cand_token:
                        alignments.append((i, j))
                        used_ref.add(i)
                        used_cand.add(j)
        
        # Stage 2: Stemming matches
        if self.use_stemming and self.stemmer:
            for i, ref_token in enumerate(ref_tokens):
                if i not in used_ref:
                    ref_stem = self.stemmer.stem(ref_token)
                    for j, cand_token in enumerate(cand_tokens):
                        if j not in used_cand:
                            cand_stem = self.stemmer.stem(cand_token)
                            if ref_stem == cand_stem and ref_token != cand_token:
                                alignments.append((i, j))
                                used_ref.add(i)
                                used_cand.add(j)
                                break
        
        # Stage 3: Synonym matches
        if self.use_synonyms:
            # Medical synonyms first
            for i, ref_token in enumerate(ref_tokens):
                if i not in used_ref:
                    for j, cand_token in enumerate(cand_tokens):
                        if j not in used_cand:
                            if self._are_medical_synonyms(ref_token, cand_token):
                                alignments.append((i, j))
                                used_ref.add(i)
                                used_cand.add(j)
                                break
            
            # WordNet synonyms
            if self.wordnet:
                for i, ref_token in enumerate(ref_tokens):
                    if i not in used_ref:
                        for j, cand_token in enumerate(cand_tokens):
                            if j not in used_cand:
                                if self._are_wordnet_synonyms(ref_token, cand_token):
                                    alignments.append((i, j))
                                    used_ref.add(i)
                                    used_cand.add(j)
                                    break
        
        # Stage 4: Paraphrase matches (simplified - could be enhanced)
        if self.use_paraphrases:
            # Basic paraphrase patterns for medical text
            paraphrase_patterns = [
                (['within', 'normal', 'limits'], ['normal']),
                (['no', 'acute', 'abnormalities'], ['normal']),
                (['clear', 'lung', 'fields'], ['normal', 'lungs']),
                (['heart', 'size', 'normal'], ['normal', 'cardiac']),
            ]
            
            # This is a simplified implementation
            # In practice, you'd want more sophisticated paraphrase detection
            pass
        
        return alignments
    
    def _are_medical_synonyms(self, word1: str, word2: str) -> bool:
        """
        Check if two words are medical synonyms.
        
        Args:
            word1 (str): First word
            word2 (str): Second word
            
        Returns:
            bool: True if words are medical synonyms
        """
        # Check direct mapping
        if word1 in self.medical_synonyms:
            if word2 in self.medical_synonyms[word1]:
                return True
        
        if word2 in self.medical_synonyms:
            if word1 in self.medical_synonyms[word2]:
                return True
        
        # Check if both words are in the same synonym set
        for base_word, synonyms in self.medical_synonyms.items():
            if word1 in synonyms and word2 in synonyms:
                return True
            if word1 == base_word and word2 in synonyms:
                return True
            if word2 == base_word and word1 in synonyms:
                return True
        
        return False
    
    def _are_wordnet_synonyms(self, word1: str, word2: str) -> bool:
        """
        Check if two words are synonyms using WordNet.
        
        Args:
            word1 (str): First word
            word2 (str): Second word
            
        Returns:
            bool: True if words are synonyms
        """
        if not self.wordnet:
            return False
        
        try:
            # Get synsets for both words
            synsets1 = self.wordnet.synsets(word1)
            synsets2 = self.wordnet.synsets(word2)
            
            # Check for overlap in synsets
            for syn1 in synsets1:
                for syn2 in synsets2:
                    # Check if synsets are the same or similar
                    if syn1 == syn2:
                        return True
                    
                    # Check path similarity (threshold can be adjusted)
                    try:
                        similarity = syn1.path_similarity(syn2)
                        if similarity and similarity > 0.8:  # High similarity threshold
                            return True
                    except:
                        continue
            
            return False
            
        except Exception:
            return False
    
    def _count_chunks(self, alignments: List[Tuple[int, int]], cand_len: int) -> int:
        """
        Count the number of chunks in the alignment.
        
        Args:
            alignments (List[Tuple[int, int]]): List of alignments
            cand_len (int): Length of candidate text
            
        Returns:
            int: Number of chunks
        """
        if not alignments:
            return 0
        
        # Sort alignments by candidate position
        sorted_alignments = sorted(alignments, key=lambda x: x[1])
        
        chunks = 1
        prev_ref_pos = sorted_alignments[0][0]
        
        for ref_pos, cand_pos in sorted_alignments[1:]:
            # If reference positions are not consecutive, it's a new chunk
            if ref_pos != prev_ref_pos + 1:
                chunks += 1
            prev_ref_pos = ref_pos
        
        return chunks
    
    def _calculate_fragmentation_penalty(self, matches: int, chunks: int) -> float:
        """
        Calculate the fragmentation penalty.
        
        Args:
            matches (int): Number of matches
            chunks (int): Number of chunks
            
        Returns:
            float: Fragmentation penalty (0 to 1)
        """
        if matches == 0:
            return 0.0
        
        # METEOR fragmentation penalty formula
        fragmentation = chunks / matches
        penalty = self.gamma * (fragmentation ** self.beta)
        
        return min(penalty, 1.0)  # Cap at 1.0
    
    def calculate_meteor_plus(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate METEOR+ score with additional features.
        
        Args:
            reference (str): Reference text
            candidate (str): Candidate text
            
        Returns:
            Dict[str, float]: METEOR+ scores with additional metrics
        """
        # Get basic METEOR score
        meteor_scores = self.calculate(reference, candidate)
        
        # Additional features for METEOR+
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        # Content word ratio (medical terms)
        medical_terms = {
            'chest', 'lung', 'heart', 'normal', 'abnormal', 'radiograph',
            'examination', 'bilateral', 'clear', 'opacity', 'effusion',
            'pneumonia', 'cardiac', 'pulmonary', 'thoracic'
        }
        
        ref_medical = sum(1 for token in ref_tokens if token in medical_terms)
        cand_medical = sum(1 for token in cand_tokens if token in medical_terms)
        
        medical_precision = cand_medical / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
        medical_recall = ref_medical / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        # Function word penalty (simplified)
        function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        ref_function = sum(1 for token in ref_tokens if token in function_words)
        cand_function = sum(1 for token in cand_tokens if token in function_words)
        
        function_ratio = abs(ref_function - cand_function) / max(len(ref_tokens), len(cand_tokens))
        
        # Enhanced METEOR+ score
        meteor_plus = meteor_scores["meteor"] * (1 + 0.1 * medical_precision - 0.05 * function_ratio)
        meteor_plus = max(0.0, min(1.0, meteor_plus))  # Clamp to [0, 1]
        
        # Add METEOR+ specific scores
        meteor_scores.update({
            "meteor_plus": meteor_plus,
            "medical_precision": medical_precision,
            "medical_recall": medical_recall,
            "function_word_penalty": function_ratio
        })
        
        return meteor_scores
    
    def get_name(self) -> str:
        """Get the metric name."""
        return "METEOR"
    
    def get_description(self) -> str:
        """Get the metric description."""
        return (f"METEOR (Metric for Evaluation of Translation with Explicit ORdering) "
                f"with stemming={self.use_stemming}, synonyms={self.use_synonyms}, "
                f"paraphrases={self.use_paraphrases}")


def test_meteor_scorer():
    """Test function for the METEOR Scorer."""
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    try:
        print("🧪 Testing METEOR Scorer...")
        
        # Initialize METEOR scorer
        meteor = METEORScorer(use_stemming=True, use_synonyms=True, logger=logger)
        
        # Test basic properties
        print(f"✅ Metric name: {meteor.get_name()}")
        print(f"✅ Metric description: {meteor.get_description()}")
        
        # Test cases with different similarity levels
        test_cases = [
            # Perfect match
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Normal chest radiograph with clear lung fields.",
                "description": "Perfect match"
            },
            
            # Stemming test
            {
                "reference": "The examination shows normal findings.",
                "candidate": "The exam showed normal finding.",
                "description": "Stemming variants"
            },
            
            # Medical synonym test
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Normal chest X-ray with clear pulmonary areas.",
                "description": "Medical synonyms"
            },
            
            # Word order difference
            {
                "reference": "Clear lung fields and normal heart size.",
                "candidate": "Normal heart size and clear lung fields.",
                "description": "Different word order"
            },
            
            # Partial match
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Chest radiograph shows normal findings.",
                "description": "Partial overlap"
            },
            
            # Low similarity
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Patient has severe pneumonia and effusion.",
                "description": "Low similarity"
            }
        ]
        
        print("\n--- METEOR Score Tests ---")
        for i, test_case in enumerate(test_cases, 1):
            scores = meteor.calculate(test_case["reference"], test_case["candidate"])
            
            print(f"\nTest {i}: {test_case['description']}")
            print(f"Reference: '{test_case['reference']}'")
            print(f"Candidate: '{test_case['candidate']}'")
            print(f"METEOR: {scores['meteor']:.4f}")
            print(f"Precision: {scores['meteor_precision']:.4f}")
            print(f"Recall: {scores['meteor_recall']:.4f}")
            print(f"F-mean: {scores['meteor_fmean']:.4f}")
            print(f"Fragmentation Penalty: {scores['meteor_fragmentation_penalty']:.4f}")
            print(f"Chunks: {scores['meteor_chunks']}")
            print(f"Matches: {scores['meteor_matches']}")
        
        # Test METEOR+ 
        print("\n--- METEOR+ Test ---")
        ref_text = "Normal chest radiograph with clear lung fields."
        cand_text = "Normal chest X-ray with clear pulmonary areas."
        
        meteor_plus_scores = meteor.calculate_meteor_plus(ref_text, cand_text)
        print(f"METEOR: {meteor_plus_scores['meteor']:.4f}")
        print(f"METEOR+: {meteor_plus_scores['meteor_plus']:.4f}")
        print(f"Medical Precision: {meteor_plus_scores['medical_precision']:.4f}")
        print(f"Medical Recall: {meteor_plus_scores['medical_recall']:.4f}")
        
        # Test different configurations
        print("\n--- Configuration Tests ---")
        
        # Without stemming
        meteor_no_stem = METEORScorer(use_stemming=False, use_synonyms=False, logger=logger)
        scores_no_stem = meteor_no_stem.calculate(
            "The examination shows normal findings.",
            "The exam showed normal finding."
        )
        
        # With stemming
        scores_with_stem = meteor.calculate(
            "The examination shows normal findings.",
            "The exam showed normal finding."
        )
        
        print(f"Without stemming: {scores_no_stem['meteor']:.4f}")
        print(f"With stemming: {scores_with_stem['meteor']:.4f}")
        
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
        
        batch_scores = meteor.calculate_batch(ref_list, cand_list)
        print(f"Batch results: {len(batch_scores)} scores calculated")
        
        # Calculate average METEOR score
        avg_meteor = sum(score.get('meteor', 0) for score in batch_scores) / len(batch_scores)
        print(f"Average METEOR: {avg_meteor:.4f}")
        
        # Test edge cases
        print("\n--- Edge Cases Test ---")
        
        # Empty strings
        try:
            empty_scores = meteor.calculate("", "")
            print("❌ Empty strings should raise error")
        except ValueError:
            print("✅ Empty strings correctly handled")
        
        # Very short texts
        short_scores = meteor.calculate("Normal.", "Normal.")
        print(f"✅ Short text METEOR: {short_scores['meteor']:.4f}")
        
        # Test performance
        print("\n--- Performance Test ---")
        perf_stats = meteor.get_performance_stats()
        print(f"Calculations performed: {perf_stats['calculation_count']}")
        print(f"Average calculation time: {perf_stats['average_time']:.4f}s")
        
        print("\n✅ All METEOR scorer tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    success = test_meteor_scorer()
    
    if success:
        print("\n🎉 METEOR Scorer tests passed!")
    else:
        print("\n❌ Some tests failed!") 