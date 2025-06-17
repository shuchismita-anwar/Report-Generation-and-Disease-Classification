"""
BLEU Scorer for Medical Keywords

Implements BLEU (Bilingual Evaluation Understudy) scoring for medical keyword generation.
Adapted for medical domain with preprocessing for clinical terminology.

Task 5.4: Add training metrics (BLEU/ROUGE on train and val)
"""

import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from typing import List, Dict, Any, Tuple
import re
import logging

# Set up logging
logger = logging.getLogger(__name__)

class BLEUScorer:
    """
    BLEU scorer for medical keyword evaluation.
    
    Handles medical terminology preprocessing and provides both sentence-level
    and corpus-level BLEU scores for medical keyword generation.
    """
    
    def __init__(self, weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)):
        """
        Initialize BLEU scorer.
        
        Args:
            weights: N-gram weights for BLEU calculation (default: equal weights for 1-4 grams)
        """
        self.weights = weights
        self.smoothing = SmoothingFunction().method1  # Smoothing for short sequences
        
        # Medical keyword preprocessing patterns
        self.medical_patterns = [
            (r'\bno\s+', ''),  # Remove "no " prefix
            (r'\bmild\s+', 'mild_'),  # Preserve mild as modifier
            (r'\bmoderate\s+', 'moderate_'),  # Preserve moderate as modifier
            (r'\bsevere\s+', 'severe_'),  # Preserve severe as modifier
            (r'\bbilateral\s+', 'bilateral_'),  # Preserve bilateral as modifier
            (r'\bunilateral\s+', 'unilateral_'),  # Preserve unilateral as modifier
        ]
        
        logger.info(f"BLEU scorer initialized with weights: {weights}")
    
    def preprocess_medical_text(self, text: str) -> List[str]:
        """
        Preprocess medical text for BLEU evaluation.
        
        Args:
            text: Medical keyword string
            
        Returns:
            List of preprocessed tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Apply medical-specific preprocessing
        for pattern, replacement in self.medical_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Split on common delimiters in medical keywords
        tokens = re.split(r'[,;\s]+', text)
        
        # Filter out empty tokens and common stopwords
        stopwords = {'and', 'or', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for'}
        tokens = [token.strip() for token in tokens if token.strip() and token.strip() not in stopwords]
        
        return tokens
    
    def compute_sentence_bleu(self, reference: str, hypothesis: str) -> float:
        """
        Compute sentence-level BLEU score.
        
        Args:
            reference: Ground truth keywords
            hypothesis: Generated keywords
            
        Returns:
            BLEU score (0.0 to 1.0)
        """
        try:
            # Preprocess both reference and hypothesis
            ref_tokens = self.preprocess_medical_text(reference)
            hyp_tokens = self.preprocess_medical_text(hypothesis)
            
            if not ref_tokens or not hyp_tokens:
                return 0.0
            
            # BLEU expects list of reference lists
            references = [ref_tokens]
            
            # Compute BLEU with smoothing
            score = sentence_bleu(
                references, 
                hyp_tokens, 
                weights=self.weights,
                smoothing_function=self.smoothing
            )
            
            return score
            
        except Exception as e:
            logger.warning(f"Error computing sentence BLEU: {e}")
            return 0.0
    
    def compute_corpus_bleu(self, references: List[str], hypotheses: List[str]) -> float:
        """
        Compute corpus-level BLEU score.
        
        Args:
            references: List of ground truth keyword strings
            hypotheses: List of generated keyword strings
            
        Returns:
            Corpus BLEU score (0.0 to 1.0)
        """
        try:
            if len(references) != len(hypotheses):
                logger.error(f"Length mismatch: {len(references)} references vs {len(hypotheses)} hypotheses")
                return 0.0
            
            # Preprocess all references and hypotheses
            processed_refs = []
            processed_hyps = []
            
            for ref, hyp in zip(references, hypotheses):
                ref_tokens = self.preprocess_medical_text(ref)
                hyp_tokens = self.preprocess_medical_text(hyp)
                
                if ref_tokens and hyp_tokens:  # Only include non-empty pairs
                    processed_refs.append([ref_tokens])  # BLEU expects list of reference lists
                    processed_hyps.append(hyp_tokens)
            
            if not processed_refs or not processed_hyps:
                logger.warning("No valid reference-hypothesis pairs for corpus BLEU")
                return 0.0
            
            # Compute corpus BLEU
            score = corpus_bleu(
                processed_refs, 
                processed_hyps, 
                weights=self.weights,
                smoothing_function=self.smoothing
            )
            
            return score
            
        except Exception as e:
            logger.error(f"Error computing corpus BLEU: {e}")
            return 0.0
    
    def compute_batch_metrics(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute comprehensive BLEU metrics for a batch.
        
        Args:
            references: Batch of ground truth keywords
            hypotheses: Batch of generated keywords
            
        Returns:
            Dictionary with BLEU metrics
        """
        try:
            if not references or not hypotheses:
                return {'bleu_corpus': 0.0, 'bleu_avg': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
            
            # Corpus-level BLEU
            corpus_bleu = self.compute_corpus_bleu(references, hypotheses)
            
            # Average sentence-level BLEU
            sentence_bleus = []
            for ref, hyp in zip(references, hypotheses):
                score = self.compute_sentence_bleu(ref, hyp)
                sentence_bleus.append(score)
            
            avg_bleu = sum(sentence_bleus) / len(sentence_bleus) if sentence_bleus else 0.0
            
            # Individual n-gram BLEU scores
            bleu_1 = self._compute_ngram_bleu(references, hypotheses, (1.0,))
            bleu_2 = self._compute_ngram_bleu(references, hypotheses, (0.5, 0.5))
            bleu_3 = self._compute_ngram_bleu(references, hypotheses, (0.33, 0.33, 0.34))
            bleu_4 = corpus_bleu  # 4-gram is our default corpus BLEU
            
            metrics = {
                'bleu_corpus': corpus_bleu,
                'bleu_avg': avg_bleu,
                'bleu_1': bleu_1,
                'bleu_2': bleu_2,
                'bleu_3': bleu_3,
                'bleu_4': bleu_4
            }
            
            logger.debug(f"BLEU metrics computed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing batch BLEU metrics: {e}")
            return {'bleu_corpus': 0.0, 'bleu_avg': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
    
    def _compute_ngram_bleu(self, references: List[str], hypotheses: List[str], weights: Tuple[float, ...]) -> float:
        """
        Compute BLEU score with specific n-gram weights.
        
        Args:
            references: Ground truth keywords
            hypotheses: Generated keywords
            weights: N-gram weights
            
        Returns:
            BLEU score for specified n-grams
        """
        try:
            processed_refs = []
            processed_hyps = []
            
            for ref, hyp in zip(references, hypotheses):
                ref_tokens = self.preprocess_medical_text(ref)
                hyp_tokens = self.preprocess_medical_text(hyp)
                
                if ref_tokens and hyp_tokens:
                    processed_refs.append([ref_tokens])
                    processed_hyps.append(hyp_tokens)
            
            if not processed_refs or not processed_hyps:
                return 0.0
            
            score = corpus_bleu(
                processed_refs, 
                processed_hyps, 
                weights=weights,
                smoothing_function=self.smoothing
            )
            
            return score
            
        except Exception as e:
            logger.warning(f"Error computing {weights}-gram BLEU: {e}")
            return 0.0


def test_bleu_scorer():
    """Test BLEU scorer with medical examples"""
    
    print("="*60)
    print("TESTING BLEU SCORER FOR MEDICAL KEYWORDS")
    print("="*60)
    
    # Initialize scorer
    scorer = BLEUScorer()
    
    # Test cases with medical keywords
    test_cases = [
        {
            'reference': 'atelectasis, pneumonia, cardiomegaly',
            'hypothesis': 'atelectasis, pneumonia, enlarged heart',
            'description': 'Similar medical findings'
        },
        {
            'reference': 'bilateral pleural effusion',
            'hypothesis': 'bilateral effusion',
            'description': 'Partial match'
        },
        {
            'reference': 'no acute cardiopulmonary process',
            'hypothesis': 'normal chest',
            'description': 'Different phrasing, same meaning'
        },
        {
            'reference': 'mild cardiomegaly, chronic changes',
            'hypothesis': 'moderate heart enlargement',
            'description': 'Different severity terms'
        }
    ]
    
    print("📊 Individual Test Cases:")
    for i, case in enumerate(test_cases, 1):
        score = scorer.compute_sentence_bleu(case['reference'], case['hypothesis'])
        print(f"   {i}. {case['description']}")
        print(f"      Reference: '{case['reference']}'")
        print(f"      Hypothesis: '{case['hypothesis']}'")
        print(f"      BLEU Score: {score:.4f}")
        print()
    
    # Test batch metrics
    references = [case['reference'] for case in test_cases]
    hypotheses = [case['hypothesis'] for case in test_cases]
    
    print("📈 Batch Metrics:")
    batch_metrics = scorer.compute_batch_metrics(references, hypotheses)
    for metric, value in batch_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    print(f"\n✅ BLEU scorer test completed!")
    return True


if __name__ == "__main__":
    test_bleu_scorer() 