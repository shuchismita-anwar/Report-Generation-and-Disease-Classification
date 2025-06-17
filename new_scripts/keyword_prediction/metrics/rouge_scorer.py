"""
ROUGE Scorer for Medical Keywords

Implements ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scoring 
for medical keyword generation. Optimized for medical domain terminology.

Task 5.4: Add training metrics (BLEU/ROUGE on train and val)
"""

import torch
from rouge_score import rouge_scorer
from typing import List, Dict, Any
import re
import logging

# Set up logging
logger = logging.getLogger(__name__)

class ROUGEScorer:
    """
    ROUGE scorer for medical keyword evaluation.
    
    Provides ROUGE-1, ROUGE-2, ROUGE-L scores optimized for medical terminology.
    Focuses on recall which is critical for medical keyword completeness.
    """
    
    def __init__(self, rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL']):
        """
        Initialize ROUGE scorer.
        
        Args:
            rouge_types: List of ROUGE metrics to compute
        """
        self.rouge_types = rouge_types
        self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        
        # Medical preprocessing patterns
        self.medical_patterns = [
            (r'\bno\s+acute\s+', ''),  # Remove "no acute" patterns
            (r'\bno\s+', 'negative_'),  # Convert "no" to negative marker
            (r'\bmild\s+', 'mild_'),  # Preserve severity modifiers
            (r'\bmoderate\s+', 'moderate_'),
            (r'\bsevere\s+', 'severe_'),
            (r'\bbilateral\s+', 'bilateral_'),
            (r'\bunilateral\s+', 'unilateral_'),
            (r'\bchronic\s+', 'chronic_'),
            (r'\bacute\s+', 'acute_'),
        ]
        
        logger.info(f"ROUGE scorer initialized with metrics: {rouge_types}")
    
    def preprocess_medical_text(self, text: str) -> str:
        """
        Preprocess medical text for ROUGE evaluation.
        
        Args:
            text: Medical keyword string
            
        Returns:
            Preprocessed text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Apply medical-specific preprocessing
        for pattern, replacement in self.medical_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Normalize punctuation and spacing
        text = re.sub(r'[,;]+', ' ', text)  # Replace commas/semicolons with spaces
        text = re.sub(r'\s+', ' ', text)    # Normalize whitespace
        text = text.strip()
        
        return text
    
    def compute_rouge_scores(self, reference: str, hypothesis: str) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE scores for a single reference-hypothesis pair.
        
        Args:
            reference: Ground truth keywords
            hypothesis: Generated keywords
            
        Returns:
            Dictionary with ROUGE scores {metric: {precision, recall, fmeasure}}
        """
        try:
            # Preprocess texts
            ref_processed = self.preprocess_medical_text(reference)
            hyp_processed = self.preprocess_medical_text(hypothesis)
            
            if not ref_processed or not hyp_processed:
                # Return zero scores for empty inputs
                return {metric: {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0} 
                        for metric in self.rouge_types}
            
            # Compute ROUGE scores
            scores = self.scorer.score(ref_processed, hyp_processed)
            
            # Convert to standard dictionary format
            result = {}
            for metric in self.rouge_types:
                if metric in scores:
                    result[metric] = {
                        'precision': scores[metric].precision,
                        'recall': scores[metric].recall,
                        'fmeasure': scores[metric].fmeasure
                    }
                else:
                    result[metric] = {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
            
            return result
            
        except Exception as e:
            logger.warning(f"Error computing ROUGE scores: {e}")
            return {metric: {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0} 
                    for metric in self.rouge_types}
    
    def compute_batch_metrics(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute average ROUGE metrics for a batch.
        
        Args:
            references: Batch of ground truth keywords
            hypotheses: Batch of generated keywords
            
        Returns:
            Dictionary with averaged ROUGE metrics
        """
        try:
            if not references or not hypotheses or len(references) != len(hypotheses):
                logger.error(f"Invalid input: {len(references)} refs, {len(hypotheses)} hyps")
                return self._get_zero_metrics()
            
            # Collect scores for all pairs
            all_scores = []
            valid_pairs = 0
            
            for ref, hyp in zip(references, hypotheses):
                scores = self.compute_rouge_scores(ref, hyp)
                all_scores.append(scores)
                
                # Check if this pair has valid scores
                if any(scores[metric]['fmeasure'] > 0 for metric in self.rouge_types):
                    valid_pairs += 1
            
            if valid_pairs == 0:
                logger.warning("No valid pairs found for ROUGE computation")
                return self._get_zero_metrics()
            
            # Average the scores
            averaged_metrics = {}
            
            for metric in self.rouge_types:
                precision_sum = sum(score[metric]['precision'] for score in all_scores)
                recall_sum = sum(score[metric]['recall'] for score in all_scores)
                fmeasure_sum = sum(score[metric]['fmeasure'] for score in all_scores)
                
                averaged_metrics[f'{metric}_precision'] = precision_sum / len(all_scores)
                averaged_metrics[f'{metric}_recall'] = recall_sum / len(all_scores)
                averaged_metrics[f'{metric}_fmeasure'] = fmeasure_sum / len(all_scores)
            
            # Add summary metrics
            averaged_metrics['rouge_avg_recall'] = sum(
                averaged_metrics[f'{metric}_recall'] for metric in self.rouge_types
            ) / len(self.rouge_types)
            
            averaged_metrics['rouge_avg_fmeasure'] = sum(
                averaged_metrics[f'{metric}_fmeasure'] for metric in self.rouge_types
            ) / len(self.rouge_types)
            
            logger.debug(f"ROUGE metrics computed for {len(all_scores)} pairs, {valid_pairs} valid")
            return averaged_metrics
            
        except Exception as e:
            logger.error(f"Error computing batch ROUGE metrics: {e}")
            return self._get_zero_metrics()
    
    def _get_zero_metrics(self) -> Dict[str, float]:
        """Get zero-initialized metrics dictionary"""
        metrics = {}
        for metric in self.rouge_types:
            metrics[f'{metric}_precision'] = 0.0
            metrics[f'{metric}_recall'] = 0.0
            metrics[f'{metric}_fmeasure'] = 0.0
        
        metrics['rouge_avg_recall'] = 0.0
        metrics['rouge_avg_fmeasure'] = 0.0
        
        return metrics
    
    def compute_detailed_metrics(self, references: List[str], hypotheses: List[str]) -> Dict[str, Any]:
        """
        Compute detailed ROUGE metrics with individual scores.
        
        Args:
            references: Ground truth keywords
            hypotheses: Generated keywords
            
        Returns:
            Detailed metrics dictionary
        """
        try:
            # Get batch averages
            batch_metrics = self.compute_batch_metrics(references, hypotheses)
            
            # Compute individual scores for analysis
            individual_scores = []
            for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
                scores = self.compute_rouge_scores(ref, hyp)
                individual_scores.append({
                    'index': i,
                    'reference': ref,
                    'hypothesis': hyp,
                    'scores': scores
                })
            
            # Find best and worst performing examples
            fmeasure_scores = [
                sum(score['scores'][metric]['fmeasure'] for metric in self.rouge_types) 
                for score in individual_scores
            ]
            
            best_idx = fmeasure_scores.index(max(fmeasure_scores)) if fmeasure_scores else 0
            worst_idx = fmeasure_scores.index(min(fmeasure_scores)) if fmeasure_scores else 0
            
            detailed_metrics = {
                'batch_averages': batch_metrics,
                'individual_scores': individual_scores,
                'best_example': individual_scores[best_idx] if individual_scores else None,
                'worst_example': individual_scores[worst_idx] if individual_scores else None,
                'total_pairs': len(references),
                'valid_pairs': sum(1 for score in fmeasure_scores if score > 0)
            }
            
            return detailed_metrics
            
        except Exception as e:
            logger.error(f"Error computing detailed ROUGE metrics: {e}")
            return {
                'batch_averages': self._get_zero_metrics(),
                'individual_scores': [],
                'best_example': None,
                'worst_example': None,
                'total_pairs': 0,
                'valid_pairs': 0
            }


def test_rouge_scorer():
    """Test ROUGE scorer with medical examples"""
    
    print("="*60)
    print("TESTING ROUGE SCORER FOR MEDICAL KEYWORDS")
    print("="*60)
    
    # Initialize scorer
    scorer = ROUGEScorer()
    
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
            'description': 'Partial match (should have high recall)'
        },
        {
            'reference': 'no acute cardiopulmonary process',
            'hypothesis': 'normal chest',
            'description': 'Different phrasing, same meaning'
        },
        {
            'reference': 'mild cardiomegaly, chronic changes',
            'hypothesis': 'moderate heart enlargement, old findings',
            'description': 'Different severity, similar content'
        }
    ]
    
    print("📊 Individual Test Cases:")
    for i, case in enumerate(test_cases, 1):
        scores = scorer.compute_rouge_scores(case['reference'], case['hypothesis'])
        print(f"   {i}. {case['description']}")
        print(f"      Reference: '{case['reference']}'")
        print(f"      Hypothesis: '{case['hypothesis']}'")
        print(f"      ROUGE-1 F1: {scores['rouge1']['fmeasure']:.4f}")
        print(f"      ROUGE-2 F1: {scores['rouge2']['fmeasure']:.4f}")
        print(f"      ROUGE-L F1: {scores['rougeL']['fmeasure']:.4f}")
        print(f"      ROUGE-1 Recall: {scores['rouge1']['recall']:.4f}")
        print()
    
    # Test batch metrics
    references = [case['reference'] for case in test_cases]
    hypotheses = [case['hypothesis'] for case in test_cases]
    
    print("📈 Batch Metrics:")
    batch_metrics = scorer.compute_batch_metrics(references, hypotheses)
    for metric, value in batch_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Test detailed metrics
    print("\n📋 Detailed Analysis:")
    detailed = scorer.compute_detailed_metrics(references, hypotheses)
    print(f"   Total pairs: {detailed['total_pairs']}")
    print(f"   Valid pairs: {detailed['valid_pairs']}")
    
    if detailed['best_example']:
        best = detailed['best_example']
        print(f"   Best example (idx {best['index']}):")
        print(f"      Ref: '{best['reference']}'")
        print(f"      Hyp: '{best['hypothesis']}'")
        print(f"      ROUGE-1 F1: {best['scores']['rouge1']['fmeasure']:.4f}")
    
    print(f"\n✅ ROUGE scorer test completed!")
    return True


if __name__ == "__main__":
    test_rouge_scorer() 