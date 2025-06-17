"""
Medical Keyword Metrics Coordinator

Combines BLEU, ROUGE, and medical-specific metrics for comprehensive evaluation
of medical keyword generation during training and validation.

Task 5.4: Add training metrics (BLEU/ROUGE on train and val)
"""

import torch
from typing import List, Dict, Any, Optional
import logging
import time
from .bleu_scorer import BLEUScorer
from .rouge_scorer import ROUGEScorer

# Set up logging
logger = logging.getLogger(__name__)

class MedicalKeywordMetrics:
    """
    Comprehensive metrics coordinator for medical keyword evaluation.
    
    Combines multiple evaluation metrics optimized for medical domain:
    - BLEU scores (precision-oriented)
    - ROUGE scores (recall-oriented) 
    - Medical-specific metrics
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize medical metrics coordinator.
        
        Args:
            device: Device for computations
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        
        # Initialize scorers
        self.bleu_scorer = BLEUScorer()
        self.rouge_scorer = ROUGEScorer()
        
        # Medical-specific metrics
        self.medical_keywords = self._load_medical_vocabulary()
        
        logger.info(f"Medical metrics initialized on device: {self.device}")
        logger.info(f"Medical vocabulary: {len(self.medical_keywords)} terms")
    
    def _load_medical_vocabulary(self) -> set:
        """
        Load medical vocabulary for domain-specific evaluation.
        
        Returns:
            Set of medical terms
        """
        # Core medical terminology for chest X-rays
        medical_terms = {
            # Anatomical structures
            'heart', 'lung', 'lungs', 'chest', 'thorax', 'pleura', 'pleural',
            'diaphragm', 'mediastinum', 'hilum', 'hila', 'aorta', 'cardiac',
            'cardiopulmonary', 'pulmonary', 'vascular', 'vasculature',
            
            # Pathological findings
            'atelectasis', 'pneumonia', 'effusion', 'pneumothorax', 'edema',
            'consolidation', 'opacity', 'opacities', 'infiltrate', 'infiltrates',
            'nodule', 'nodules', 'mass', 'masses', 'lesion', 'lesions',
            'cardiomegaly', 'enlargement', 'enlarged', 'prominent',
            
            # Severity descriptors
            'mild', 'moderate', 'severe', 'minimal', 'extensive', 'diffuse',
            'localized', 'focal', 'widespread', 'scattered',
            
            # Laterality
            'bilateral', 'unilateral', 'left', 'right', 'central', 'peripheral',
            
            # Temporal descriptors
            'acute', 'chronic', 'old', 'new', 'stable', 'worsening', 'improving',
            
            # Common negations and normal findings
            'normal', 'clear', 'unremarkable', 'negative', 'absence',
            'no', 'without', 'free'
        }
        
        return medical_terms
    
    def compute_training_metrics(self, 
                                references: List[str], 
                                hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute metrics optimized for training monitoring.
        
        Args:
            references: Ground truth keywords
            hypotheses: Generated keywords
            
        Returns:
            Dictionary with training metrics
        """
        try:
            start_time = time.time()
            
            # Core metrics for training
            bleu_metrics = self.bleu_scorer.compute_batch_metrics(references, hypotheses)
            rouge_metrics = self.rouge_scorer.compute_batch_metrics(references, hypotheses)
            medical_metrics = self._compute_medical_metrics(references, hypotheses)
            
            # Combine all metrics
            training_metrics = {
                # BLEU metrics (precision-oriented)
                'bleu_4': bleu_metrics['bleu_4'],
                'bleu_avg': bleu_metrics['bleu_avg'],
                
                # ROUGE metrics (recall-oriented)
                'rouge1_f1': rouge_metrics['rouge1_fmeasure'],
                'rouge2_f1': rouge_metrics['rouge2_fmeasure'],
                'rougeL_f1': rouge_metrics['rougeL_fmeasure'],
                'rouge_avg_recall': rouge_metrics['rouge_avg_recall'],
                
                # Medical-specific metrics
                'medical_coverage': medical_metrics['medical_coverage'],
                'medical_precision': medical_metrics['medical_precision'],
                'keyword_overlap': medical_metrics['keyword_overlap'],
                
                # Summary metrics for training monitoring
                'overall_f1': (rouge_metrics['rouge_avg_fmeasure'] + bleu_metrics['bleu_4']) / 2,
                'medical_quality': medical_metrics['medical_quality']
            }
            
            computation_time = time.time() - start_time
            training_metrics['metrics_time'] = computation_time
            
            logger.debug(f"Training metrics computed in {computation_time:.3f}s")
            return training_metrics
            
        except Exception as e:
            logger.error(f"Error computing training metrics: {e}")
            return self._get_zero_training_metrics()
    
    def compute_validation_metrics(self, 
                                  references: List[str], 
                                  hypotheses: List[str]) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for validation evaluation.
        
        Args:
            references: Ground truth keywords
            hypotheses: Generated keywords
            
        Returns:
            Dictionary with detailed validation metrics
        """
        try:
            start_time = time.time()
            
            # Full metric computation for validation
            bleu_metrics = self.bleu_scorer.compute_batch_metrics(references, hypotheses)
            rouge_detailed = self.rouge_scorer.compute_detailed_metrics(references, hypotheses)
            medical_metrics = self._compute_medical_metrics(references, hypotheses)
            
            # Analysis metrics
            analysis_metrics = self._compute_analysis_metrics(references, hypotheses)
            
            validation_metrics = {
                # Core metrics
                'bleu_metrics': bleu_metrics,
                'rouge_metrics': rouge_detailed['batch_averages'],
                'medical_metrics': medical_metrics,
                'analysis_metrics': analysis_metrics,
                
                # Key summary metrics
                'summary': {
                    'bleu_4': bleu_metrics['bleu_4'],
                    'rouge1_f1': rouge_detailed['batch_averages']['rouge1_fmeasure'],
                    'rouge_avg_recall': rouge_detailed['batch_averages']['rouge_avg_recall'],
                    'medical_coverage': medical_metrics['medical_coverage'],
                    'overall_quality': self._compute_overall_quality(bleu_metrics, rouge_detailed['batch_averages'], medical_metrics)
                },
                
                # Detailed analysis
                'best_example': rouge_detailed['best_example'],
                'worst_example': rouge_detailed['worst_example'],
                'total_samples': len(references),
                'valid_samples': rouge_detailed['valid_pairs']
            }
            
            computation_time = time.time() - start_time
            validation_metrics['metrics_time'] = computation_time
            
            logger.info(f"Validation metrics computed in {computation_time:.3f}s for {len(references)} samples")
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Error computing validation metrics: {e}")
            return self._get_zero_validation_metrics()
    
    def _compute_medical_metrics(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute medical domain-specific metrics.
        
        Args:
            references: Ground truth keywords
            hypotheses: Generated keywords
            
        Returns:
            Medical-specific metrics
        """
        try:
            total_pairs = len(references)
            if total_pairs == 0:
                return self._get_zero_medical_metrics()
            
            medical_coverage_scores = []
            medical_precision_scores = []
            keyword_overlap_scores = []
            
            for ref, hyp in zip(references, hypotheses):
                # Extract medical terms
                ref_medical = self._extract_medical_terms(ref)
                hyp_medical = self._extract_medical_terms(hyp)
                
                # Medical coverage (recall of medical terms)
                if ref_medical:
                    coverage = len(ref_medical.intersection(hyp_medical)) / len(ref_medical)
                else:
                    coverage = 1.0 if not hyp_medical else 0.0
                medical_coverage_scores.append(coverage)
                
                # Medical precision (precision of medical terms)
                if hyp_medical:
                    precision = len(ref_medical.intersection(hyp_medical)) / len(hyp_medical)
                else:
                    precision = 1.0 if not ref_medical else 0.0
                medical_precision_scores.append(precision)
                
                # Keyword overlap (Jaccard similarity)
                ref_keywords = set(ref.lower().replace(',', ' ').split())
                hyp_keywords = set(hyp.lower().replace(',', ' ').split())
                
                if ref_keywords or hyp_keywords:
                    overlap = len(ref_keywords.intersection(hyp_keywords)) / len(ref_keywords.union(hyp_keywords))
                else:
                    overlap = 1.0
                keyword_overlap_scores.append(overlap)
            
            # Average metrics
            medical_metrics = {
                'medical_coverage': sum(medical_coverage_scores) / total_pairs,
                'medical_precision': sum(medical_precision_scores) / total_pairs,
                'keyword_overlap': sum(keyword_overlap_scores) / total_pairs,
                'medical_f1': 0.0,  # Will compute below
                'medical_quality': 0.0  # Will compute below
            }
            
            # Compute F1 for medical terms
            if medical_metrics['medical_coverage'] + medical_metrics['medical_precision'] > 0:
                medical_metrics['medical_f1'] = (
                    2 * medical_metrics['medical_coverage'] * medical_metrics['medical_precision'] /
                    (medical_metrics['medical_coverage'] + medical_metrics['medical_precision'])
                )
            
            # Overall medical quality score
            medical_metrics['medical_quality'] = (
                medical_metrics['medical_f1'] * 0.5 +
                medical_metrics['keyword_overlap'] * 0.3 +
                medical_metrics['medical_coverage'] * 0.2
            )
            
            return medical_metrics
            
        except Exception as e:
            logger.error(f"Error computing medical metrics: {e}")
            return self._get_zero_medical_metrics()
    
    def _extract_medical_terms(self, text: str) -> set:
        """
        Extract medical terms from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of medical terms found
        """
        if not text:
            return set()
        
        # Normalize text
        text = text.lower().replace(',', ' ').replace(';', ' ')
        words = text.split()
        
        # Find medical terms
        medical_terms = set()
        for word in words:
            word_clean = word.strip('.,;()[]')
            if word_clean in self.medical_keywords:
                medical_terms.add(word_clean)
        
        return medical_terms
    
    def _compute_analysis_metrics(self, references: List[str], hypotheses: List[str]) -> Dict[str, Any]:
        """
        Compute analysis metrics for understanding model behavior.
        
        Args:
            references: Ground truth keywords
            hypotheses: Generated keywords
            
        Returns:
            Analysis metrics
        """
        try:
            # Length statistics
            ref_lengths = [len(ref.split(',')) for ref in references]
            hyp_lengths = [len(hyp.split(',')) for hyp in hypotheses]
            
            # Vocabulary statistics
            ref_vocab = set()
            hyp_vocab = set()
            for ref, hyp in zip(references, hypotheses):
                ref_vocab.update(ref.lower().replace(',', ' ').split())
                hyp_vocab.update(hyp.lower().replace(',', ' ').split())
            
            analysis = {
                'length_stats': {
                    'avg_ref_length': sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0,
                    'avg_hyp_length': sum(hyp_lengths) / len(hyp_lengths) if hyp_lengths else 0,
                    'length_ratio': (sum(hyp_lengths) / sum(ref_lengths)) if sum(ref_lengths) > 0 else 0
                },
                'vocabulary_stats': {
                    'ref_vocab_size': len(ref_vocab),
                    'hyp_vocab_size': len(hyp_vocab),
                    'vocab_overlap': len(ref_vocab.intersection(hyp_vocab)),
                    'vocab_coverage': len(ref_vocab.intersection(hyp_vocab)) / len(ref_vocab) if ref_vocab else 0
                },
                'medical_analysis': {
                    'ref_medical_terms': len([term for term in ref_vocab if term in self.medical_keywords]),
                    'hyp_medical_terms': len([term for term in hyp_vocab if term in self.medical_keywords]),
                    'medical_term_ratio': 0.0  # Will compute below
                }
            }
            
            # Medical term ratio
            if analysis['medical_analysis']['ref_medical_terms'] > 0:
                analysis['medical_analysis']['medical_term_ratio'] = (
                    analysis['medical_analysis']['hyp_medical_terms'] /
                    analysis['medical_analysis']['ref_medical_terms']
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error computing analysis metrics: {e}")
            return {'length_stats': {}, 'vocabulary_stats': {}, 'medical_analysis': {}}
    
    def _compute_overall_quality(self, bleu_metrics: Dict, rouge_metrics: Dict, medical_metrics: Dict) -> float:
        """
        Compute overall quality score combining all metrics.
        
        Args:
            bleu_metrics: BLEU scores
            rouge_metrics: ROUGE scores  
            medical_metrics: Medical-specific scores
            
        Returns:
            Overall quality score (0-1)
        """
        try:
            # Weighted combination of metrics
            quality_score = (
                bleu_metrics['bleu_4'] * 0.25 +           # Precision
                rouge_metrics['rouge_avg_fmeasure'] * 0.35 +  # Recall/F1
                medical_metrics['medical_quality'] * 0.40      # Medical domain
            )
            
            return min(1.0, max(0.0, quality_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Error computing overall quality: {e}")
            return 0.0
    
    def _get_zero_training_metrics(self) -> Dict[str, float]:
        """Get zero-initialized training metrics"""
        return {
            'bleu_4': 0.0,
            'bleu_avg': 0.0,
            'rouge1_f1': 0.0,
            'rouge2_f1': 0.0,
            'rougeL_f1': 0.0,
            'rouge_avg_recall': 0.0,
            'medical_coverage': 0.0,
            'medical_precision': 0.0,
            'keyword_overlap': 0.0,
            'overall_f1': 0.0,
            'medical_quality': 0.0,
            'metrics_time': 0.0
        }
    
    def _get_zero_validation_metrics(self) -> Dict[str, Any]:
        """Get zero-initialized validation metrics"""
        return {
            'bleu_metrics': {},
            'rouge_metrics': {},
            'medical_metrics': {},
            'analysis_metrics': {},
            'summary': {},
            'best_example': None,
            'worst_example': None,
            'total_samples': 0,
            'valid_samples': 0,
            'metrics_time': 0.0
        }
    
    def _get_zero_medical_metrics(self) -> Dict[str, float]:
        """Get zero-initialized medical metrics"""
        return {
            'medical_coverage': 0.0,
            'medical_precision': 0.0,
            'keyword_overlap': 0.0,
            'medical_f1': 0.0,
            'medical_quality': 0.0
        }


def test_medical_metrics():
    """Test medical metrics with realistic examples"""
    
    print("="*60)
    print("TESTING MEDICAL KEYWORD METRICS")
    print("="*60)
    
    # Initialize metrics
    metrics = MedicalKeywordMetrics()
    
    # Test cases with realistic medical examples
    references = [
        "atelectasis, bilateral pleural effusion, cardiomegaly",
        "pneumonia, right lower lobe consolidation",
        "no acute cardiopulmonary process",
        "mild cardiomegaly, chronic changes",
        "bilateral infiltrates, pneumothorax"
    ]
    
    hypotheses = [
        "atelectasis, effusion, enlarged heart",
        "pneumonia, right consolidation", 
        "normal chest",
        "heart enlargement, old findings",
        "bilateral pneumonia, air leak"
    ]
    
    print("🧪 Testing Training Metrics...")
    training_metrics = metrics.compute_training_metrics(references, hypotheses)
    
    print("📊 Training Metrics:")
    for key, value in training_metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n🔍 Testing Validation Metrics...")
    validation_metrics = metrics.compute_validation_metrics(references, hypotheses)
    
    print("📈 Validation Summary:")
    for key, value in validation_metrics['summary'].items():
        print(f"   {key}: {value:.4f}")
    
    print(f"\n📋 Analysis:")
    analysis = validation_metrics['analysis_metrics']
    print(f"   Avg reference length: {analysis['length_stats']['avg_ref_length']:.1f}")
    print(f"   Avg hypothesis length: {analysis['length_stats']['avg_hyp_length']:.1f}")
    print(f"   Medical terms in refs: {analysis['medical_analysis']['ref_medical_terms']}")
    print(f"   Medical terms in hyps: {analysis['medical_analysis']['hyp_medical_terms']}")
    
    if validation_metrics['best_example']:
        best = validation_metrics['best_example']
        print(f"\n🏆 Best Example:")
        print(f"   Ref: '{best['reference']}'")
        print(f"   Hyp: '{best['hypothesis']}'")
        print(f"   ROUGE-1 F1: {best['scores']['rouge1']['fmeasure']:.4f}")
    
    print(f"\n✅ Medical metrics test completed!")
    print(f"   Total samples: {validation_metrics['total_samples']}")
    print(f"   Valid samples: {validation_metrics['valid_samples']}")
    print(f"   Computation time: {validation_metrics['metrics_time']:.3f}s")
    
    return True


if __name__ == "__main__":
    test_medical_metrics() 