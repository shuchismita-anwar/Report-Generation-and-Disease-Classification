#!/usr/bin/env python3
"""
Medical-Specific Scorer
=======================
Implementation of medical domain-specific evaluation metrics for radiology reports.

This module provides:
- Clinical terminology matching
- Anatomy and pathology detection
- Medical concept alignment
- Severity assessment
- Negation handling
- Medical abbreviation normalization

Author: Medical Report Evaluation System
Version: 1.0
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set, Union
import logging

# Import base metric class
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from base_metric import BaseMetric


class MedicalScorer(BaseMetric):
    """
    Medical domain-specific scorer for radiology reports.
    
    Evaluates medical reports based on clinical terminology,
    anatomical structures, pathological findings, and medical concepts.
    """
    
    def __init__(self, use_negation_detection: bool = True, 
                 use_severity_matching: bool = True,
                 normalize_abbreviations: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the medical scorer.
        
        Args:
            use_negation_detection (bool): Whether to detect negations
            use_severity_matching (bool): Whether to match severity levels
            normalize_abbreviations (bool): Whether to normalize medical abbreviations
            logger (Optional[logging.Logger]): Logger instance
        """
        super().__init__("medical", logger)
        self.description = "Medical domain-specific scorer for radiology reports"
        self.metric_type = "clinical"
        
        # Configuration
        self.use_negation_detection = use_negation_detection
        self.use_severity_matching = use_severity_matching
        self.normalize_abbreviations = normalize_abbreviations
        
        # Medical vocabularies
        self._initialize_medical_vocabularies()
        
        # Negation patterns
        self.negation_patterns = [
            r'\bno\b', r'\bnot\b', r'\bnegative\b', r'\babsent\b', r'\bwithout\b',
            r'\bdenies\b', r'\bdenied\b', r'\bunremarkable\b', r'\bnormal\b',
            r'\bno evidence of\b', r'\bno signs of\b', r'\bno indication of\b'
        ]
        
        # Severity levels
        self.severity_levels = {
            'mild': 1, 'minimal': 1, 'slight': 1, 'small': 1,
            'moderate': 2, 'medium': 2,
            'severe': 3, 'marked': 3, 'significant': 3, 'large': 3,
            'massive': 4, 'extensive': 4, 'complete': 4
        }
        
        self.logger.debug("Initialized Medical scorer with clinical vocabularies")
    
    def _initialize_medical_vocabularies(self) -> None:
        """Initialize medical vocabularies and mappings."""
        
        # Anatomical structures
        self.anatomy_terms = {
            # Chest anatomy
            'lung', 'lungs', 'pulmonary', 'respiratory', 'bronchi', 'bronchus',
            'alveoli', 'pleura', 'pleural', 'diaphragm', 'mediastinum',
            'heart', 'cardiac', 'cardio', 'aorta', 'aortic', 'ventricle', 'atrium',
            'ribs', 'sternum', 'clavicle', 'spine', 'vertebra', 'thorax', 'thoracic',
            
            # Specific lung regions
            'apex', 'base', 'hilum', 'hila', 'bilateral', 'unilateral',
            'upper_lobe', 'middle_lobe', 'lower_lobe', 'lingula',
            'right_lung', 'left_lung', 'costophrenic', 'cardiophrenic'
        }
        
        # Pathological findings
        self.pathology_terms = {
            # Common findings
            'pneumonia', 'infection', 'infiltrate', 'consolidation', 'opacity',
            'effusion', 'fluid', 'edema', 'congestion', 'atelectasis',
            'pneumothorax', 'collapse', 'mass', 'nodule', 'lesion',
            'cardiomegaly', 'enlargement', 'hypertrophy', 'dilation',
            
            # Specific conditions
            'copd', 'emphysema', 'fibrosis', 'scarring', 'granuloma',
            'tuberculosis', 'cancer', 'malignancy', 'metastasis',
            'fracture', 'break', 'displacement', 'deformity'
        }
        
        # Normal findings
        self.normal_terms = {
            'normal', 'unremarkable', 'clear', 'patent', 'intact',
            'within_normal_limits', 'wnl', 'negative', 'no_abnormalities',
            'stable', 'unchanged', 'appropriate', 'adequate'
        }
        
        # Medical abbreviations
        self.abbreviations = {
            'wnl': 'within normal limits',
            'copd': 'chronic obstructive pulmonary disease',
            'chf': 'congestive heart failure',
            'pe': 'pulmonary embolism',
            'ptx': 'pneumothorax',
            'cxr': 'chest x-ray',
            'ct': 'computed tomography',
            'mri': 'magnetic resonance imaging',
            'rul': 'right upper lobe',
            'rml': 'right middle lobe',
            'rll': 'right lower lobe',
            'lul': 'left upper lobe',
            'lll': 'left lower lobe',
            'bil': 'bilateral',
            'r/o': 'rule out',
            'h/o': 'history of',
            's/p': 'status post'
        }
        
        # Clinical concepts mapping
        self.concept_categories = {
            'anatomy': self.anatomy_terms,
            'pathology': self.pathology_terms,
            'normal': self.normal_terms
        }
    
    def calculate(self, reference: str, candidate: str, **kwargs) -> Dict[str, float]:
        """
        Calculate medical-specific scores between reference and candidate texts.
        
        Args:
            reference (str): Ground truth reference text
            candidate (str): Generated candidate text to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, float]: Dictionary containing medical scores
        """
        # Validate inputs
        is_valid, issues = self.validate_inputs(reference, candidate)
        if not is_valid:
            raise ValueError(f"Invalid inputs for Medical scoring: {issues}")
        
        # Preprocess texts
        ref_processed = self._preprocess_medical_text(reference)
        cand_processed = self._preprocess_medical_text(candidate)
        
        # Extract medical concepts
        ref_concepts = self._extract_medical_concepts(ref_processed)
        cand_concepts = self._extract_medical_concepts(cand_processed)
        
        # Calculate concept-based scores
        concept_scores = self._calculate_concept_scores(ref_concepts, cand_concepts)
        
        # Calculate anatomy matching
        anatomy_score = self._calculate_anatomy_score(ref_concepts, cand_concepts)
        
        # Calculate pathology matching
        pathology_score = self._calculate_pathology_score(ref_concepts, cand_concepts)
        
        # Calculate negation consistency
        negation_score = self._calculate_negation_score(reference, candidate)
        
        # Calculate severity matching
        severity_score = self._calculate_severity_score(reference, candidate)
        
        # Calculate clinical terminology overlap
        terminology_score = self._calculate_terminology_score(ref_processed, cand_processed)
        
        # Overall medical score (weighted combination)
        medical_score = (
            0.25 * concept_scores['concept_f1'] +
            0.20 * anatomy_score +
            0.25 * pathology_score +
            0.15 * negation_score +
            0.10 * severity_score +
            0.05 * terminology_score
        )
        
        scores = {
            "medical_score": medical_score,
            "medical_concept_precision": concept_scores['concept_precision'],
            "medical_concept_recall": concept_scores['concept_recall'],
            "medical_concept_f1": concept_scores['concept_f1'],
            "medical_anatomy_score": anatomy_score,
            "medical_pathology_score": pathology_score,
            "medical_negation_score": negation_score,
            "medical_severity_score": severity_score,
            "medical_terminology_score": terminology_score,
            "reference_concepts": float(len(ref_concepts['all'])),
            "candidate_concepts": float(len(cand_concepts['all'])),
            "concept_overlap": float(len(ref_concepts['all'] & cand_concepts['all']))
        }
        
        return scores
    
    def _preprocess_medical_text(self, text: str) -> str:
        """
        Preprocess medical text with abbreviation normalization.
        
        Args:
            text (str): Input medical text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Normalize abbreviations if enabled
        if self.normalize_abbreviations:
            for abbrev, expansion in self.abbreviations.items():
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(abbrev) + r'\b'
                text = re.sub(pattern, expansion, text)
        
        # Handle medical punctuation
        text = re.sub(r'([a-z])\.([a-z])', r'\1. \2', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_medical_concepts(self, text: str) -> Dict[str, Set[str]]:
        """
        Extract medical concepts from text.
        
        Args:
            text (str): Preprocessed medical text
            
        Returns:
            Dict[str, Set[str]]: Medical concepts by category
        """
        concepts = {
            'anatomy': set(),
            'pathology': set(),
            'normal': set(),
            'all': set()
        }
        
        # Tokenize text
        tokens = re.findall(r'\b\w+\b', text)
        
        # Extract concepts by category
        for category, terms in self.concept_categories.items():
            for token in tokens:
                if token in terms:
                    concepts[category].add(token)
                    concepts['all'].add(token)
        
        # Handle multi-word concepts
        for category, terms in self.concept_categories.items():
            for term in terms:
                if '_' in term:  # Multi-word term
                    term_pattern = term.replace('_', r'\s+')
                    if re.search(r'\b' + term_pattern + r'\b', text):
                        concepts[category].add(term)
                        concepts['all'].add(term)
        
        return concepts
    
    def _calculate_concept_scores(self, ref_concepts: Dict[str, Set[str]], 
                                 cand_concepts: Dict[str, Set[str]]) -> Dict[str, float]:
        """
        Calculate concept-based precision, recall, and F1 scores.
        
        Args:
            ref_concepts (Dict[str, Set[str]]): Reference concepts
            cand_concepts (Dict[str, Set[str]]): Candidate concepts
            
        Returns:
            Dict[str, float]: Concept scores
        """
        ref_all = ref_concepts['all']
        cand_all = cand_concepts['all']
        
        if not ref_all and not cand_all:
            return {'concept_precision': 1.0, 'concept_recall': 1.0, 'concept_f1': 1.0}
        
        if not ref_all:
            return {'concept_precision': 0.0, 'concept_recall': 0.0, 'concept_f1': 0.0}
        
        if not cand_all:
            return {'concept_precision': 0.0, 'concept_recall': 0.0, 'concept_f1': 0.0}
        
        # Calculate overlap
        overlap = ref_all & cand_all
        
        # Calculate precision and recall
        precision = len(overlap) / len(cand_all)
        recall = len(overlap) / len(ref_all)
        
        # Calculate F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'concept_precision': precision,
            'concept_recall': recall,
            'concept_f1': f1
        }
    
    def _calculate_anatomy_score(self, ref_concepts: Dict[str, Set[str]], 
                                cand_concepts: Dict[str, Set[str]]) -> float:
        """
        Calculate anatomy-specific matching score.
        
        Args:
            ref_concepts (Dict[str, Set[str]]): Reference concepts
            cand_concepts (Dict[str, Set[str]]): Candidate concepts
            
        Returns:
            float: Anatomy matching score
        """
        ref_anatomy = ref_concepts['anatomy']
        cand_anatomy = cand_concepts['anatomy']
        
        if not ref_anatomy and not cand_anatomy:
            return 1.0
        
        if not ref_anatomy or not cand_anatomy:
            return 0.0
        
        # Calculate Jaccard similarity for anatomy terms
        overlap = ref_anatomy & cand_anatomy
        union = ref_anatomy | cand_anatomy
        
        return len(overlap) / len(union) if union else 0.0
    
    def _calculate_pathology_score(self, ref_concepts: Dict[str, Set[str]], 
                                  cand_concepts: Dict[str, Set[str]]) -> float:
        """
        Calculate pathology-specific matching score.
        
        Args:
            ref_concepts (Dict[str, Set[str]]): Reference concepts
            cand_concepts (Dict[str, Set[str]]): Candidate concepts
            
        Returns:
            float: Pathology matching score
        """
        ref_pathology = ref_concepts['pathology']
        cand_pathology = cand_concepts['pathology']
        
        # Special handling for normal vs pathological findings
        ref_normal = ref_concepts['normal']
        cand_normal = cand_concepts['normal']
        
        # If both indicate normal findings
        if ref_normal and cand_normal and not ref_pathology and not cand_pathology:
            return 1.0
        
        # If one indicates normal and other indicates pathology
        if (ref_normal and cand_pathology) or (ref_pathology and cand_normal):
            return 0.0
        
        # If both have pathological findings
        if ref_pathology and cand_pathology:
            overlap = ref_pathology & cand_pathology
            union = ref_pathology | cand_pathology
            return len(overlap) / len(union) if union else 0.0
        
        # If neither has clear pathological findings
        return 0.5
    
    def _calculate_negation_score(self, reference: str, candidate: str) -> float:
        """
        Calculate negation consistency score.
        
        Args:
            reference (str): Reference text
            candidate (str): Candidate text
            
        Returns:
            float: Negation consistency score
        """
        if not self.use_negation_detection:
            return 1.0
        
        # Detect negations in both texts
        ref_negations = self._detect_negations(reference)
        cand_negations = self._detect_negations(candidate)
        
        # Calculate consistency
        if ref_negations == cand_negations:
            return 1.0
        elif abs(ref_negations - cand_negations) <= 1:
            return 0.5
        else:
            return 0.0
    
    def _detect_negations(self, text: str) -> int:
        """
        Detect number of negations in text.
        
        Args:
            text (str): Input text
            
        Returns:
            int: Number of negations detected
        """
        text = text.lower()
        negation_count = 0
        
        for pattern in self.negation_patterns:
            matches = re.findall(pattern, text)
            negation_count += len(matches)
        
        return negation_count
    
    def _calculate_severity_score(self, reference: str, candidate: str) -> float:
        """
        Calculate severity level matching score.
        
        Args:
            reference (str): Reference text
            candidate (str): Candidate text
            
        Returns:
            float: Severity matching score
        """
        if not self.use_severity_matching:
            return 1.0
        
        # Extract severity levels
        ref_severity = self._extract_severity(reference)
        cand_severity = self._extract_severity(candidate)
        
        # If no severity mentioned in either
        if ref_severity == 0 and cand_severity == 0:
            return 1.0
        
        # If severity mentioned in only one
        if ref_severity == 0 or cand_severity == 0:
            return 0.5
        
        # Calculate severity difference
        severity_diff = abs(ref_severity - cand_severity)
        
        # Score based on difference (closer = higher score)
        if severity_diff == 0:
            return 1.0
        elif severity_diff == 1:
            return 0.7
        elif severity_diff == 2:
            return 0.4
        else:
            return 0.1
    
    def _extract_severity(self, text: str) -> int:
        """
        Extract severity level from text.
        
        Args:
            text (str): Input text
            
        Returns:
            int: Severity level (0 = none, 1-4 = mild to severe)
        """
        text = text.lower()
        
        for severity_term, level in self.severity_levels.items():
            if re.search(r'\b' + severity_term + r'\b', text):
                return level
        
        return 0  # No severity mentioned
    
    def _calculate_terminology_score(self, ref_text: str, cand_text: str) -> float:
        """
        Calculate clinical terminology overlap score.
        
        Args:
            ref_text (str): Reference text
            cand_text (str): Candidate text
            
        Returns:
            float: Terminology overlap score
        """
        # Extract all medical terms
        ref_terms = set()
        cand_terms = set()
        
        ref_tokens = re.findall(r'\b\w+\b', ref_text)
        cand_tokens = re.findall(r'\b\w+\b', cand_text)
        
        # Collect all medical vocabulary terms
        all_medical_terms = set()
        for terms in self.concept_categories.values():
            all_medical_terms.update(terms)
        
        # Find medical terms in texts
        for token in ref_tokens:
            if token in all_medical_terms:
                ref_terms.add(token)
        
        for token in cand_tokens:
            if token in all_medical_terms:
                cand_terms.add(token)
        
        # Calculate Jaccard similarity
        if not ref_terms and not cand_terms:
            return 1.0
        
        overlap = ref_terms & cand_terms
        union = ref_terms | cand_terms
        
        return len(overlap) / len(union) if union else 0.0
    
    def get_name(self) -> str:
        """Get the metric name."""
        return "Medical"
    
    def get_description(self) -> str:
        """Get the metric description."""
        return (f"Medical domain-specific scorer for radiology reports "
                f"(negation={self.use_negation_detection}, "
                f"severity={self.use_severity_matching}, "
                f"abbreviations={self.normalize_abbreviations})")


def test_medical_scorer():
    """Test function for the Medical Scorer."""
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    try:
        print("🧪 Testing Medical Scorer...")
        
        # Initialize Medical scorer
        medical = MedicalScorer(
            use_negation_detection=True,
            use_severity_matching=True,
            normalize_abbreviations=True,
            logger=logger
        )
        
        # Test basic properties
        print(f"✅ Metric name: {medical.get_name()}")
        print(f"✅ Metric description: {medical.get_description()}")
        
        # Test cases with different medical scenarios
        test_cases = [
            # Perfect medical match
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Normal chest radiograph with clear lung fields.",
                "description": "Perfect medical match"
            },
            
            # Anatomy and pathology match
            {
                "reference": "Bilateral lung infiltrates consistent with pneumonia.",
                "candidate": "Bilateral pulmonary infiltrates suggesting pneumonia.",
                "description": "Anatomy and pathology match"
            },
            
            # Normal vs pathological mismatch
            {
                "reference": "Normal chest radiograph with clear lung fields.",
                "candidate": "Bilateral pneumonia with consolidation.",
                "description": "Normal vs pathological mismatch"
            },
            
            # Severity level matching
            {
                "reference": "Mild cardiomegaly with no acute abnormalities.",
                "candidate": "Moderate heart enlargement without acute findings.",
                "description": "Severity level difference"
            },
            
            # Negation handling
            {
                "reference": "No evidence of pneumonia or effusion.",
                "candidate": "No pneumonia or pleural fluid identified.",
                "description": "Negation consistency"
            },
            
            # Medical abbreviations
            {
                "reference": "CXR shows wnl findings in rul and lul.",
                "candidate": "Chest x-ray shows within normal limits findings in right upper lobe and left upper lobe.",
                "description": "Medical abbreviation normalization"
            }
        ]
        
        print("\n--- Medical Score Tests ---")
        for i, test_case in enumerate(test_cases, 1):
            scores = medical.calculate(test_case["reference"], test_case["candidate"])
            
            print(f"\nTest {i}: {test_case['description']}")
            print(f"Reference: '{test_case['reference']}'")
            print(f"Candidate: '{test_case['candidate']}'")
            print(f"Medical Score: {scores['medical_score']:.4f}")
            print(f"Concept F1: {scores['medical_concept_f1']:.4f}")
            print(f"Anatomy Score: {scores['medical_anatomy_score']:.4f}")
            print(f"Pathology Score: {scores['medical_pathology_score']:.4f}")
            print(f"Negation Score: {scores['medical_negation_score']:.4f}")
            print(f"Severity Score: {scores['medical_severity_score']:.4f}")
        
        print("\n✅ All Medical scorer tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    success = test_medical_scorer()
    
    if success:
        print("\n🎉 Medical Scorer tests passed!")
    else:
        print("\n❌ Some tests failed!") 