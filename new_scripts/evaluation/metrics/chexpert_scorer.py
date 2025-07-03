#!/usr/bin/env python3
"""
CheXpert Label Scorer
=====================
Implementation of CheXpert label overlap metric for radiology reports.

This module provides:
- Extraction of 14 common CheXpert labels
- Negation handling for clinical findings
- Calculation of precision, recall, and F1 score for label agreement

Author: Medical Report Evaluation System
Version: 1.0
"""

import re
from typing import Dict, List, Optional, Set
import logging

# Import base metric class
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from base_metric import BaseMetric

class ChexpertScorer(BaseMetric):
    """
    CheXpert label-based scorer for radiology reports.
    
    Evaluates reports by extracting and comparing 14 standard CheXpert labels.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the CheXpert scorer.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance
        """
        super().__init__("chexpert", logger)
        self.description = "CheXpert label agreement metric for clinical accuracy"
        self.metric_type = "clinical"
        
        # Initialize CheXpert label definitions and keywords
        self._initialize_chexpert_labels()

    def _initialize_chexpert_labels(self):
        """Defines the 14 CheXpert labels and their associated keywords."""
        self.chexpert_labels = {
            "No Finding": {"keywords": ["no finding", "normal", "clear"], "type": "global"},
            "Enlarged Cardiomediastinum": {"keywords": ["enlarged cardiomediastinum", "wide mediastinum"]},
            "Cardiomegaly": {"keywords": ["cardiomegaly", "enlarged heart", "cardiac enlargement"]},
            "Lung Opacity": {"keywords": ["opacity", "opacification"]},
            "Lung Lesion": {"keywords": ["lesion", "nodule", "mass"]},
            "Edema": {"keywords": ["edema", "pulmonary edema", "congestion"]},
            "Consolidation": {"keywords": ["consolidation", "infiltrate"]},
            "Pneumonia": {"keywords": ["pneumonia", "infection"]},
            "Atelectasis": {"keywords": ["atelectasis", "collapse"]},
            "Pneumothorax": {"keywords": ["pneumothorax"]},
            "Pleural Effusion": {"keywords": ["effusion", "pleural effusion", "fluid"]},
            "Pleural Other": {"keywords": ["pleural thickening", "pleural disease"]},
            "Fracture": {"keywords": ["fracture", "rib fracture"]},
            "Support Devices": {"keywords": ["device", "tube", "line", "catheter", "pacemaker"]},
        }
        
        self.negation_patterns = re.compile(
            r'\b(no|not|negative|absent|without|denies|denied|unremarkable|clear of|free of|no evidence of|no sign of)\b', 
            re.IGNORECASE
        )

    def calculate(self, reference: str, candidate: str, **kwargs) -> Dict[str, float]:
        """
        Calculate CheXpert label agreement between reference and candidate texts.
        
        Args:
            reference (str): Ground truth reference text
            candidate (str): Generated candidate text to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, float]: Dictionary containing CheXpert label scores
        """
        # Validate inputs
        is_valid, issues = self.validate_inputs(reference, candidate)
        if not is_valid:
            raise ValueError(f"Invalid inputs for CheXpert scoring: {issues}")
        
        # Extract labels
        ref_labels = self._extract_labels(reference)
        cand_labels = self._extract_labels(candidate)
        
        # Get sets of positive labels
        ref_positive = {label for label, present in ref_labels.items() if present == 1}
        cand_positive = {label for label, present in cand_labels.items() if present == 1}
        
        # Calculate precision, recall, F1
        tp = len(ref_positive.intersection(cand_positive))
        fp = len(cand_positive.difference(ref_positive))
        fn = len(ref_positive.difference(cand_positive))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate label-wise agreement
        label_agreement = 0
        for label in self.chexpert_labels:
            if ref_labels[label] == cand_labels[label]:
                label_agreement += 1
        
        accuracy = label_agreement / len(self.chexpert_labels)

        return {
            "chexpert_f1": f1,
            "chexpert_precision": precision,
            "chexpert_recall": recall,
            "chexpert_accuracy": accuracy,
            "true_positives": float(tp),
            "false_positives": float(fp),
            "false_negatives": float(fn),
        }

    def _extract_labels(self, text: str) -> Dict[str, int]:
        """
        Extract CheXpert labels from text.
        
        Returns:
            Dict[str, int]: Dictionary mapping label to 1 (positive) or 0 (negative).
        """
        text = text.lower()
        labels = {label: 0 for label in self.chexpert_labels}

        for label, data in self.chexpert_labels.items():
            for keyword in data["keywords"]:
                if keyword in text:
                    # Check for negation in a window around the keyword
                    window_size = 15
                    for match in re.finditer(keyword, text):
                        start = max(0, match.start() - window_size)
                        end = match.end()
                        window = text[start:end]
                        if not self.negation_patterns.search(window):
                            labels[label] = 1
                            break # Move to next label
            if labels[label] == 1:
                continue
        
        return labels

    def get_name(self) -> str:
        return "CheXpert Labeler"

    def get_description(self) -> str:
        return self.description

def test_chexpert_scorer():
    """Test function for the CheXpert Scorer."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    print("🧪 Testing CheXpert Scorer...")
    scorer = ChexpertScorer(logger=logger)
    
    ref = "Patient has cardiomegaly and a small pleural effusion. No sign of pneumonia."
    cand = "The x-ray shows an enlarged heart and some fluid in the pleura. Lungs are clear."
    
    scores = scorer.calculate(ref, cand)
    
    print(f"Reference: '{ref}'")
    print(f"Candidate: '{cand}'")
    print(f"Scores: {scores}")
    
    assert scores['chexpert_f1'] > 0.5
    assert scores['chexpert_accuracy'] > 0.5

    ref_labels = scorer._extract_labels(ref.lower())
    cand_labels = scorer._extract_labels(cand.lower())
    
    print(f"Reference Labels: {[k for k,v in ref_labels.items() if v==1]}")
    print(f"Candidate Labels: {[k for k,v in cand_labels.items() if v==1]}")
    
    assert ref_labels['Cardiomegaly'] == 1
    assert ref_labels['Pleural Effusion'] == 1
    assert ref_labels['Pneumonia'] == 0
    
    assert cand_labels['Cardiomegaly'] == 1
    assert cand_labels['Pleural Effusion'] == 1
    assert cand_labels['Pneumonia'] == 0
    
    print("\n✅ CheXpert scorer tests completed!")

if __name__ == "__main__":
    test_chexpert_scorer() 