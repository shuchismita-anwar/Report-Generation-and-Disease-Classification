#!/usr/bin/env python3
"""
RadGraph F1 Scorer
==================
Implementation of RadGraph F1 metric for evaluating entity and relation extraction.

This module provides:
- Parsing of clinical text into RadGraph graphs (entities and relations)
- Calculation of F1 score for entity overlap
- Calculation of F1 score for relation overlap
- A combined RadGraph F1 score

Author: Medical Report Evaluation System
Version: 1.0
"""

from typing import Dict, Optional, Set, Tuple
import logging

# Import base metric class
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from base_metric import BaseMetric

class RadGraphF1Scorer(BaseMetric):
    """
    RadGraph F1 scorer for clinical entity and relation extraction evaluation.
    
    Uses the RadGraph parser to convert text to graphs and calculates the F1
    score based on the overlap of entities and relations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the RadGraph F1 scorer.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance
        """
        super().__init__("radgraph_f1", logger)
        self.description = "RadGraph F1 metric for entity and relation overlap"
        self.metric_type = "clinical"
        self.parser = None
        
    def initialize(self):
        """Initialize the RadGraph parser."""
        if self.is_initialized:
            return
        
        try:
            from radgraph import RadGraph
            # Use the correct model type - "radgraph-s" is not valid
            self.logger.info("Loading RadGraph model (radgraph)...")
            self.parser = RadGraph(model_type="radgraph", is_multiprocessing=False)
            self.logger.info("RadGraph model loaded.")
        except ImportError:
            self.logger.error("RadGraph library not installed. Please run: pip install radgraph")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize RadGraph parser: {e}")
            raise
        
        super().initialize()

    def calculate(self, reference: str, candidate: str, **kwargs) -> Dict[str, float]:
        """
        Calculate RadGraph F1 score between reference and candidate texts.
        
        Args:
            reference (str): Ground truth reference text
            candidate (str): Generated candidate text to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, float]: Dictionary containing RadGraph F1 scores
        """
        if not self.is_initialized:
            self.initialize()
            
        # Validate inputs
        is_valid, issues = self.validate_inputs(reference, candidate)
        if not is_valid:
            raise ValueError(f"Invalid inputs for RadGraph scoring: {issues}")
        
        # Parse reports into graphs
        ref_entities, ref_relations = self._parse_text(reference)
        cand_entities, cand_relations = self._parse_text(candidate)

        # Calculate entity F1
        entity_scores = self._calculate_f1(ref_entities, cand_entities)
        
        # Calculate relation F1
        relation_scores = self._calculate_f1(ref_relations, cand_relations)

        # Overall F1 is the F1 of relations, as is standard
        overall_f1 = relation_scores['f1']

        return {
            "radgraph_f1": overall_f1,
            "radgraph_entity_precision": entity_scores['precision'],
            "radgraph_entity_recall": entity_scores['recall'],
            "radgraph_entity_f1": entity_scores['f1'],
            "radgraph_relation_precision": relation_scores['precision'],
            "radgraph_relation_recall": relation_scores['recall'],
            "radgraph_relation_f1": relation_scores['f1'],
        }

    def _parse_text(self, text: str) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
        """Parse text and return sets of entities and relations."""
        if not self.parser:
            raise RuntimeError("RadGraph parser not initialized.")
        
        try:
            # RadGraph is callable directly and returns a dict
            result = self.parser(text)
            
            entity_set: Set[str] = set()
            relation_set: Set[Tuple[str, str, str]] = set()
            
            # RadGraph returns: {'0': {'text': str, 'entities': {id: entity_dict}, ...}}
            for doc_id, doc_data in result.items():
                entities = doc_data.get("entities", {})
                
                # Build canonical entity strings: label|text (lower-cased)
                for ent_id, ent in entities.items():
                    label = ent.get("label", "?")
                    ent_text = ent.get("tokens", "").lower()
                    entity_repr = f"{label}|{ent_text}"
                    entity_set.add(entity_repr)
                
                # Build canonical relation tuples
                # Each entity has a 'relations' list: [['relation_type', 'target_entity_id'], ...]
                for ent_id, ent in entities.items():
                    relations = ent.get("relations", [])
                    for rel in relations:
                        if len(rel) >= 2:
                            rel_type = rel[0]
                            target_id = rel[1]
                            
                            # Get source entity representation
                            source_label = ent.get("label", "?")
                            source_text = ent.get("tokens", "").lower()
                            source_repr = f"{source_label}|{source_text}"
                            
                            # Get target entity representation
                            if target_id in entities:
                                target_ent = entities[target_id]
                                target_label = target_ent.get("label", "?")
                                target_text = target_ent.get("tokens", "").lower()
                                target_repr = f"{target_label}|{target_text}"
                                
                                relation_set.add((source_repr, rel_type, target_repr))

            return entity_set, relation_set
        except Exception as e:
            self.logger.error(f"Error parsing text with RadGraph: {e}")
            return set(), set()

    def _calculate_f1(self, ref_set: Set, cand_set: Set) -> Dict[str, float]:
        """Calculate Precision, Recall, and F1 for two sets."""
        # If both sets are empty, we consider the information unavailable -> F1 = 0
        if not ref_set and not cand_set:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        tp = len(ref_set.intersection(cand_set))
        fp = len(cand_set.difference(ref_set))
        fn = len(ref_set.difference(cand_set))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}

    def get_name(self) -> str:
        return "RadGraph F1"

    def get_description(self) -> str:
        return self.description

def test_radgraph_f1_scorer():
    """Test function for the RadGraph F1 Scorer."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    print("🧪 Testing RadGraph F1 Scorer...")
    
    try:
        scorer = RadGraphF1Scorer(logger=logger)
        
        ref = "Lungs are clear. No pleural effusion."
        cand = "Clear lungs without pleural effusion."
        
        scores = scorer.calculate(ref, cand)
        
        print(f"Reference: '{ref}'")
        print(f"Candidate: '{cand}'")
        print(f"Scores: {scores}")
        
        assert scores['radgraph_f1'] > 0.5
        assert scores['radgraph_entity_f1'] > 0.5

        print("\n✅ RadGraph F1 scorer tests completed!")
    except ImportError:
        print("⚠️ RadGraph library not found, skipping test. Please run `pip install radgraph`.")
    except Exception as e:
        print(f"❌ RadGraph F1 scorer test failed: {e}")

if __name__ == "__main__":
    test_radgraph_f1_scorer() 