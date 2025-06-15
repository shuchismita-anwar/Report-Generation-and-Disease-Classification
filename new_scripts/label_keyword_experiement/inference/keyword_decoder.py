"""
Keyword Decoder for Converting Model Outputs to Text
===================================================

Converts keyword prediction logits to human-readable keyword lists
with confidence scores and category information.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

class KeywordDecoder:
    """
    Decodes keyword prediction logits to human-readable text with confidence scores.
    
    Handles:
    - Global keyword prediction decoding
    - Region-specific keyword prediction decoding
    - Confidence score calculation
    - Medical category mapping
    """
    
    def __init__(self, 
                 vocabulary: Dict,
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.1):
        """
        Initialize keyword decoder with vocabulary.
        
        Args:
            vocabulary: Vocabulary dictionary with word2idx, idx2word mappings
            device: Device for tensor operations
            confidence_threshold: Minimum confidence for keyword inclusion
        """
        self.vocabulary = vocabulary
        self.word2idx = vocabulary.get("word2idx", {})
        self.idx2word = vocabulary.get("idx2word", {})
        self.word_frequencies = vocabulary.get("word_frequencies", {})
        self.category_frequencies = vocabulary.get("category_frequencies", {})
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.confidence_threshold = confidence_threshold
        
        # Region names from config
        self.region_names = [
            "cardiac silhouette", "left clavicle", "left costophrenic angle",
            "left hilar structures", "left lower lung zone", "left lung",
            "left mid lung zone", "left upper lung zone", "right clavicle",
            "right costophrenic angle", "right hilar structures", "right lower lung zone",
            "right lung", "right mid lung zone", "right upper lung zone",
            "trachea", "upper mediastinum"
        ]
        
        # Medical categories for keyword organization
        self.medical_categories = {
            "anatomy": ["thorax", "cardiomediastinal", "pleura", "lung", "cardiac", "mediastinal", "hilar"],
            "pathology": ["pneumothorax", "atelectasis", "effusion", "abnormal", "abnormality", "cardiomegaly", "consolidation"],
            "devices": ["tube", "catheter", "picc", "endotracheal"],
            "descriptors": ["normal", "acute", "small", "large", "enlarged"],
            "location": ["right", "left", "upper", "lower"],
            "negation": ["normal", "no"]
        }
        
        logging.info(f"KeywordDecoder initialized with {len(self.word2idx)} vocabulary words")
        
    def decode_global_keywords(self, 
                             keyword_logits: torch.Tensor, 
                             top_k: int = 10,
                             threshold: Optional[float] = None) -> List[Dict]:
        """
        Decode global keyword predictions.
        
        Args:
            keyword_logits: Keyword prediction logits [vocab_size]
            top_k: Number of top keywords to return
            threshold: Confidence threshold (uses instance threshold if None)
            
        Returns:
            List of keyword dictionaries with confidence scores
        """
        if threshold is None:
            threshold = self.confidence_threshold
            
        # Convert to probabilities
        keyword_probs = torch.sigmoid(keyword_logits).cpu().numpy()
        
        # Get indices above threshold
        valid_indices = np.where(keyword_probs > threshold)[0]
        
        # Sort by confidence
        sorted_indices = valid_indices[np.argsort(keyword_probs[valid_indices])[::-1]]
        
        # Take top-k
        top_indices = sorted_indices[:top_k]
        
        # Build keyword list
        keywords = []
        for idx in top_indices:
            idx_str = str(idx)
            if idx_str in self.idx2word:
                keyword = self.idx2word[idx_str]
                confidence = float(keyword_probs[idx])
                
                # Skip special tokens
                if keyword in ["<PAD>", "<UNK>", "<START>", "<END>"]:
                    continue
                    
                # Get category and frequency info
                category = self._get_keyword_category(keyword)
                frequency = self.word_frequencies.get(keyword, 0)
                
                keywords.append({
                    "keyword": keyword,
                    "confidence": confidence,
                    "category": category,
                    "frequency": frequency,
                    "global_rank": len(keywords) + 1
                })
                
        return keywords
    
    def decode_top_k(self, 
                    keyword_logits: torch.Tensor, 
                    k: int = 10) -> List[str]:
        """
        Decode top-k keywords (simplified interface for compatibility).
        
        Args:
            keyword_logits: Keyword prediction logits [batch_size, vocab_size] or [vocab_size]
            k: Number of top keywords to return
            
        Returns:
            List of keyword strings
        """
        # Handle batched input - take first sample
        if keyword_logits.dim() == 2:
            keyword_logits = keyword_logits[0]
        
        # Decode using existing method
        keyword_dicts = self.decode_global_keywords(keyword_logits, top_k=k, threshold=0.0)
        
        # Extract just the keyword strings
        return [kw['keyword'] for kw in keyword_dicts]
        
    def decode_region_keywords(self, 
                             region_keyword_logits: torch.Tensor,
                             top_k: int = 5,
                             threshold: Optional[float] = None) -> Dict[str, List[Dict]]:
        """
        Decode region-specific keyword predictions.
        
        Args:
            region_keyword_logits: Region keyword logits [num_regions, vocab_size]
            top_k: Number of top keywords per region
            threshold: Confidence threshold
            
        Returns:
            Dictionary mapping region names to keyword lists
        """
        if threshold is None:
            threshold = self.confidence_threshold
            
        region_keywords = {}
        
        for region_idx in range(min(len(self.region_names), region_keyword_logits.size(0))):
            region_name = self.region_names[region_idx]
            region_logits = region_keyword_logits[region_idx]
            
            # Decode keywords for this region
            keywords = self.decode_global_keywords(
                region_logits, 
                top_k=top_k, 
                threshold=threshold
            )
            
            # Add region-specific information
            for kw in keywords:
                kw["region"] = region_name
                kw["region_rank"] = kw["global_rank"]
                del kw["global_rank"]
                
            region_keywords[region_name] = keywords
            
        return region_keywords
        
    def _get_keyword_category(self, keyword: str) -> str:
        """
        Determine medical category for a keyword.
        
        Args:
            keyword: Medical keyword
            
        Returns:
            Category name
        """
        keyword_lower = keyword.lower()
        
        for category, keywords in self.medical_categories.items():
            if any(kw in keyword_lower for kw in keywords):
                return category
                
        # Check in vocabulary category frequencies
        for category, category_words in self.category_frequencies.items():
            if keyword in category_words:
                return category
                
        return "general"
        
    def get_keywords_by_category(self, 
                               keywords: List[Dict],
                               sort_by_confidence: bool = True) -> Dict[str, List[Dict]]:
        """
        Group keywords by medical category.
        
        Args:
            keywords: List of keyword dictionaries
            sort_by_confidence: Whether to sort within categories by confidence
            
        Returns:
            Dictionary mapping categories to keyword lists
        """
        categorized = {}
        
        for kw in keywords:
            category = kw["category"]
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(kw)
            
        if sort_by_confidence:
            for category in categorized:
                categorized[category].sort(key=lambda x: x["confidence"], reverse=True)
                
        return categorized
        
    def filter_keywords_by_relevance(self, 
                                   keywords: List[Dict],
                                   min_frequency: int = 2,
                                   exclude_categories: Optional[List[str]] = None) -> List[Dict]:
        """
        Filter keywords by relevance criteria.
        
        Args:
            keywords: List of keyword dictionaries
            min_frequency: Minimum frequency in training data
            exclude_categories: Categories to exclude
            
        Returns:
            Filtered keyword list
        """
        if exclude_categories is None:
            exclude_categories = ["negation"]
            
        filtered = []
        for kw in keywords:
            # Check frequency
            if kw["frequency"] < min_frequency:
                continue
                
            # Check category exclusion
            if kw["category"] in exclude_categories:
                continue
                
            filtered.append(kw)
            
        return filtered
        
    def get_medical_summary(self, 
                          global_keywords: List[Dict],
                          region_keywords: Dict[str, List[Dict]]) -> Dict:
        """
        Generate medical summary from keywords.
        
        Args:
            global_keywords: Global keyword predictions
            region_keywords: Region-specific keyword predictions
            
        Returns:
            Medical summary dictionary
        """
        # Categorize global keywords
        categorized = self.get_keywords_by_category(global_keywords)
        
        # Count region mentions
        region_mentions = {}
        for region, keywords in region_keywords.items():
            if keywords:  # Only count regions with predicted keywords
                region_mentions[region] = len(keywords)
                
        # Find most confident pathology
        pathology_keywords = categorized.get("pathology", [])
        primary_pathology = pathology_keywords[0]["keyword"] if pathology_keywords else "normal"
        
        # Find most confident anatomy
        anatomy_keywords = categorized.get("anatomy", [])
        primary_anatomy = anatomy_keywords[0]["keyword"] if anatomy_keywords else "chest"
        
        # Detect devices
        device_keywords = categorized.get("devices", [])
        devices_present = len(device_keywords) > 0
        
        return {
            "primary_finding": primary_pathology,
            "primary_anatomy": primary_anatomy,
            "devices_present": devices_present,
            "active_regions": len(region_mentions),
            "region_distribution": region_mentions,
            "keyword_categories": {cat: len(kws) for cat, kws in categorized.items()},
            "confidence_distribution": {
                "high": len([kw for kw in global_keywords if kw["confidence"] > 0.7]),
                "medium": len([kw for kw in global_keywords if 0.3 < kw["confidence"] <= 0.7]),
                "low": len([kw for kw in global_keywords if kw["confidence"] <= 0.3])
            }
        }
        
    def decode_medical_report(self, 
                            global_keywords: List[Dict],
                            region_keywords: Dict[str, List[Dict]],
                            disease_predictions: List[Dict]) -> str:
        """
        Generate a medical report summary from predictions.
        
        Args:
            global_keywords: Global keyword predictions
            region_keywords: Region-specific predictions
            disease_predictions: Disease classification results
            
        Returns:
            Medical report text
        """
        # Get positive disease predictions
        positive_diseases = [d["condition"] for d in disease_predictions if d["predicted"]]
        
        # Get top pathology keywords
        categorized = self.get_keywords_by_category(global_keywords)
        pathology_kws = [kw["keyword"] for kw in categorized.get("pathology", [])[:3]]
        anatomy_kws = [kw["keyword"] for kw in categorized.get("anatomy", [])[:3]]
        
        # Find active regions
        active_regions = [region for region, kws in region_keywords.items() if kws]
        
        # Build report
        report_parts = []
        
        if positive_diseases:
            report_parts.append(f"FINDINGS: {', '.join(positive_diseases)}")
        else:
            report_parts.append("FINDINGS: No acute cardiopulmonary findings")
            
        if pathology_kws:
            report_parts.append(f"KEY PATHOLOGY: {', '.join(pathology_kws)}")
            
        if anatomy_kws:
            report_parts.append(f"ANATOMICAL FOCUS: {', '.join(anatomy_kws)}")
            
        if active_regions:
            report_parts.append(f"ACTIVE REGIONS: {', '.join(active_regions[:5])}")
            
        return ". ".join(report_parts) + "." 