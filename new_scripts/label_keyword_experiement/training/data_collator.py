"""
Enhanced Data Collator for Multi-Task Training
==============================================

This module provides data collation functionality for the hybrid model training,
supporting both disease classification and keyword prediction tasks.

The HybridDataCollator extends the basic multimodal collation to include:
- Keyword target tensors (global and region-specific)
- Region availability masks
- Variable vocabulary sizes
- Graceful handling of missing keyword data
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)

# PyTorch availability check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Data collator will not function properly.")

@dataclass
class HybridDataCollator:
    """
    Enhanced data collator for multi-task training with disease classification and keyword prediction.
    
    Handles:
    - Standard multimodal inputs (image, bbox, fixation, transcript)
    - Disease labels
    - Global keyword labels
    - Region-specific keyword labels
    - Region availability masks
    - Variable sequence lengths and vocabulary sizes
    """
    
    vocab_size: int = 35
    num_regions: int = 17
    max_fixation_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __post_init__(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for HybridDataCollator")
        
        logger.info(f"HybridDataCollator initialized:")
        logger.info(f"  - Vocabulary size: {self.vocab_size}")
        logger.info(f"  - Number of regions: {self.num_regions}")
        logger.info(f"  - Max fixation length: {self.max_fixation_length}")
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of features for multi-task training.
        
        Args:
            features: List of feature dictionaries from dataset
            
        Returns:
            Batched dictionary with all necessary tensors
        """
        if not features:
            raise ValueError("Cannot collate empty feature list")
        
        batch_size = len(features)
        batch = {}
        
        # 1. Collate standard multimodal inputs
        batch.update(self._collate_multimodal_inputs(features))
        
        # 2. Collate disease labels
        batch.update(self._collate_disease_labels(features))
        
        # 3. Collate keyword labels (if available)
        batch.update(self._collate_keyword_labels(features))
        
        # 4. Collate region masks (if available)
        batch.update(self._collate_region_masks(features))
        
        # 5. Add batch metadata
        batch["batch_size"] = batch_size
        
        logger.debug(f"Collated batch with {batch_size} samples")
        return batch
    
    def _collate_multimodal_inputs(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate standard multimodal inputs (image, bbox, fixation, transcript)."""
        batch = {}
        
        # Stack image tensors
        if "img" in features[0]:
            batch["img"] = torch.stack([f["img"] for f in features])
        
        # Stack bbox tensors
        if "bbox" in features[0]:
            batch["bbox"] = torch.stack([f["bbox"] for f in features])
        
        # Pad and stack fixation sequences
        if "fix_seq" in features[0]:
            fix_data = self._collate_fixation_sequences(features)
            batch.update(fix_data)
        
        # Collect transcript strings
        if "transcript" in features[0]:
            batch["transcript"] = [f["transcript"] for f in features]
        
        return batch
    
    def _collate_fixation_sequences(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate and pad fixation sequences."""
        fix_seqs = []
        fix_masks = []
        
        # Determine maximum sequence length
        max_len = max(f["fix_seq"].size(0) for f in features)
        
        # Apply maximum length constraint if specified
        if self.max_fixation_length is not None:
            max_len = min(max_len, self.max_fixation_length)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        for f in features:
            seq = f["fix_seq"]
            mask = f.get("fix_mask", torch.ones(seq.size(0), dtype=torch.bool))
            
            # Truncate if necessary
            if seq.size(0) > max_len:
                seq = seq[:max_len]
                mask = mask[:max_len]
            
            # Pad sequence and mask
            pad_len = max_len - seq.size(0)
            if pad_len > 0:
                seq_pad = torch.zeros(pad_len, seq.size(1), dtype=seq.dtype)
                mask_pad = torch.zeros(pad_len, dtype=torch.bool)
                
                seq = torch.cat([seq, seq_pad], dim=0)
                mask = torch.cat([mask, mask_pad], dim=0)
            
            fix_seqs.append(seq)
            fix_masks.append(mask)
        
        return {
            "fix_seq": torch.stack(fix_seqs),
            "fix_mask": torch.stack(fix_masks)
        }
    
    def _collate_disease_labels(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate disease classification labels."""
        batch = {}
        
        if "labels" in features[0]:
            batch["labels"] = torch.stack([f["labels"] for f in features])
        elif "disease_labels" in features[0]:
            batch["labels"] = torch.stack([f["disease_labels"] for f in features])
        
        return batch
    
    def _collate_keyword_labels(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate keyword prediction labels (global and region-specific)."""
        batch = {}
        
        # Check if any feature has global keywords
        has_global_keywords = any("global_keywords" in f for f in features)
        if has_global_keywords:
            global_keywords = []
            for f in features:
                if "global_keywords" in f and isinstance(f["global_keywords"], torch.Tensor):
                    kw = f["global_keywords"]
                    # Ensure correct vocabulary size
                    if kw.size(-1) != self.vocab_size:
                        kw = self._resize_keyword_tensor(kw, self.vocab_size)
                    global_keywords.append(kw)
                else:
                    # Create zero tensor if missing
                    global_keywords.append(torch.zeros(self.vocab_size))
            
            batch["global_keywords"] = torch.stack(global_keywords)
        
        # Check if any feature has region keywords
        has_region_keywords = any("region_keywords" in f for f in features)
        if has_region_keywords:
            region_keywords = []
            for f in features:
                if "region_keywords" in f and isinstance(f["region_keywords"], torch.Tensor):
                    kw = f["region_keywords"]
                    # Ensure correct shape [num_regions, vocab_size]
                    if kw.shape != (self.num_regions, self.vocab_size):
                        kw = self._resize_region_keyword_tensor(kw, self.num_regions, self.vocab_size)
                    region_keywords.append(kw)
                else:
                    # Create zero tensor if missing
                    region_keywords.append(torch.zeros(self.num_regions, self.vocab_size))
            
            batch["region_keywords"] = torch.stack(region_keywords)
        
        return batch
    
    def _collate_region_masks(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate region availability masks."""
        batch = {}
        
        # Check if any feature has region masks
        has_region_masks = any("region_mask" in f for f in features)
        if has_region_masks:
            region_masks = []
            for f in features:
                if "region_mask" in f and isinstance(f["region_mask"], torch.Tensor):
                    mask = f["region_mask"]
                    # Ensure correct number of regions
                    if mask.size(-1) != self.num_regions:
                        mask = self._resize_region_mask(mask, self.num_regions)
                    region_masks.append(mask)
                else:
                    # Create default mask (all regions available)
                    region_masks.append(torch.ones(self.num_regions, dtype=torch.bool))
            
            batch["region_mask"] = torch.stack(region_masks)
        else:
            # Create default region masks if not provided
            batch["region_mask"] = torch.ones(len(features), self.num_regions, dtype=torch.bool)
        
        return batch
    
    def _resize_keyword_tensor(self, tensor: torch.Tensor, target_vocab_size: int) -> torch.Tensor:
        """Resize keyword tensor to match target vocabulary size."""
        current_size = tensor.size(-1)
        
        if current_size == target_vocab_size:
            return tensor
        elif current_size < target_vocab_size:
            # Pad with zeros
            pad_size = target_vocab_size - current_size
            padding = torch.zeros(*tensor.shape[:-1], pad_size, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=-1)
        else:
            # Truncate
            return tensor[..., :target_vocab_size]
    
    def _resize_region_keyword_tensor(self, tensor: torch.Tensor, target_regions: int, target_vocab_size: int) -> torch.Tensor:
        """Resize region keyword tensor to match target shape."""
        current_regions, current_vocab = tensor.shape[-2:]
        
        # Resize regions dimension
        if current_regions != target_regions:
            if current_regions < target_regions:
                # Pad regions with zeros
                pad_regions = target_regions - current_regions
                region_padding = torch.zeros(*tensor.shape[:-2], pad_regions, current_vocab, dtype=tensor.dtype)
                tensor = torch.cat([tensor, region_padding], dim=-2)
            else:
                # Truncate regions
                tensor = tensor[..., :target_regions, :]
        
        # Resize vocabulary dimension
        if current_vocab != target_vocab_size:
            if current_vocab < target_vocab_size:
                # Pad vocabulary with zeros
                pad_vocab = target_vocab_size - current_vocab
                vocab_padding = torch.zeros(*tensor.shape[:-1], pad_vocab, dtype=tensor.dtype)
                tensor = torch.cat([tensor, vocab_padding], dim=-1)
            else:
                # Truncate vocabulary
                tensor = tensor[..., :target_vocab_size]
        
        return tensor
    
    def _resize_region_mask(self, mask: torch.Tensor, target_regions: int) -> torch.Tensor:
        """Resize region mask to match target number of regions."""
        current_regions = mask.size(-1)
        
        if current_regions == target_regions:
            return mask
        elif current_regions < target_regions:
            # Pad with True (assume new regions are available)
            pad_size = target_regions - current_regions
            padding = torch.ones(*mask.shape[:-1], pad_size, dtype=torch.bool)
            return torch.cat([mask, padding], dim=-1)
        else:
            # Truncate
            return mask[..., :target_regions]


@dataclass
class BasicDataCollator:
    """
    Basic data collator for disease-only training (backward compatibility).
    
    This is a simplified version that only handles disease classification
    without keyword prediction capabilities.
    """
    
    max_fixation_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate features for disease-only training."""
        if not features:
            raise ValueError("Cannot collate empty feature list")
        
        batch = {}
        
        # Stack image tensors
        if "img" in features[0]:
            batch["img"] = torch.stack([f["img"] for f in features])
        
        # Stack bbox tensors
        if "bbox" in features[0]:
            batch["bbox"] = torch.stack([f["bbox"] for f in features])
        
        # Pad fixation sequences
        if "fix_seq" in features[0]:
            max_len = max(f["fix_seq"].size(0) for f in features)
            if self.max_fixation_length is not None:
                max_len = min(max_len, self.max_fixation_length)
            
            fix_seqs = []
            fix_masks = []
            
            for f in features:
                seq = f["fix_seq"]
                mask = f.get("fix_mask", torch.ones(seq.size(0), dtype=torch.bool))
                
                # Truncate if necessary
                if seq.size(0) > max_len:
                    seq = seq[:max_len]
                    mask = mask[:max_len]
                
                # Pad
                pad_len = max_len - seq.size(0)
                if pad_len > 0:
                    seq = torch.cat([seq, torch.zeros(pad_len, seq.size(1))], dim=0)
                    mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool)], dim=0)
                
                fix_seqs.append(seq)
                fix_masks.append(mask)
            
            batch["fix_seq"] = torch.stack(fix_seqs)
            batch["fix_mask"] = torch.stack(fix_masks)
        
        # Collect transcripts
        if "transcript" in features[0]:
            batch["transcript"] = [f["transcript"] for f in features]
        
        # Stack labels
        if "labels" in features[0]:
            batch["labels"] = torch.stack([f["labels"] for f in features])
        
        return batch


def create_data_collator(
    collator_type: str = "hybrid",
    vocab_size: int = 35,
    num_regions: int = 17,
    max_fixation_length: Optional[int] = None,
    **kwargs
) -> Union[HybridDataCollator, BasicDataCollator]:
    """
    Factory function to create appropriate data collator.
    
    Args:
        collator_type: Type of collator ("hybrid" or "basic")
        vocab_size: Size of keyword vocabulary
        num_regions: Number of anatomical regions
        max_fixation_length: Maximum fixation sequence length
        **kwargs: Additional arguments for collator
        
    Returns:
        Configured data collator instance
    """
    if collator_type == "hybrid":
        return HybridDataCollator(
            vocab_size=vocab_size,
            num_regions=num_regions,
            max_fixation_length=max_fixation_length,
            **kwargs
        )
    elif collator_type == "basic":
        return BasicDataCollator(
            max_fixation_length=max_fixation_length,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown collator type: {collator_type}")


# Utility functions for testing and debugging
def create_dummy_features(
    batch_size: int = 4,
    vocab_size: int = 35,
    num_regions: int = 17,
    include_keywords: bool = True
) -> List[Dict[str, Any]]:
    """Create dummy features for testing the data collator."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for creating dummy features")
    
    features = []
    
    for i in range(batch_size):
        fix_len = torch.randint(5, 15, (1,)).item()
        feature = {
            "img": torch.randn(3, 224, 224),
            "bbox": torch.randn(1, 224, 224),
            "fix_seq": torch.randn(fix_len, 4),
            "fix_mask": torch.ones(fix_len, dtype=torch.bool),
            "transcript": f"Sample medical transcript {i}",
            "labels": torch.randint(0, 2, (8,)).float()
        }
        
        if include_keywords:
            feature.update({
                "global_keywords": torch.randint(0, 2, (vocab_size,)).float(),
                "region_keywords": torch.randint(0, 2, (num_regions, vocab_size)).float(),
                "region_mask": torch.ones(num_regions, dtype=torch.bool)
            })
            
            # Randomly make some regions unavailable
            num_unavailable = torch.randint(0, 3, (1,)).item()
            if num_unavailable > 0:
                unavailable_indices = torch.randperm(num_regions)[:num_unavailable]
                feature["region_mask"][unavailable_indices] = False
        
        features.append(feature)
    
    return features


def test_data_collators():
    """Test function for data collators."""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available - skipping data collator test")
        return
    
    print("=== Testing Data Collators ===")
    
    # Test parameters
    batch_size = 4
    vocab_size = 35
    num_regions = 17
    
    # Create dummy features
    features_with_keywords = create_dummy_features(batch_size, vocab_size, num_regions, include_keywords=True)
    features_without_keywords = create_dummy_features(batch_size, vocab_size, num_regions, include_keywords=False)
    
    print(f"Created {batch_size} dummy features")
    
    # Test HybridDataCollator
    print(f"\n1. Testing HybridDataCollator...")
    hybrid_collator = HybridDataCollator(vocab_size=vocab_size, num_regions=num_regions)
    
    batch_with_keywords = hybrid_collator(features_with_keywords)
    print(f"   Batch with keywords:")
    for key, value in batch_with_keywords.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"     {key}: list with {len(value)} items")
        else:
            print(f"     {key}: {value}")
    
    batch_without_keywords = hybrid_collator(features_without_keywords)
    print(f"   Batch without keywords:")
    for key, value in batch_without_keywords.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"     {key}: list with {len(value)} items")
        else:
            print(f"     {key}: {value}")
    
    # Test BasicDataCollator
    print(f"\n2. Testing BasicDataCollator...")
    basic_collator = BasicDataCollator()
    
    basic_batch = basic_collator(features_without_keywords)
    print(f"   Basic batch:")
    for key, value in basic_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"     {key}: list with {len(value)} items")
        else:
            print(f"     {key}: {value}")
    
    # Test factory function
    print(f"\n3. Testing factory function...")
    hybrid_from_factory = create_data_collator("hybrid", vocab_size=vocab_size, num_regions=num_regions)
    basic_from_factory = create_data_collator("basic")
    
    print(f"   Hybrid collator from factory: {type(hybrid_from_factory).__name__}")
    print(f"   Basic collator from factory: {type(basic_from_factory).__name__}")
    
    print(f"\n✅ All data collators working correctly!")


if __name__ == "__main__":
    test_data_collators() 