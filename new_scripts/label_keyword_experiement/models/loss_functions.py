"""
Multi-Task Loss Functions for Hybrid RadBERT + MedCLIP Model
============================================================

This module implements loss functions for multi-task learning combining:
- Disease classification loss (focal loss)
- Global keyword prediction loss (BCE)
- Region-specific keyword prediction loss (masked BCE)
- Consistency loss between global and region predictions
- Optional contrastive loss for vision-language alignment
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any

# Optional imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    
    # Create mock torch for type hints
    class MockTorch:
        class Tensor:
            pass
    torch = MockTorch()

# Fallback for when PyTorch is not available
if not TORCH_AVAILABLE:
    class MockModule:
        """Mock PyTorch module for when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            pass
        
        def __call__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Please install torch to use loss functions.")
    
    nn = type('nn', (), {'Module': MockModule})()


class FocalLoss(nn.Module):
    """
    Enhanced Focal Loss for addressing class imbalance in disease classification.
    
    Focal Loss = -α(1-pt)^γ * log(pt)
    where pt = p if y=1 else 1-p
    
    Enhanced with:
    - Higher gamma for focusing on hard examples
    - Optional class weights for severe imbalance
    - Label smoothing support
    """
    
    def __init__(self, 
                 alpha: float = 1.0, 
                 gamma: float = 3.0,  # Increased default gamma
                 reduction: str = 'mean',
                 class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        """
        Initialize Enhanced Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 3.0, increased from 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
            class_weights: Per-class weights [num_classes]
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super(FocalLoss, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch to use loss functions.")
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute enhanced focal loss with class weights and label smoothing.
        
        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes]
            
        Returns:
            Enhanced focal loss tensor
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Apply class weights if specified
        pos_weight = self.class_weights if self.class_weights is not None else None
        
        # Compute BCE loss with optional class weighting
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=pos_weight,
            reduction='none'
        )
        
        # Compute probabilities
        pt = torch.exp(-bce_loss)
        
        # Compute focal loss with higher gamma to focus more on hard examples
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class KeywordBCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss for keyword prediction with optional class weighting.
    """
    
    def __init__(self, 
                 pos_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        """
        Initialize Keyword BCE Loss.
        
        Args:
            pos_weight: Weight for positive class [vocab_size]
            reduction: Reduction method ('mean', 'sum', 'none')
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super(KeywordBCELoss, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch to use loss functions.")
        
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute keyword BCE loss.
        
        Args:
            inputs: Predicted logits [batch_size, vocab_size]
            targets: Ground truth labels [batch_size, vocab_size]
            
        Returns:
            BCE loss tensor
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Compute BCE loss with optional positive weighting
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )
        
        return loss


class RegionKeywordLoss(nn.Module):
    """
    Region-specific keyword loss with masking for unavailable regions.
    """
    
    def __init__(self, 
                 pos_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 normalize_by_regions: bool = True):
        """
        Initialize Region Keyword Loss.
        
        Args:
            pos_weight: Weight for positive class [vocab_size]
            reduction: Reduction method ('mean', 'sum', 'none')
            normalize_by_regions: Whether to normalize loss by number of available regions
        """
        super(RegionKeywordLoss, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch to use loss functions.")
        
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.normalize_by_regions = normalize_by_regions
    
    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor,
                region_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute region-specific keyword loss.
        
        Args:
            inputs: Predicted logits [batch_size, num_regions, vocab_size]
            targets: Ground truth labels [batch_size, num_regions, vocab_size]
            region_mask: Region availability mask [batch_size, num_regions]
            
        Returns:
            Region keyword loss tensor
        """
        batch_size, num_regions, vocab_size = inputs.shape
        
        # Reshape for BCE computation
        inputs_flat = inputs.view(-1, vocab_size)  # [batch*regions, vocab_size]
        targets_flat = targets.view(-1, vocab_size)  # [batch*regions, vocab_size]
        
        # Compute BCE loss for all region-keyword pairs
        loss_flat = F.binary_cross_entropy_with_logits(
            inputs_flat, targets_flat,
            pos_weight=self.pos_weight,
            reduction='none'
        )  # [batch*regions, vocab_size]
        
        # Reshape back to region structure
        loss_regions = loss_flat.view(batch_size, num_regions, vocab_size)  # [batch, regions, vocab_size]
        
        # Apply region masking if provided
        if region_mask is not None:
            # Expand mask to match loss dimensions
            mask_expanded = region_mask.unsqueeze(-1).expand(-1, -1, vocab_size)  # [batch, regions, vocab_size]
            loss_regions = loss_regions * mask_expanded.float()
            
            if self.normalize_by_regions:
                # Normalize by number of available regions per sample
                region_counts = region_mask.sum(dim=1, keepdim=True).float()  # [batch, 1]
                region_counts = torch.clamp(region_counts, min=1.0)  # Avoid division by zero
                loss_regions = loss_regions.sum(dim=1) / region_counts.unsqueeze(-1)  # [batch, vocab_size]
            else:
                loss_regions = loss_regions.sum(dim=1)  # [batch, vocab_size]
        else:
            # No masking - average across all regions
            loss_regions = loss_regions.mean(dim=1)  # [batch, vocab_size]
        
        # Apply final reduction
        if self.reduction == 'mean':
            return loss_regions.mean()
        elif self.reduction == 'sum':
            return loss_regions.sum()
        else:
            return loss_regions


class ConsistencyLoss(nn.Module):
    """
    Consistency loss between global and region-specific keyword predictions.
    """
    
    def __init__(self, loss_type: str = 'mse', reduction: str = 'mean'):
        """
        Initialize Consistency Loss.
        
        Args:
            loss_type: Type of consistency loss ('mse', 'kl', 'cosine')
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(ConsistencyLoss, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch to use loss functions.")
        
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(self, 
                global_logits: torch.Tensor,
                region_logits: torch.Tensor,
                region_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute consistency loss between global and region predictions.
        
        Args:
            global_logits: Global keyword logits [batch_size, vocab_size]
            region_logits: Region keyword logits [batch_size, num_regions, vocab_size]
            region_mask: Region availability mask [batch_size, num_regions]
            
        Returns:
            Consistency loss tensor
        """
        # Aggregate region predictions to global level
        if region_mask is not None:
            # Weighted average by available regions
            mask_expanded = region_mask.unsqueeze(-1).expand(-1, -1, region_logits.size(-1)).float()
            masked_region_logits = region_logits * mask_expanded
            region_counts = region_mask.sum(dim=1, keepdim=True).float()  # [batch, 1]
            region_counts = torch.clamp(region_counts, min=1.0)  # Avoid division by zero
            aggregated_region_logits = masked_region_logits.sum(dim=1) / region_counts  # [batch, vocab_size]
        else:
            # Simple average across all regions
            aggregated_region_logits = region_logits.mean(dim=1)  # [batch, vocab_size]
        
        # Compute consistency loss based on specified type
        if self.loss_type == 'mse':
            loss = F.mse_loss(global_logits, aggregated_region_logits, reduction=self.reduction)
        elif self.loss_type == 'kl':
            # KL divergence between probability distributions
            global_probs = torch.softmax(global_logits, dim=-1)
            region_probs = torch.softmax(aggregated_region_logits, dim=-1)
            loss = F.kl_div(region_probs.log(), global_probs, reduction=self.reduction)
        elif self.loss_type == 'cosine':
            # Cosine similarity loss
            cosine_sim = F.cosine_similarity(global_logits, aggregated_region_logits, dim=-1)
            loss = 1 - cosine_sim  # Convert similarity to loss
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            raise ValueError(f"Unknown consistency loss type: {self.loss_type}")
        
        return loss


class HardNegativeLoss(nn.Module):
    """
    Hard Negative Mining Loss for preventing "always positive" predictions.
    
    This loss heavily penalizes false positives to force the model to learn
    proper negative examples instead of predicting positive for everything.
    
    From improvements.md Priority 2: Force model to learn negative examples.
    """
    
    def __init__(self, 
                 pos_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 negative_weight_multiplier: float = 2.0):
        """
        Initialize Hard Negative Mining Loss.
        
        Args:
            pos_weight: Reduced positive weights to combat over-prediction
            reduction: Reduction method ('mean', 'sum', 'none')
            negative_weight_multiplier: Extra penalty for false positives
        """
        super(HardNegativeLoss, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch to use loss functions.")
        
        # 🔥 PRIORITY 2 FIX: Reduce positive bias - use pos_weight < 1.0
        if pos_weight is None:
            # Default: Equal weighting (0.5) to reduce positive bias
            self.pos_weight = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        else:
            self.pos_weight = pos_weight
            
        self.reduction = reduction
        self.negative_weight_multiplier = negative_weight_multiplier
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute hard negative mining loss.
        
        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes]
            
        Returns:
            Hard negative mining loss with extra penalty for false positives
        """
        # Ensure pos_weight is on the same device as inputs
        pos_weight = self.pos_weight.to(inputs.device) if self.pos_weight is not None else None
        
        # Standard BCE loss with reduced positive weights
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=pos_weight,
            reduction='none'
        )
        
        # Extra penalty for false positives (predictions > 0.5 when target = 0)
        predictions = torch.sigmoid(inputs)
        false_positives = (predictions > 0.5) & (targets == 0)
        
        # Apply extra penalty to false positives
        penalty = torch.where(
            false_positives, 
            bce_loss * self.negative_weight_multiplier,
            bce_loss
        )
        
        if self.reduction == 'mean':
            return penalty.mean()
        elif self.reduction == 'sum':
            return penalty.sum()
        else:
            return penalty


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for disease classification and keyword prediction.
    
    Combines:
    1. Disease classification loss (Focal Loss)
    2. Global keyword prediction loss (BCE)
    3. Region-specific keyword prediction loss (Masked BCE)
    4. Consistency loss between global and region predictions
    5. Optional contrastive loss for vision-language alignment
    """
    
    def __init__(self,
                 loss_weights: Optional[Dict[str, float]] = None,
                 disease_loss_config: Optional[Dict] = None,
                 keyword_loss_config: Optional[Dict] = None,
                 region_loss_config: Optional[Dict] = None,
                 consistency_loss_config: Optional[Dict] = None,
                 adaptive_weighting: bool = False):
        """
        Initialize Multi-Task Loss.
        
        Args:
            loss_weights: Dictionary of loss component weights
            disease_loss_config: Configuration for disease classification loss
            keyword_loss_config: Configuration for keyword prediction loss
            region_loss_config: Configuration for region keyword loss
            consistency_loss_config: Configuration for consistency loss
            adaptive_weighting: Whether to use adaptive loss weighting
        """
        super(MultiTaskLoss, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch to use loss functions.")
        
        # Default loss weights
        self.loss_weights = loss_weights or {
            'disease': 1.0,
            'keyword_global': 0.5,
            'keyword_region': 0.3,
            'consistency': 0.2,
            'contrastive': 0.1
        }
        
        # Initialize individual loss functions
        disease_config = disease_loss_config or {'alpha': 1.0, 'gamma': 2.0}
        self.disease_loss = FocalLoss(**disease_config)
        
        keyword_config = keyword_loss_config or {}
        self.keyword_loss = KeywordBCELoss(**keyword_config)
        
        region_config = region_loss_config or {}
        self.region_loss = RegionKeywordLoss(**region_config)
        
        consistency_config = consistency_loss_config or {'loss_type': 'mse'}
        self.consistency_loss = ConsistencyLoss(**consistency_config)
        
        # Adaptive weighting parameters
        self.adaptive_weighting = adaptive_weighting
        if adaptive_weighting:
            # Learnable loss weights (log-parameterized for stability)
            self.log_weights = nn.Parameter(torch.zeros(len(self.loss_weights)))
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                return_individual_losses: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dictionary with model predictions
            targets: Dictionary with target labels
            return_individual_losses: Whether to return breakdown of individual losses
            
        Returns:
            Total loss (scalar) or dictionary with individual losses
        """
        losses = {}
        
        # 1. Disease classification loss
        if ('disease_logits' in predictions and 'disease_labels' in targets and 
            targets['disease_labels'] is not None):
            losses['disease'] = self.disease_loss(
                predictions['disease_logits'], 
                targets['disease_labels']
            )
        
        # 2. Global keyword prediction loss - 🔥 FIX: Check for None targets
        if ('global_keyword_logits' in predictions and 'global_keyword_labels' in targets and 
            targets['global_keyword_labels'] is not None):
            losses['keyword_global'] = self.keyword_loss(
                predictions['global_keyword_logits'],
                targets['global_keyword_labels']
            )
        
        # 3. Region-specific keyword prediction loss - 🔥 FIX: Check for None targets
        if ('region_keyword_logits' in predictions and 'region_keyword_labels' in targets and
            targets['region_keyword_labels'] is not None):
            region_mask = targets.get('region_mask', None)
            losses['keyword_region'] = self.region_loss(
                predictions['region_keyword_logits'],
                targets['region_keyword_labels'],
                region_mask
            )
        
        # 4. Consistency loss between global and region predictions - 🔥 FIX: Check for None targets
        if ('global_keyword_logits' in predictions and 
            'region_keyword_logits' in predictions and
            targets.get('global_keyword_labels') is not None and
            targets.get('region_keyword_labels') is not None):
            region_mask = targets.get('region_mask', None)
            losses['consistency'] = self.consistency_loss(
                predictions['global_keyword_logits'],
                predictions['region_keyword_logits'],
                region_mask
            )
        
        # 5. MedCLIP contrastive loss
        if 'medclip_contrastive_loss' in predictions and predictions['medclip_contrastive_loss'] is not None:
            losses['medclip_contrastive'] = predictions['medclip_contrastive_loss']
        elif 'contrastive_loss' in predictions:
            # Backward compatibility
            losses['contrastive'] = predictions['contrastive_loss']
        
        # Compute weighted combination
        if self.adaptive_weighting:
            # Use learnable weights
            weights = torch.exp(self.log_weights)
            total_loss = sum(weights[i] * loss for i, (name, loss) in enumerate(losses.items()) 
                           if name in self.loss_weights)
        else:
            # Use fixed weights
            total_loss = sum(self.loss_weights.get(name, 0.0) * loss 
                           for name, loss in losses.items())
        
        if return_individual_losses:
            losses['total'] = total_loss
            return losses
        else:
            return total_loss
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights (useful for adaptive weighting)."""
        if self.adaptive_weighting:
            weights = torch.exp(self.log_weights).detach().cpu().numpy()
            return {name: float(weights[i]) for i, name in enumerate(self.loss_weights.keys())}
        else:
            return self.loss_weights.copy()
    
    def set_loss_weights(self, new_weights: Dict[str, float]):
        """Set new loss weights."""
        self.loss_weights.update(new_weights)
        if self.adaptive_weighting:
            # Update learnable parameters
            for i, name in enumerate(self.loss_weights.keys()):
                if name in new_weights:
                    self.log_weights.data[i] = math.log(new_weights[name])


# Utility functions for testing and debugging
def create_dummy_predictions(batch_size: int = 4, 
                           num_diseases: int = 8,
                           vocab_size: int = 100, 
                           num_regions: int = 17) -> Dict[str, torch.Tensor]:
    """Create dummy predictions for testing."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available.")
    
    return {
        'disease_logits': torch.randn(batch_size, num_diseases),
        'global_keyword_logits': torch.randn(batch_size, vocab_size),
        'region_keyword_logits': torch.randn(batch_size, num_regions, vocab_size)
    }


def create_dummy_targets(batch_size: int = 4,
                        num_diseases: int = 8,
                        vocab_size: int = 100,
                        num_regions: int = 17) -> Dict[str, torch.Tensor]:
    """Create dummy targets for testing."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available.")
    
    # Create realistic binary targets
    disease_labels = torch.randint(0, 2, (batch_size, num_diseases)).float()
    global_keyword_labels = torch.randint(0, 2, (batch_size, vocab_size)).float()
    region_keyword_labels = torch.randint(0, 2, (batch_size, num_regions, vocab_size)).float()
    
    # Create region mask (some regions unavailable)
    region_mask = torch.ones(batch_size, num_regions, dtype=torch.bool)
    for i in range(batch_size):
        num_unavailable = torch.randint(0, 5, (1,)).item()  # 0-4 regions unavailable
        if num_unavailable > 0:
            unavailable_indices = torch.randperm(num_regions)[:num_unavailable]
            region_mask[i, unavailable_indices] = False
    
    return {
        'disease_labels': disease_labels,
        'global_keyword_labels': global_keyword_labels,
        'region_keyword_labels': region_keyword_labels,
        'region_mask': region_mask
    }


def test_loss_functions():
    """Test function for multi-task loss functions."""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available - skipping loss functions test")
        return
    
    print("=== Testing Multi-Task Loss Functions ===")
    
    # Test parameters
    batch_size = 4
    num_diseases = 8
    vocab_size = 100
    num_regions = 17
    
    # Create dummy data
    predictions = create_dummy_predictions(batch_size, num_diseases, vocab_size, num_regions)
    targets = create_dummy_targets(batch_size, num_diseases, vocab_size, num_regions)
    
    print(f"Input shapes:")
    print(f"  Disease logits: {predictions['disease_logits'].shape}")
    print(f"  Global keyword logits: {predictions['global_keyword_logits'].shape}")
    print(f"  Region keyword logits: {predictions['region_keyword_logits'].shape}")
    print(f"  Region mask: {targets['region_mask'].shape}")
    
    # Test individual loss functions
    print(f"\n1. Testing individual loss functions...")
    
    # Focal Loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    disease_loss_val = focal_loss(predictions['disease_logits'], targets['disease_labels'])
    print(f"   Focal Loss: {disease_loss_val.item():.4f}")
    
    # Keyword BCE Loss
    keyword_bce = KeywordBCELoss()
    keyword_loss_val = keyword_bce(predictions['global_keyword_logits'], targets['global_keyword_labels'])
    print(f"   Keyword BCE Loss: {keyword_loss_val.item():.4f}")
    
    # Region Keyword Loss
    region_loss = RegionKeywordLoss()
    region_loss_val = region_loss(
        predictions['region_keyword_logits'], 
        targets['region_keyword_labels'],
        targets['region_mask']
    )
    print(f"   Region Keyword Loss: {region_loss_val.item():.4f}")
    
    # Consistency Loss
    consistency_loss = ConsistencyLoss()
    consistency_loss_val = consistency_loss(
        predictions['global_keyword_logits'],
        predictions['region_keyword_logits'],
        targets['region_mask']
    )
    print(f"   Consistency Loss: {consistency_loss_val.item():.4f}")
    
    # Test Multi-Task Loss
    print(f"\n2. Testing MultiTaskLoss...")
    
    # Fixed weighting
    multi_loss = MultiTaskLoss(adaptive_weighting=False)
    total_loss = multi_loss(predictions, targets)
    print(f"   Total Loss (fixed weights): {total_loss.item():.4f}")
    
    # Individual losses
    individual_losses = multi_loss(predictions, targets, return_individual_losses=True)
    print(f"   Individual losses:")
    for name, loss in individual_losses.items():
        print(f"     {name}: {loss.item():.4f}")
    
    # Adaptive weighting
    adaptive_multi_loss = MultiTaskLoss(adaptive_weighting=True)
    adaptive_total_loss = adaptive_multi_loss(predictions, targets)
    print(f"   Total Loss (adaptive weights): {adaptive_total_loss.item():.4f}")
    
    # Test weight management
    print(f"\n3. Testing weight management...")
    current_weights = multi_loss.get_loss_weights()
    print(f"   Current weights: {current_weights}")
    
    new_weights = {'disease': 2.0, 'keyword_global': 1.0}
    multi_loss.set_loss_weights(new_weights)
    updated_weights = multi_loss.get_loss_weights()
    print(f"   Updated weights: {updated_weights}")
    
    print(f"\n✅ All loss functions working correctly!")


if __name__ == "__main__":
    test_loss_functions() 