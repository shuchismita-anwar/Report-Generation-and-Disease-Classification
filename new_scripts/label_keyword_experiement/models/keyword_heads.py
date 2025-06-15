"""
Keyword Prediction Heads for Hybrid RadBERT + MedCLIP Model
===========================================================

This module implements prediction heads for keyword prediction in the multi-task learning setup.
Includes both global keyword prediction and region-specific keyword prediction heads.
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
            raise ImportError("PyTorch is not available. Please install torch to use these models.")
    
    nn = type('nn', (), {'Module': MockModule, 'Linear': MockModule, 'Dropout': MockModule, 
                         'LayerNorm': MockModule, 'MultiheadAttention': MockModule})()


class GlobalKeywordHead(nn.Module):
    """
    Global keyword prediction head that predicts keywords for the entire image.
    
    This head takes multimodal features and predicts a vocabulary-sized binary vector
    indicating which medical keywords are present in the image.
    """
    
    def __init__(self, 
                 hidden_dim: int, 
                 vocab_size: int,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True):
        """
        Initialize global keyword prediction head.
        
        Args:
            hidden_dim: Dimension of input features
            vocab_size: Size of medical vocabulary
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        super(GlobalKeywordHead, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch to use keyword heads.")
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout_prob = dropout
        self.use_layer_norm = use_layer_norm
        
        # Feature processing layers
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Keyword prediction layers
        self.keyword_projection = nn.Linear(hidden_dim, hidden_dim // 2)
        self.keyword_classifier = nn.Linear(hidden_dim // 2, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for global keyword prediction.
        
        Args:
            features: Input features [batch_size, hidden_dim]
            
        Returns:
            Keyword logits [batch_size, vocab_size]
        """
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            features = self.layer_norm(features)
        
        # Apply dropout
        features = self.dropout(features)
        
        # Project to intermediate dimension
        projected = F.relu(self.keyword_projection(features))
        projected = self.dropout(projected)
        
        # Classify keywords
        keyword_logits = self.keyword_classifier(projected)
        
        return keyword_logits
    
    def predict_keywords(self, 
                        features: torch.Tensor, 
                        threshold: float = 0.5,
                        top_k: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Predict keywords with post-processing.
        
        Args:
            features: Input features [batch_size, hidden_dim]
            threshold: Sigmoid threshold for binary prediction
            top_k: If provided, return top-k keywords instead of threshold-based
            
        Returns:
            Dictionary with logits, probabilities, and predictions
        """
        with torch.no_grad():
            logits = self.forward(features)
            probabilities = torch.sigmoid(logits)
            
            if top_k is not None:
                # Top-k prediction
                top_probs, top_indices = torch.topk(probabilities, top_k, dim=-1)
                predictions = torch.zeros_like(probabilities)
                predictions.scatter_(-1, top_indices, 1.0)
            else:
                # Threshold-based prediction
                predictions = (probabilities > threshold).float()
            
            return {
                'logits': logits,
                'probabilities': probabilities,
                'predictions': predictions,
                'top_indices': top_indices if top_k else None,
                'top_probabilities': top_probs if top_k else None
            }


class RegionKeywordHead(nn.Module):
    """
    Region-specific keyword prediction head that predicts keywords for each anatomical region.
    
    This head takes multimodal features and region availability masks to predict
    region-specific keyword vectors for each of the 17 anatomical regions.
    """
    
    def __init__(self, 
                 hidden_dim: int, 
                 vocab_size: int, 
                 num_regions: int = 17,
                 dropout: float = 0.1,
                 use_attention: bool = True,
                 use_layer_norm: bool = True):
        """
        Initialize region-specific keyword prediction head.
        
        Args:
            hidden_dim: Dimension of input features
            vocab_size: Size of medical vocabulary
            num_regions: Number of anatomical regions (default 17)
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism for region focus
            use_layer_norm: Whether to use layer normalization
        """
        super(RegionKeywordHead, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch to use keyword heads.")
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_regions = num_regions
        self.dropout_prob = dropout
        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        
        # Feature processing layers
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Region-specific processing
        if use_attention:
            # Attention mechanism for region focus
            self.region_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            # Learnable region queries
            self.region_queries = nn.Parameter(torch.randn(num_regions, hidden_dim))
        
        # Region-specific keyword prediction
        self.region_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, vocab_size)
            ) for _ in range(num_regions)
        ])
        
        # Alternative: Shared projection with region-specific heads
        self.shared_projection = nn.Linear(hidden_dim, hidden_dim // 2)
        self.region_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim // 2, vocab_size) for _ in range(num_regions)
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Initialize region queries if using attention
        if self.use_attention:
            nn.init.xavier_uniform_(self.region_queries)
    
    def forward(self, 
                features: torch.Tensor, 
                region_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for region-specific keyword prediction.
        
        Args:
            features: Input features [batch_size, hidden_dim]
            region_mask: Region availability mask [batch_size, num_regions]
            
        Returns:
            Region keyword logits [batch_size, num_regions, vocab_size]
        """
        batch_size = features.size(0)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            features = self.layer_norm(features)
        
        # Apply dropout
        features = self.dropout(features)
        
        if self.use_attention:
            # Use attention mechanism for region-specific features
            region_features = self._compute_region_features_with_attention(features)
        else:
            # Use same features for all regions
            region_features = features.unsqueeze(1).expand(-1, self.num_regions, -1)
        
        # Predict keywords for each region
        region_keyword_logits = []
        
        # Shared projection approach
        shared_features = F.relu(self.shared_projection(region_features))
        shared_features = self.dropout(shared_features)
        
        for region_idx in range(self.num_regions):
            region_logits = self.region_classifiers[region_idx](shared_features[:, region_idx, :])
            region_keyword_logits.append(region_logits)
        
        # Stack region predictions
        region_keyword_logits = torch.stack(region_keyword_logits, dim=1)  # [batch, num_regions, vocab_size]
        
        # Apply region mask if provided
        if region_mask is not None:
            # Mask unavailable regions (set logits to very negative values)
            mask_expanded = region_mask.unsqueeze(-1).expand(-1, -1, self.vocab_size)
            region_keyword_logits = region_keyword_logits.masked_fill(~mask_expanded, -1e9)
        
        return region_keyword_logits
    
    def _compute_region_features_with_attention(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute region-specific features using attention mechanism.
        
        Args:
            features: Input features [batch_size, hidden_dim]
            
        Returns:
            Region-specific features [batch_size, num_regions, hidden_dim]
        """
        batch_size = features.size(0)
        
        # Expand region queries for batch
        queries = self.region_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_regions, hidden_dim]
        
        # Use features as keys and values
        keys = features.unsqueeze(1).expand(-1, self.num_regions, -1)  # [batch, num_regions, hidden_dim]
        values = keys
        
        # Apply attention
        region_features, attention_weights = self.region_attention(
            query=queries,
            key=keys,
            value=values
        )
        
        return region_features
    
    def predict_keywords(self, 
                        features: torch.Tensor,
                        region_mask: Optional[torch.Tensor] = None,
                        threshold: float = 0.5,
                        top_k: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Predict region-specific keywords with post-processing.
        
        Args:
            features: Input features [batch_size, hidden_dim]
            region_mask: Region availability mask [batch_size, num_regions]
            threshold: Sigmoid threshold for binary prediction
            top_k: If provided, return top-k keywords per region
            
        Returns:
            Dictionary with logits, probabilities, and predictions
        """
        with torch.no_grad():
            logits = self.forward(features, region_mask)
            probabilities = torch.sigmoid(logits)
            
            if top_k is not None:
                # Top-k prediction per region
                top_probs, top_indices = torch.topk(probabilities, top_k, dim=-1)
                predictions = torch.zeros_like(probabilities)
                predictions.scatter_(-1, top_indices, 1.0)
            else:
                # Threshold-based prediction
                predictions = (probabilities > threshold).float()
            
            # Apply region mask to predictions if provided
            if region_mask is not None:
                mask_expanded = region_mask.unsqueeze(-1).expand(-1, -1, self.vocab_size)
                predictions = predictions * mask_expanded.float()
            
            return {
                'logits': logits,
                'probabilities': probabilities,
                'predictions': predictions,
                'region_mask': region_mask,
                'top_indices': top_indices if top_k else None,
                'top_probabilities': top_probs if top_k else None
            }


class CombinedKeywordHead(nn.Module):
    """
    Combined keyword prediction head that includes both global and region-specific prediction.
    
    This head coordinates between global and region-specific keyword prediction,
    potentially using consistency losses and feature sharing.
    """
    
    def __init__(self, 
                 hidden_dim: int, 
                 vocab_size: int, 
                 num_regions: int = 17,
                 dropout: float = 0.1,
                 use_consistency_loss: bool = True,
                 consistency_weight: float = 0.1):
        """
        Initialize combined keyword prediction head.
        
        Args:
            hidden_dim: Dimension of input features
            vocab_size: Size of medical vocabulary
            num_regions: Number of anatomical regions
            dropout: Dropout probability
            use_consistency_loss: Whether to enforce global-region consistency
            consistency_weight: Weight for consistency loss
        """
        super(CombinedKeywordHead, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch to use keyword heads.")
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_regions = num_regions
        self.use_consistency_loss = use_consistency_loss
        self.consistency_weight = consistency_weight
        
        # Global and region-specific heads
        self.global_head = GlobalKeywordHead(hidden_dim, vocab_size, dropout)
        self.region_head = RegionKeywordHead(hidden_dim, vocab_size, num_regions, dropout)
        
        # Consistency enforcement layers
        if use_consistency_loss:
            self.consistency_projection = nn.Linear(vocab_size * num_regions, vocab_size)
    
    def forward(self, 
                features: torch.Tensor, 
                region_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for combined keyword prediction.
        
        Args:
            features: Input features [batch_size, hidden_dim]
            region_mask: Region availability mask [batch_size, num_regions]
            
        Returns:
            Dictionary with global and region-specific predictions
        """
        # Global keyword prediction
        global_logits = self.global_head(features)
        
        # Region-specific keyword prediction
        region_logits = self.region_head(features, region_mask)
        
        results = {
            'global_logits': global_logits,
            'region_logits': region_logits
        }
        
        # Compute consistency loss if enabled
        if self.use_consistency_loss:
            consistency_loss = self._compute_consistency_loss(global_logits, region_logits, region_mask)
            results['consistency_loss'] = consistency_loss
        
        return results
    
    def _compute_consistency_loss(self, 
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
            Consistency loss scalar
        """
        # Aggregate region predictions to global level
        if region_mask is not None:
            # Weighted average by available regions
            mask_expanded = region_mask.unsqueeze(-1).expand(-1, -1, self.vocab_size).float()
            masked_region_logits = region_logits * mask_expanded
            region_counts = region_mask.sum(dim=1, keepdim=True).float()  # [batch, 1]
            region_counts = torch.clamp(region_counts, min=1.0)  # Avoid division by zero
            aggregated_region_logits = masked_region_logits.sum(dim=1) / region_counts  # [batch, vocab_size]
        else:
            # Simple average across all regions
            aggregated_region_logits = region_logits.mean(dim=1)  # [batch, vocab_size]
        
        # Compute consistency loss (MSE between global and aggregated region predictions)
        consistency_loss = F.mse_loss(global_logits, aggregated_region_logits)
        
        return consistency_loss * self.consistency_weight
    
    def predict_keywords(self, 
                        features: torch.Tensor,
                        region_mask: Optional[torch.Tensor] = None,
                        threshold: float = 0.5,
                        top_k: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Predict keywords using both global and region-specific heads.
        
        Args:
            features: Input features [batch_size, hidden_dim]
            region_mask: Region availability mask [batch_size, num_regions]
            threshold: Sigmoid threshold for binary prediction
            top_k: If provided, return top-k keywords
            
        Returns:
            Dictionary with global and region-specific predictions
        """
        with torch.no_grad():
            # Get forward pass results
            results = self.forward(features, region_mask)
            
            # Get individual predictions from each head
            global_predictions = self.global_head.predict_keywords(
                features, threshold=threshold, top_k=top_k
            )
            
            region_predictions = self.region_head.predict_keywords(
                features, region_mask=region_mask, threshold=threshold, top_k=top_k
            )
            
            # Combine results
            combined_results = {
                'global_logits': results['global_logits'],
                'region_logits': results['region_logits'],
                'global_predictions': global_predictions['predictions'],
                'region_predictions': region_predictions['predictions'],
                'global_probabilities': global_predictions['probabilities'],
                'region_probabilities': region_predictions['probabilities']
            }
            
            # Add consistency loss if computed
            if 'consistency_loss' in results:
                combined_results['consistency_loss'] = results['consistency_loss']
            
            # Add top-k results if requested
            if top_k is not None:
                combined_results['global_top_indices'] = global_predictions.get('top_indices')
                combined_results['global_top_probabilities'] = global_predictions.get('top_probabilities')
                combined_results['region_top_indices'] = region_predictions.get('top_indices')
                combined_results['region_top_probabilities'] = region_predictions.get('top_probabilities')
            
            return combined_results


# Utility functions for testing and debugging
def create_dummy_features(batch_size: int = 4, hidden_dim: int = 768) -> torch.Tensor:
    """Create dummy features for testing."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available.")
    return torch.randn(batch_size, hidden_dim)


def create_dummy_region_mask(batch_size: int = 4, num_regions: int = 17) -> torch.Tensor:
    """Create dummy region mask for testing."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available.")
    # Random region availability (at least 5 regions available per sample)
    mask = torch.zeros(batch_size, num_regions, dtype=torch.bool)
    for i in range(batch_size):
        num_available = torch.randint(5, num_regions, (1,)).item()
        available_indices = torch.randperm(num_regions)[:num_available]
        mask[i, available_indices] = True
    return mask


def test_keyword_heads():
    """Test function for keyword prediction heads."""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available - skipping keyword heads test")
        return
    
    print("=== Testing Keyword Prediction Heads ===")
    
    # Test parameters
    batch_size = 4
    hidden_dim = 768
    vocab_size = 100
    num_regions = 17
    
    # Create dummy data
    features = create_dummy_features(batch_size, hidden_dim)
    region_mask = create_dummy_region_mask(batch_size, num_regions)
    
    print(f"Input shapes:")
    print(f"  Features: {features.shape}")
    print(f"  Region mask: {region_mask.shape}")
    
    # Test Global Keyword Head
    print(f"\n1. Testing GlobalKeywordHead...")
    global_head = GlobalKeywordHead(hidden_dim, vocab_size)
    global_logits = global_head(features)
    print(f"   Global logits shape: {global_logits.shape}")
    
    global_predictions = global_head.predict_keywords(features, top_k=10)
    print(f"   Global predictions keys: {list(global_predictions.keys())}")
    
    # Test Region Keyword Head
    print(f"\n2. Testing RegionKeywordHead...")
    region_head = RegionKeywordHead(hidden_dim, vocab_size, num_regions)
    region_logits = region_head(features, region_mask)
    print(f"   Region logits shape: {region_logits.shape}")
    
    region_predictions = region_head.predict_keywords(features, region_mask, top_k=5)
    print(f"   Region predictions keys: {list(region_predictions.keys())}")
    
    # Test Combined Keyword Head
    print(f"\n3. Testing CombinedKeywordHead...")
    combined_head = CombinedKeywordHead(hidden_dim, vocab_size, num_regions)
    combined_results = combined_head(features, region_mask)
    print(f"   Combined results keys: {list(combined_results.keys())}")
    print(f"   Global logits shape: {combined_results['global_logits'].shape}")
    print(f"   Region logits shape: {combined_results['region_logits'].shape}")
    
    if 'consistency_loss' in combined_results:
        print(f"   Consistency loss: {combined_results['consistency_loss'].item():.4f}")
    
    print(f"\n✅ All keyword heads working correctly!")


if __name__ == "__main__":
    test_keyword_heads() 