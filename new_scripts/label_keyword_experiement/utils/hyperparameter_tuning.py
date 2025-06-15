"""
Hyperparameter Tuning for Hybrid Model
======================================

Systematic optimization of multi-task loss weights and learning parameters
for the RadBERT + MedCLIP hybrid medical model.

Key areas for optimization:
1. Multi-task loss weight combinations
2. Learning rates for different components
3. Batch size and gradient accumulation
4. Keyword vocabulary size
5. Model architecture parameters
"""

import torch
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from itertools import product
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass 
class HyperparameterConfig:
    """Configuration for hyperparameter search spaces."""
    
    # Loss weight combinations
    disease_weight: float = 1.0
    global_keyword_weight: float = 0.5
    region_keyword_weight: float = 0.3
    radbert_semantic_weight: float = 0.2
    medclip_contrastive_weight: float = 0.3
    
    # Learning rates
    base_learning_rate: float = 3e-5
    backbone_lr_multiplier: float = 0.5  # Lower LR for pretrained backbone
    keyword_head_lr_multiplier: float = 1.0
    medclip_lr_multiplier: float = 0.8
    
    # Training parameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    
    # Model parameters
    vocab_size: int = 1000
    dropout_rate: float = 0.15
    hidden_dim: int = 768
    
    # Training schedule
    num_epochs: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    
    def get_effective_batch_size(self) -> int:
        """Get effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    def get_total_loss_weight(self) -> float:
        """Get sum of all loss weights for normalization."""
        return (self.disease_weight + self.global_keyword_weight + 
                self.region_keyword_weight + self.radbert_semantic_weight + 
                self.medclip_contrastive_weight)


class HyperparameterSearchSpace:
    """Define search spaces for different hyperparameters."""
    
    def __init__(self):
        # Loss weight search spaces
        self.loss_weight_combinations = [
            # Balanced approach
            {"disease": 1.0, "global_kw": 0.5, "region_kw": 0.3, "radbert": 0.2, "medclip": 0.3},
            # Disease-focused
            {"disease": 1.5, "global_kw": 0.3, "region_kw": 0.2, "radbert": 0.1, "medclip": 0.2},
            # Keyword-focused  
            {"disease": 0.8, "global_kw": 0.8, "region_kw": 0.5, "radbert": 0.3, "medclip": 0.4},
            # Semantic-focused
            {"disease": 1.0, "global_kw": 0.4, "region_kw": 0.3, "radbert": 0.4, "medclip": 0.5},
            # Minimal multi-task
            {"disease": 1.0, "global_kw": 0.1, "region_kw": 0.1, "radbert": 0.05, "medclip": 0.1},
        ]
        
        # Learning rate combinations
        self.learning_rate_configs = [
            {"base": 2e-5, "backbone_mult": 0.3, "keyword_mult": 1.0, "medclip_mult": 0.5},
            {"base": 3e-5, "backbone_mult": 0.5, "keyword_mult": 1.0, "medclip_mult": 0.8},
            {"base": 5e-5, "backbone_mult": 0.7, "keyword_mult": 1.2, "medclip_mult": 1.0},
            {"base": 1e-4, "backbone_mult": 0.4, "keyword_mult": 0.8, "medclip_mult": 0.6},
        ]
        
        # Batch size and accumulation combinations
        self.batch_configs = [
            {"batch_size": 4, "grad_accum": 8},   # Effective: 32
            {"batch_size": 8, "grad_accum": 4},   # Effective: 32
            {"batch_size": 16, "grad_accum": 2},  # Effective: 32
            {"batch_size": 6, "grad_accum": 6},   # Effective: 36
        ]
        
        # Vocabulary sizes to test
        self.vocab_sizes = [500, 1000, 1500, 2000]
        
        # Dropout rates
        self.dropout_rates = [0.1, 0.15, 0.2]
        
    def get_grid_search_configs(self, search_type: str = "loss_weights") -> List[Dict]:
        """Get configurations for grid search."""
        if search_type == "loss_weights":
            return self.loss_weight_combinations
        elif search_type == "learning_rates":
            return self.learning_rate_configs  
        elif search_type == "batch_configs":
            return self.batch_configs
        elif search_type == "vocab_sizes":
            return [{"vocab_size": size} for size in self.vocab_sizes]
        elif search_type == "dropout_rates":
            return [{"dropout_rate": rate} for rate in self.dropout_rates]
        else:
            raise ValueError(f"Unknown search type: {search_type}")


class HyperparameterTuner:
    """
    Systematic hyperparameter tuning for the hybrid model.
    
    Performs staged optimization:
    1. Loss weight optimization
    2. Learning rate optimization  
    3. Batch size optimization
    4. Vocabulary size optimization
    5. Final fine-tuning
    """
    
    def __init__(self, 
                 base_config: Optional[HyperparameterConfig] = None,
                 output_dir: str = "output/hyperparameter_tuning",
                 validation_metric: str = "combined_score"):
        """
        Initialize hyperparameter tuner.
        
        Args:
            base_config: Base configuration to start from
            output_dir: Directory to save tuning results
            validation_metric: Primary metric for optimization
        """
        self.base_config = base_config or HyperparameterConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_metric = validation_metric
        
        self.search_space = HyperparameterSearchSpace()
        self.tuning_history = []
        self.best_config = None
        self.best_score = -float('inf')
        
        logger.info(f"HyperparameterTuner initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Validation metric: {self.validation_metric}")
        
    def run_staged_optimization(self, 
                              max_trials_per_stage: int = 5,
                              quick_evaluation: bool = True) -> HyperparameterConfig:
        """
        Run staged hyperparameter optimization.
        
        Args:
            max_trials_per_stage: Maximum trials per optimization stage
            quick_evaluation: Use quick evaluation (fewer epochs) for tuning
            
        Returns:
            Best hyperparameter configuration found
        """
        logger.info("🚀 Starting staged hyperparameter optimization")
        
        stages = [
            ("loss_weights", "Loss Weight Optimization"),
            ("learning_rates", "Learning Rate Optimization"), 
            ("batch_configs", "Batch Size Optimization"),
            ("vocab_sizes", "Vocabulary Size Optimization"),
            ("dropout_rates", "Dropout Rate Optimization")
        ]
        
        current_config = self.base_config
        
        for stage_name, stage_desc in stages:
            logger.info(f"\n{'='*60}")
            logger.info(f"Stage: {stage_desc}")
            logger.info(f"{'='*60}")
            
            best_stage_config = self._optimize_stage(
                stage_name, 
                current_config, 
                max_trials_per_stage,
                quick_evaluation
            )
            
            if best_stage_config is not None:
                current_config = best_stage_config
                logger.info(f"✅ Stage completed. Best score: {self.best_score:.4f}")
            else:
                logger.warning(f"⚠️ Stage {stage_name} found no improvements")
        
        # Final validation with best configuration
        logger.info(f"\n{'='*60}")
        logger.info("Final Validation")
        logger.info(f"{'='*60}")
        
        final_score = self._evaluate_config(current_config, quick_evaluation=False)
        
        self.best_config = current_config
        self.best_score = final_score
        
        # Save results
        self._save_tuning_results()
        
        logger.info(f"🎉 Hyperparameter optimization completed!")
        logger.info(f"Best configuration found with score: {self.best_score:.4f}")
        
        return self.best_config
    
    def _optimize_stage(self, 
                       stage_name: str, 
                       base_config: HyperparameterConfig,
                       max_trials: int,
                       quick_evaluation: bool) -> Optional[HyperparameterConfig]:
        """Optimize a specific stage of hyperparameters."""
        
        configs_to_try = self.search_space.get_grid_search_configs(stage_name)
        
        if len(configs_to_try) > max_trials:
            # Randomly sample if too many configurations
            import random
            configs_to_try = random.sample(configs_to_try, max_trials)
        
        stage_best_config = None
        stage_best_score = -float('inf')
        
        for i, param_update in enumerate(configs_to_try):
            logger.info(f"Trial {i+1}/{len(configs_to_try)}: {param_update}")
            
            # Create new config with updated parameters
            trial_config = self._update_config(base_config, stage_name, param_update)
            
            # Evaluate configuration
            try:
                score = self._evaluate_config(trial_config, quick_evaluation)
                
                self.tuning_history.append({
                    'stage': stage_name,
                    'trial': i + 1,
                    'config': asdict(trial_config),
                    'score': score,
                    'param_update': param_update,
                    'timestamp': time.time()
                })
                
                logger.info(f"  Score: {score:.4f}")
                
                if score > stage_best_score:
                    stage_best_score = score
                    stage_best_config = trial_config
                    logger.info(f"  🌟 New best for stage: {score:.4f}")
                
                if score > self.best_score:
                    self.best_score = score
                    logger.info(f"  🏆 New global best: {score:.4f}")
                    
            except Exception as e:
                logger.error(f"  ❌ Trial failed: {e}")
                continue
        
        return stage_best_config
    
    def _update_config(self, 
                      base_config: HyperparameterConfig, 
                      stage_name: str, 
                      param_update: Dict) -> HyperparameterConfig:
        """Update configuration with stage-specific parameters."""
        
        # Create a copy of the base config
        new_config = HyperparameterConfig(**asdict(base_config))
        
        if stage_name == "loss_weights":
            new_config.disease_weight = param_update["disease"]
            new_config.global_keyword_weight = param_update["global_kw"]
            new_config.region_keyword_weight = param_update["region_kw"]
            new_config.radbert_semantic_weight = param_update["radbert"]
            new_config.medclip_contrastive_weight = param_update["medclip"]
            
        elif stage_name == "learning_rates":
            new_config.base_learning_rate = param_update["base"]
            new_config.backbone_lr_multiplier = param_update["backbone_mult"]
            new_config.keyword_head_lr_multiplier = param_update["keyword_mult"]
            new_config.medclip_lr_multiplier = param_update["medclip_mult"]
            
        elif stage_name == "batch_configs":
            new_config.batch_size = param_update["batch_size"]
            new_config.gradient_accumulation_steps = param_update["grad_accum"]
            
        elif stage_name == "vocab_sizes":
            new_config.vocab_size = param_update["vocab_size"]
            
        elif stage_name == "dropout_rates":
            new_config.dropout_rate = param_update["dropout_rate"]
        
        return new_config
    
    def _evaluate_config(self, 
                        config: HyperparameterConfig, 
                        quick_evaluation: bool = True) -> float:
        """
        Evaluate a hyperparameter configuration.
        
        Args:
            config: Configuration to evaluate
            quick_evaluation: Use fewer epochs for faster evaluation
            
        Returns:
            Validation score
        """
        # This is a placeholder for the actual training/evaluation
        # In practice, this would:
        # 1. Initialize model with config
        # 2. Train for specified epochs
        # 3. Evaluate on validation set
        # 4. Return composite score
        
        logger.info(f"Evaluating config: {config}")
        
        if quick_evaluation:
            # Simulate quick evaluation
            eval_epochs = 2
            eval_samples = 100
        else:
            # Full evaluation
            eval_epochs = config.num_epochs
            eval_samples = None
        
        # Simulate evaluation metrics
        # Replace this with actual training/evaluation logic
        metrics = self._simulate_evaluation(config, eval_epochs, eval_samples)
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(metrics)
        
        return composite_score
    
    def _simulate_evaluation(self, 
                           config: HyperparameterConfig,
                           epochs: int,
                           max_samples: Optional[int]) -> Dict[str, float]:
        """Simulate model evaluation (placeholder for actual training)."""
        
        # This simulates realistic performance based on configuration
        # In practice, replace with actual model training and evaluation
        
        # Base performance
        base_disease_f1 = 0.58
        base_keyword_f1 = 0.42
        base_region_f1 = 0.35
        
        # Simulate effects of hyperparameters
        
        # Loss weight effects
        total_weight = config.get_total_loss_weight()
        balance_penalty = abs(total_weight - 2.3) * 0.02  # Penalty for imbalanced weights
        
        # Learning rate effects  
        lr_factor = 1.0
        if config.base_learning_rate < 1e-5:
            lr_factor = 0.95  # Too low
        elif config.base_learning_rate > 1e-4:
            lr_factor = 0.90  # Too high
        
        # Batch size effects
        effective_batch = config.get_effective_batch_size()
        batch_factor = 1.0
        if effective_batch < 16:
            batch_factor = 0.92  # Too small
        elif effective_batch > 64:
            batch_factor = 0.88  # Too large
        
        # Vocabulary size effects
        vocab_factor = 1.0
        if config.vocab_size < 800:
            vocab_factor = 0.95  # Too small
        elif config.vocab_size > 2500:
            vocab_factor = 0.93  # Too large
        
        # Combine effects with some randomness
        noise = np.random.normal(0, 0.02)
        
        disease_f1 = base_disease_f1 * lr_factor * batch_factor * (1 - balance_penalty) + noise
        keyword_f1 = base_keyword_f1 * vocab_factor * lr_factor * (1 - balance_penalty) + noise
        region_f1 = base_region_f1 * batch_factor * vocab_factor + noise
        
        # Ensure realistic bounds
        disease_f1 = np.clip(disease_f1, 0.3, 0.75)
        keyword_f1 = np.clip(keyword_f1, 0.2, 0.65)
        region_f1 = np.clip(region_f1, 0.15, 0.55)
        
        return {
            'disease_f1': disease_f1,
            'keyword_f1': keyword_f1,
            'region_f1': region_f1,
            'disease_precision': disease_f1 * 0.95,
            'disease_recall': disease_f1 * 1.05,
            'keyword_precision': keyword_f1 * 0.88,
            'medclip_alignment': 0.72 + noise * 0.1
        }
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score from evaluation metrics."""
        
        # Weighted combination of metrics
        weights = {
            'disease_f1': 0.4,      # Primary disease classification
            'keyword_f1': 0.25,     # Global keyword prediction
            'region_f1': 0.15,      # Region-specific keywords
            'medclip_alignment': 0.2 # Vision-language alignment
        }
        
        composite_score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                composite_score += metrics[metric] * weight
        
        return composite_score
    
    def _save_tuning_results(self):
        """Save hyperparameter tuning results."""
        
        # Save tuning history
        history_df = pd.DataFrame(self.tuning_history)
        history_path = self.output_dir / "tuning_history.csv"
        history_df.to_csv(history_path, index=False)
        
        # Save best configuration
        best_config_path = self.output_dir / "best_config.json"
        with open(best_config_path, 'w') as f:
            json.dump(asdict(self.best_config), f, indent=2)
        
        # Save summary
        summary = {
            'best_score': self.best_score,
            'total_trials': len(self.tuning_history),
            'best_config': asdict(self.best_config),
            'tuning_completed': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = self.output_dir / "tuning_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Tuning results saved to {self.output_dir}")
    
    def get_best_config(self) -> Optional[HyperparameterConfig]:
        """Get the best configuration found."""
        return self.best_config
    
    def get_tuning_summary(self) -> Dict:
        """Get summary of tuning results."""
        if not self.tuning_history:
            return {"message": "No tuning performed yet"}
        
        history_df = pd.DataFrame(self.tuning_history)
        
        return {
            'total_trials': len(self.tuning_history),
            'best_score': self.best_score,
            'scores_by_stage': history_df.groupby('stage')['score'].agg(['mean', 'max', 'std']).to_dict(),
            'improvement_over_baseline': self.best_score - self.tuning_history[0]['score'] if self.tuning_history else 0
        }


def run_hyperparameter_optimization(
    output_dir: str = "output/hyperparameter_tuning",
    max_trials_per_stage: int = 3,
    quick_evaluation: bool = True
) -> HyperparameterConfig:
    """
    Run complete hyperparameter optimization.
    
    Args:
        output_dir: Directory to save results
        max_trials_per_stage: Maximum trials per optimization stage
        quick_evaluation: Use quick evaluation for faster tuning
        
    Returns:
        Best hyperparameter configuration
    """
    logger.info("🔧 Starting hyperparameter optimization")
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        output_dir=output_dir,
        validation_metric="combined_score"
    )
    
    # Run optimization
    best_config = tuner.run_staged_optimization(
        max_trials_per_stage=max_trials_per_stage,
        quick_evaluation=quick_evaluation
    )
    
    # Print summary
    summary = tuner.get_tuning_summary()
    logger.info("\n" + "="*60)
    logger.info("HYPERPARAMETER TUNING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total trials: {summary['total_trials']}")
    logger.info(f"Best score: {tuner.best_score:.4f}")
    logger.info(f"Improvement: {summary.get('improvement_over_baseline', 0):.4f}")
    
    return best_config


def test_hyperparameter_tuning():
    """Test the hyperparameter tuning system."""
    logger.info("🧪 Testing hyperparameter tuning system")
    
    # Test with minimal configuration
    best_config = run_hyperparameter_optimization(
        output_dir="output/test_tuning",
        max_trials_per_stage=2,
        quick_evaluation=True
    )
    
    logger.info("✅ Hyperparameter tuning test completed")
    logger.info(f"Best configuration: {best_config}")
    
    return best_config


if __name__ == "__main__":
    # Run hyperparameter optimization
    best_config = run_hyperparameter_optimization()
    print(f"Best configuration found: {best_config}") 