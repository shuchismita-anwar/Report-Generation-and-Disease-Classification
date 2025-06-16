"""
Dataset Scaling Manager for Phase 2 Implementation
=================================================

This module implements the gradual dataset scaling strategy from improvements.md Phase 2.
It provides safe scaling from 350 → 500 → 750 → 1000 samples with overfitting monitoring.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)

@dataclass 
class ScalingMetrics:
    """Metrics for monitoring scaling progress."""
    dataset_size: int
    training_loss: float
    eval_auc: float
    eval_recall: float
    eval_f1: float
    overfitting_detected: bool
    regression_detected: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dataset_size': self.dataset_size,
            'training_loss': self.training_loss,
            'eval_auc': self.eval_auc,
            'eval_recall': self.eval_recall,
            'eval_f1': self.eval_f1,
            'overfitting_detected': self.overfitting_detected,
            'regression_detected': self.regression_detected
        }

class DatasetScalingManager:
    """
    Manages gradual dataset scaling with overfitting detection.
    
    Implements the Phase 2 scaling roadmap:
    - 350 (baseline) → 500 → 750 → 1000 → full dataset
    - Validates at each step for overfitting regression
    - Supports rollback if scaling causes problems
    """
    
    def __init__(self, scaling_config: Dict, output_dir: str = "output/scaling_logs"):
        self.config = scaling_config
        self.output_dir = output_dir
        
        # Scaling progression
        self.scaling_sequence = [
            scaling_config['phase_1_baseline'],     # 350 - proven baseline
            scaling_config['phase_2_scale_500'],    # 500 - original max
            scaling_config['phase_2_scale_750'],    # 750 - gradual increase  
            scaling_config['phase_2_scale_1000'],   # 1000 - double baseline
        ]
        
        self.current_scale_index = 0
        self.baseline_metrics = None
        self.scaling_history = []
        
        # Validation criteria
        self.criteria = scaling_config['validation_criteria']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"DatasetScalingManager initialized:")
        logger.info(f"  - Scaling sequence: {self.scaling_sequence}")
        logger.info(f"  - Validation criteria: {self.criteria}")
        logger.info(f"  - Output directory: {output_dir}")
    
    def get_current_dataset_size(self) -> int:
        """Get the current dataset size for this scaling step."""
        return self.scaling_sequence[self.current_scale_index]
    
    def set_baseline_metrics(self, metrics: Dict[str, float]):
        """Set baseline metrics from the proven 350-sample configuration."""
        self.baseline_metrics = ScalingMetrics(
            dataset_size=350,
            training_loss=metrics.get('training_loss', 1.54),
            eval_auc=metrics.get('eval_auc', 0.5),
            eval_recall=metrics.get('eval_recall', 0.967),
            eval_f1=metrics.get('eval_f1', 0.4055),
            overfitting_detected=False,
            regression_detected=False
        )
        
        logger.info(f"✅ Baseline metrics set: {self.baseline_metrics}")
        self.save_scaling_progress()
    
    def validate_scaling_step(self, current_metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate if current scaling step is successful.
        
        Returns:
            (is_valid, reasons) - True if scaling is safe to continue
        """
        if self.baseline_metrics is None:
            logger.warning("No baseline metrics set - cannot validate scaling")
            return True, []
        
        current = ScalingMetrics(
            dataset_size=self.get_current_dataset_size(),
            training_loss=current_metrics.get('training_loss', 999.0),
            eval_auc=current_metrics.get('eval_auc', 0.0),
            eval_recall=current_metrics.get('eval_recall', 1.0),
            eval_f1=current_metrics.get('eval_f1', 0.0),
            overfitting_detected=False,
            regression_detected=False
        )
        
        issues = []
        
        # Check 1: Training loss shouldn't increase dramatically
        if current.training_loss > (self.baseline_metrics.training_loss + self.criteria['max_training_loss_increase']):
            issues.append(f"Training loss increased too much: {current.training_loss:.3f} vs baseline {self.baseline_metrics.training_loss:.3f}")
            current.overfitting_detected = True
        
        # Check 2: AUC should stay above random
        if current.eval_auc < self.criteria['min_auc_threshold']:
            issues.append(f"AUC below threshold: {current.eval_auc:.3f} < {self.criteria['min_auc_threshold']}")
            current.regression_detected = True
        
        # Check 3: Recall shouldn't go back to "always positive"
        if current.eval_recall > self.criteria['max_recall_threshold']:
            issues.append(f"Recall too high (always positive): {current.eval_recall:.3f} > {self.criteria['max_recall_threshold']}")
            current.regression_detected = True
        
        # Check 4: Training loss approaching zero (memorization)
        if current.training_loss < 0.1:
            issues.append(f"Training loss too low (memorization): {current.training_loss:.3f}")
            current.overfitting_detected = True
        
        # Store current metrics
        self.scaling_history.append(current)
        self.save_scaling_progress()
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info(f"✅ Scaling validation PASSED for size {current.dataset_size}")
            logger.info(f"   - Training loss: {current.training_loss:.3f} (baseline: {self.baseline_metrics.training_loss:.3f})")
            logger.info(f"   - AUC: {current.eval_auc:.3f} (threshold: {self.criteria['min_auc_threshold']})")
            logger.info(f"   - Recall: {current.eval_recall:.3f} (threshold: {self.criteria['max_recall_threshold']})")
        else:
            logger.warning(f"❌ Scaling validation FAILED for size {current.dataset_size}")
            for issue in issues:
                logger.warning(f"   - {issue}")
        
        return is_valid, issues
    
    def advance_to_next_scale(self) -> Optional[int]:
        """
        Advance to the next scaling step if available.
        
        Returns:
            Next dataset size, or None if at end of sequence
        """
        if self.current_scale_index < len(self.scaling_sequence) - 1:
            self.current_scale_index += 1
            next_size = self.get_current_dataset_size()
            logger.info(f"🔄 Advanced to next scaling step: {next_size} samples")
            return next_size
        else:
            logger.info("🎯 Reached end of scaling sequence")
            return None
    
    def rollback_scaling(self) -> int:
        """
        Rollback to previous scaling step if current step failed.
        
        Returns:
            Previous dataset size
        """
        if self.current_scale_index > 0:
            self.current_scale_index -= 1
            prev_size = self.get_current_dataset_size()
            logger.warning(f"⚠️ Rolled back to previous scaling step: {prev_size} samples")
            return prev_size
        else:
            logger.warning("⚠️ Cannot rollback - already at baseline scale")
            return self.get_current_dataset_size()
    
    def get_scaling_progress(self) -> Dict[str, Any]:
        """Get current scaling progress and history."""
        return {
            'current_size': self.get_current_dataset_size(),
            'current_index': self.current_scale_index,
            'scaling_sequence': self.scaling_sequence,
            'baseline_metrics': self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            'scaling_history': [m.to_dict() for m in self.scaling_history],
            'next_steps': self.scaling_sequence[self.current_scale_index + 1:] if self.current_scale_index < len(self.scaling_sequence) - 1 else []
        }
    
    def save_scaling_progress(self):
        """Save scaling progress to disk."""
        progress = self.get_scaling_progress()
        
        progress_file = os.path.join(self.output_dir, "scaling_progress.json")
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        logger.debug(f"Scaling progress saved to {progress_file}")
    
    def is_scaling_complete(self) -> bool:
        """Check if we've completed the full scaling sequence."""
        return self.current_scale_index >= len(self.scaling_sequence) - 1


def create_scaling_manager(scaling_config: Dict, output_dir: str = "output/scaling_logs"):
    """Factory function to create a dataset scaling manager."""
    manager = DatasetScalingManager(scaling_config, output_dir)
    return manager


class HardNegativeTuner:
    """
    Implements Phase 2 hard negative mining parameter tuning.
    
    Tests different pos_weight and negative_multiplier combinations
    to optimize discrimination while maintaining training stability.
    """
    
    def __init__(self, tuning_config: Dict):
        self.config = tuning_config
        self.pos_weight_range = tuning_config['pos_weight_range']
        self.negative_multiplier_range = tuning_config['negative_multiplier_range']
        self.test_combinations = tuning_config['test_combinations']
        
        self.tuning_results = []
        
        logger.info(f"HardNegativeTuner initialized:")
        logger.info(f"  - pos_weight range: {self.pos_weight_range}")
        logger.info(f"  - negative_multiplier range: {self.negative_multiplier_range}")
        logger.info(f"  - test combinations: {self.test_combinations}")
    
    def get_next_parameters(self) -> Optional[Tuple[float, float]]:
        """
        Get next parameter combination to test.
        
        Returns:
            (pos_weight, negative_multiplier) or None if testing complete
        """
        if not self.test_combinations:
            # Just return the current default
            return (0.5, 2.0)
        
        # Generate all combinations
        all_combinations = [
            (pw, nm) for pw in self.pos_weight_range 
            for nm in self.negative_multiplier_range
        ]
        
        # Filter out already tested combinations
        tested_combinations = {(r['pos_weight'], r['negative_multiplier']) for r in self.tuning_results}
        remaining = [combo for combo in all_combinations if combo not in tested_combinations]
        
        if remaining:
            return remaining[0]
        else:
            logger.info("🎯 Hard negative parameter tuning complete")
            return None
    
    def record_result(self, pos_weight: float, negative_multiplier: float, metrics: Dict[str, float]):
        """Record the result of testing a parameter combination."""
        result = {
            'pos_weight': pos_weight,
            'negative_multiplier': negative_multiplier,
            'training_loss': metrics.get('training_loss', 999.0),
            'eval_auc': metrics.get('eval_auc', 0.0),
            'eval_f1': metrics.get('eval_f1', 0.0),
            'eval_recall': metrics.get('eval_recall', 1.0),
            'training_stable': metrics.get('training_loss', 999.0) < 5.0,  # Reasonable training loss
            'discrimination_good': metrics.get('eval_auc', 0.0) > 0.5,     # Better than random
            'not_always_positive': metrics.get('eval_recall', 1.0) < 0.99  # Not always saying yes
        }
        
        self.tuning_results.append(result)
        
        logger.info(f"📝 Recorded hard negative tuning result:")
        logger.info(f"   - pos_weight: {pos_weight}, negative_multiplier: {negative_multiplier}")
        logger.info(f"   - Training loss: {result['training_loss']:.3f}")
        logger.info(f"   - AUC: {result['eval_auc']:.3f}, F1: {result['eval_f1']:.3f}, Recall: {result['eval_recall']:.3f}")
    
    def get_best_parameters(self) -> Tuple[float, float]:
        """
        Get the best parameter combination based on results.
        
        Returns:
            (best_pos_weight, best_negative_multiplier)
        """
        if not self.tuning_results:
            logger.warning("No tuning results available - returning defaults")
            return (0.5, 2.0)
        
        # Score based on: AUC (discrimination) + training stability + not always positive
        def score_combination(result):
            score = 0.0
            
            # AUC is most important (0-1, higher better)
            score += result['eval_auc'] * 0.5
            
            # Training stability (0-1, lower loss better, cap at 5.0)
            stable_score = max(0, 1 - result['training_loss'] / 5.0)
            score += stable_score * 0.3
            
            # Not always positive (0-1, lower recall better for balance)
            balance_score = max(0, 1 - (result['eval_recall'] - 0.7) / 0.3) if result['eval_recall'] > 0.7 else 1.0
            score += balance_score * 0.2
            
            return score
        
        # Find best combination
        best_result = max(self.tuning_results, key=score_combination)
        
        logger.info(f"🏆 Best hard negative parameters found:")
        logger.info(f"   - pos_weight: {best_result['pos_weight']}")
        logger.info(f"   - negative_multiplier: {best_result['negative_multiplier']}")
        logger.info(f"   - Score: {score_combination(best_result):.3f}")
        logger.info(f"   - Metrics: AUC={best_result['eval_auc']:.3f}, F1={best_result['eval_f1']:.3f}, Loss={best_result['training_loss']:.3f}")
        
        return (best_result['pos_weight'], best_result['negative_multiplier'])


def create_hard_negative_tuner(tuning_config: Dict) -> HardNegativeTuner:
    """Factory function to create a hard negative parameter tuner."""
    return HardNegativeTuner(tuning_config) 