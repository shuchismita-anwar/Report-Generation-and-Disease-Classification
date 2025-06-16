"""
Phase 2.3: Hard Negative Parameter Tuning
==========================================

Systematically tests different pos_weight and negative_multiplier combinations
to optimize discrimination while maintaining training stability at 500 sample limit.
"""

import logging
import sys
import os
from datetime import datetime
import traceback
import itertools
from typing import List, Tuple, Dict
import torch

# Add current directory to path for imports  
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config.training_config import SCALING_CONFIG
from models.loss_functions import HardNegativeLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardNegativeParameterTuner:
    """
    Tests different hard negative mining parameters to optimize performance.
    
    Based on Phase 2.3 plan from improvements.md:
    - pos_weight range: [0.3, 0.4, 0.5, 0.6, 0.7]  
    - negative_multiplier range: [1.5, 2.0, 2.5, 3.0]
    """
    
    def __init__(self):
        self.pos_weight_range = [0.3, 0.4, 0.5, 0.6, 0.7]
        self.negative_multiplier_range = [1.5, 2.0, 2.5, 3.0]
        
        self.results = []
        self.baseline_metrics = None
        
        logger.info("HardNegativeParameterTuner initialized")
        logger.info(f"pos_weight range: {self.pos_weight_range}")
        logger.info(f"negative_multiplier range: {self.negative_multiplier_range}")
    
    def test_parameter_combination(self, pos_weight: float, negative_multiplier: float) -> Dict:
        """Test a specific parameter combination."""
        logger.info(f"Testing pos_weight={pos_weight}, negative_multiplier={negative_multiplier}")
        
        try:
            # Import trainer here to avoid circular imports
            import importlib.util
            trainer_spec = importlib.util.spec_from_file_location("train_model", os.path.join(parent_dir, "scripts", "3_train_model.py"))
            trainer_module = importlib.util.module_from_spec(trainer_spec)
            trainer_spec.loader.exec_module(trainer_module)
            SimpleHybridTrainer = trainer_module.SimpleHybridTrainer
            
            # Create trainer with modified loss parameters
            trainer = SimpleHybridTrainer()
            
            # Override training config to disable multiprocessing (Windows fix)
            trainer.training_config['dataloader_num_workers'] = 0
            trainer.training_config['dataloader_persistent_workers'] = False
            
            # Setup model first
            trainer.setup_model()
            
            # Modify the hard negative loss parameters in the model
            if hasattr(trainer.model, 'disease_loss_fn') and hasattr(trainer.model.disease_loss_fn, 'pos_weight'):
                # Update existing hard negative loss function
                num_diseases = trainer.model_config.get('num_diseases', 8)
                trainer.model.disease_loss_fn.pos_weight = torch.tensor([pos_weight] * num_diseases)
                trainer.model.disease_loss_fn.negative_weight_multiplier = negative_multiplier
                logger.info(f"Updated loss params: pos_weight={pos_weight}, neg_mult={negative_multiplier}")
            else:
                # Create new hard negative loss function with specified parameters
                from models.loss_functions import HardNegativeLoss
                num_diseases = trainer.model_config.get('num_diseases', 8)
                trainer.model.disease_loss_fn = HardNegativeLoss(
                    pos_weight=[pos_weight] * num_diseases,
                    negative_weight_multiplier=negative_multiplier
                )
            
            # Run training
            results = trainer.train(num_samples=500)  # Use safe limit
            
            # Extract metrics
            metrics = {
                'pos_weight': pos_weight,
                'negative_multiplier': negative_multiplier,
                'training_loss': results['train_result'].training_loss,
                'eval_auc': results['val_results'].get('eval_auc', 0.5),
                'eval_recall': results['val_results'].get('eval_disease_micro_recall', 1.0),
                'eval_f1': results['val_results'].get('eval_disease_micro_f1', 0.0),
                'eval_precision': results['val_results'].get('eval_disease_micro_precision', 0.0),
                'optimized_recall': results['val_results'].get('optimized_metrics', {}).get('micro_recall', 0.0),
                'training_stable': results['train_result'].training_loss > 1.0,  # Should not memorize
                'discrimination_good': results['val_results'].get('eval_auc', 0.0) > 0.5,
                'not_always_positive': results['val_results'].get('eval_disease_micro_recall', 1.0) < 0.99
            }
            
            # Calculate composite score
            metrics['composite_score'] = self.calculate_composite_score(metrics)
            
            logger.info(f"Results: Loss={metrics['training_loss']:.3f}, AUC={metrics['eval_auc']:.3f}, "
                       f"Recall={metrics['eval_recall']:.3f}, Score={metrics['composite_score']:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to test pos_weight={pos_weight}, negative_multiplier={negative_multiplier}: {e}")
            logger.error(traceback.format_exc())
            return {
                'pos_weight': pos_weight,
                'negative_multiplier': negative_multiplier,
                'training_loss': 999.0,
                'eval_auc': 0.0,
                'eval_recall': 1.0,
                'eval_f1': 0.0,
                'eval_precision': 0.0,
                'optimized_recall': 0.0,
                'training_stable': False,
                'discrimination_good': False,
                'not_always_positive': False,
                'composite_score': 0.0,
                'error': str(e)
            }
    
    def calculate_composite_score(self, metrics: Dict) -> float:
        """
        Calculate composite score for parameter combination.
        
        Factors:
        - AUC (discrimination): 40%
        - Training stability: 30%  
        - Balanced recall: 20%
        - F1 performance: 10%
        """
        score = 0.0
        
        # AUC score (0-1, higher better)
        auc_score = metrics.get('eval_auc', 0.5)
        score += auc_score * 0.4
        
        # Training stability (loss should be reasonable, not too low)
        training_loss = metrics.get('training_loss', 999.0)
        if training_loss > 2.0:
            stability_score = 0.5  # Too high
        elif training_loss < 0.8:
            stability_score = 0.2  # Too low (memorizing)
        else:
            stability_score = 1.0  # Good range
        score += stability_score * 0.3
        
        # Balanced recall (not always positive)
        recall = metrics.get('eval_recall', 1.0)
        if recall > 0.99:
            balance_score = 0.0  # Always positive
        elif recall < 0.7:
            balance_score = 0.5  # Too conservative
        else:
            balance_score = 1.0  # Good balance
        score += balance_score * 0.2
        
        # F1 performance
        f1 = metrics.get('eval_f1', 0.0)
        f1_score = min(f1 / 0.6, 1.0)  # Normalize to target F1 of 0.6
        score += f1_score * 0.1
        
        return score
    
    def run_parameter_grid_search(self, quick_test: bool = False) -> List[Dict]:
        """
        Run grid search over all parameter combinations.
        
        Args:
            quick_test: If True, test only a subset of combinations
        """
        logger.info("="*80)
        logger.info("🔧 PHASE 2.3: HARD NEGATIVE PARAMETER TUNING")
        logger.info("="*80)
        logger.info(f"Start time: {datetime.now()}")
        
        # Generate parameter combinations
        if quick_test:
            # Test only key combinations for speed
            combinations = [
                (0.3, 2.0), (0.5, 2.0), (0.7, 2.0),  # Vary pos_weight
                (0.5, 1.5), (0.5, 2.5), (0.5, 3.0),  # Vary negative_multiplier  
            ]
            logger.info("🚀 Quick test mode: 6 combinations")
        else:
            # Full grid search
            combinations = list(itertools.product(self.pos_weight_range, self.negative_multiplier_range))
            logger.info(f"🔬 Full grid search: {len(combinations)} combinations")
        
        # Test each combination
        for i, (pos_weight, neg_mult) in enumerate(combinations):
            logger.info(f"\n--- Combination {i+1}/{len(combinations)} ---")
            
            result = self.test_parameter_combination(pos_weight, neg_mult)
            self.results.append(result)
            
            # Log intermediate results
            if result.get('composite_score', 0) > 0:
                logger.info(f"✅ Success: Score={result['composite_score']:.3f}")
            else:
                logger.warning(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Sort results by composite score
        self.results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Generate report
        self.generate_tuning_report()
        
        return self.results
    
    def generate_tuning_report(self):
        """Generate comprehensive tuning report."""
        logger.info("\n" + "="*80)
        logger.info("📊 HARD NEGATIVE PARAMETER TUNING REPORT")
        logger.info("="*80)
        
        if not self.results:
            logger.warning("No results to report")
            return
        
        # Best parameters
        best = self.results[0]
        logger.info(f"🏆 BEST PARAMETERS:")
        logger.info(f"   - pos_weight: {best['pos_weight']}")
        logger.info(f"   - negative_multiplier: {best['negative_multiplier']}")
        logger.info(f"   - Composite score: {best['composite_score']:.3f}")
        logger.info(f"   - Training loss: {best['training_loss']:.3f}")
        logger.info(f"   - AUC: {best['eval_auc']:.3f}")
        logger.info(f"   - Recall: {best['eval_recall']:.3f}")
        logger.info(f"   - F1: {best['eval_f1']:.3f}")
        
        # Top 3 results
        logger.info(f"\n🥇 TOP 3 PARAMETER COMBINATIONS:")
        for i, result in enumerate(self.results[:3]):
            logger.info(f"   {i+1}. pos_weight={result['pos_weight']}, neg_mult={result['negative_multiplier']}, "
                       f"score={result['composite_score']:.3f}")
        
        # Current baseline comparison
        baseline_pos_weight = 0.5
        baseline_neg_mult = 2.0
        baseline_result = next((r for r in self.results 
                              if r['pos_weight'] == baseline_pos_weight and r['negative_multiplier'] == baseline_neg_mult), None)
        
        if baseline_result:
            improvement = best['composite_score'] - baseline_result['composite_score']
            logger.info(f"\n📈 IMPROVEMENT vs BASELINE:")
            logger.info(f"   - Baseline (0.5, 2.0): {baseline_result['composite_score']:.3f}")
            logger.info(f"   - Best: {best['composite_score']:.3f}")
            logger.info(f"   - Improvement: +{improvement:.3f} ({improvement/baseline_result['composite_score']*100:.1f}%)")
        
        # Parameter analysis
        logger.info(f"\n🔍 PARAMETER ANALYSIS:")
        
        # pos_weight analysis
        pos_weight_scores = {}
        for result in self.results:
            pw = result['pos_weight']
            if pw not in pos_weight_scores:
                pos_weight_scores[pw] = []
            pos_weight_scores[pw].append(result['composite_score'])
        
        logger.info("   pos_weight averages:")
        for pw in sorted(pos_weight_scores.keys()):
            avg_score = sum(pos_weight_scores[pw]) / len(pos_weight_scores[pw])
            logger.info(f"     {pw}: {avg_score:.3f}")
        
        # negative_multiplier analysis
        neg_mult_scores = {}
        for result in self.results:
            nm = result['negative_multiplier']
            if nm not in neg_mult_scores:
                neg_mult_scores[nm] = []
            neg_mult_scores[nm].append(result['composite_score'])
        
        logger.info("   negative_multiplier averages:")
        for nm in sorted(neg_mult_scores.keys()):
            avg_score = sum(neg_mult_scores[nm]) / len(neg_mult_scores[nm])
            logger.info(f"     {nm}: {avg_score:.3f}")
        
        # Recommendations
        logger.info(f"\n💡 RECOMMENDATIONS:")
        if best['composite_score'] > 0.7:
            logger.info("   ✅ Excellent parameters found - ready for deployment")
        elif best['composite_score'] > 0.5:
            logger.info("   ⚠️ Good parameters found - consider further architecture optimization")
        else:
            logger.info("   ❌ No satisfactory parameters - architecture changes needed")
        
        best_pos_weight = best['pos_weight']
        best_neg_mult = best['negative_multiplier']
        logger.info(f"   🎯 Use: pos_weight={best_pos_weight}, negative_multiplier={best_neg_mult}")


def main():
    """Main function to run hard negative parameter tuning."""
    logger.info("Starting Phase 2.3: Hard Negative Parameter Tuning")
    
    try:
        tuner = HardNegativeParameterTuner()
        
        # Ask user for test mode
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == '--quick':
            quick_mode = True
        else:
            print("Choose test mode:")
            print("1. Quick test (6 combinations, ~10 minutes)")
            print("2. Full grid search (20 combinations, ~45 minutes)")
            choice = input("Enter choice [1-2]: ").strip()
            quick_mode = choice == '1'
        
        # Run tuning
        results = tuner.run_parameter_grid_search(quick_test=quick_mode)
        
        if results:
            logger.info("\n" + "="*80)
            logger.info("✅ PHASE 2.3 PARAMETER TUNING COMPLETE")
            logger.info("="*80)
            logger.info(f"Best parameters: pos_weight={results[0]['pos_weight']}, "
                       f"negative_multiplier={results[0]['negative_multiplier']}")
            logger.info(f"Composite score: {results[0]['composite_score']:.3f}")
        else:
            logger.error("❌ No valid results obtained")
        
    except Exception as e:
        logger.error(f"❌ Phase 2.3 tuning failed: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main() 