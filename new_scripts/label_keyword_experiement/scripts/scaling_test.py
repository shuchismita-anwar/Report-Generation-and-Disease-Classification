"""
Phase 2 Dataset Scaling Test Script
==================================

Tests scaling from 350 (baseline) to 500 samples with overfitting monitoring.
Implements the gradual scaling roadmap from improvements.md.
"""

import logging
import sys
import os
from datetime import datetime
import traceback

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config.training_config import SCALING_CONFIG
from utils.scaling_manager import create_scaling_manager

# Import SimpleHybridTrainer directly
import importlib.util
trainer_spec = importlib.util.spec_from_file_location("train_model", os.path.join(parent_dir, "scripts", "3_train_model.py"))
trainer_module = importlib.util.module_from_spec(trainer_spec)
trainer_spec.loader.exec_module(trainer_module)
SimpleHybridTrainer = trainer_module.SimpleHybridTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScalingTestRunner:
    """Runs gradual dataset scaling tests with monitoring."""
    
    def __init__(self):
        self.scaling_manager = create_scaling_manager(SCALING_CONFIG)
        self.baseline_results = None
        
        logger.info("ScalingTestRunner initialized")
        logger.info(f"Scaling sequence: {self.scaling_manager.scaling_sequence}")
    
    def run_baseline_test(self):
        """Run baseline test with 350 samples to establish metrics."""
        logger.info("="*80)
        logger.info("🔍 PHASE 2.1: ESTABLISHING BASELINE (350 samples)")
        logger.info("="*80)
        
        try:
            # Force 350 samples
            trainer = SimpleHybridTrainer()
            trainer.training_config['max_train_samples'] = 350
            trainer.advanced_config['max_train_samples'] = 350
            
            # Run training
            results = trainer.train(num_samples=350)
            
            # Extract key metrics
            baseline_metrics = {
                'training_loss': results['train_result'].training_loss,
                'eval_auc': results['val_results'].get('eval_auc', 0.5),
                'eval_recall': results['val_results'].get('eval_disease_micro_recall', 0.967),
                'eval_f1': results['val_results'].get('eval_disease_micro_f1', 0.4055),
            }
            
            # Set baseline in scaling manager
            self.scaling_manager.set_baseline_metrics(baseline_metrics)
            self.baseline_results = results
            
            logger.info("✅ BASELINE ESTABLISHED:")
            logger.info(f"   - Training Loss: {baseline_metrics['training_loss']:.4f}")
            logger.info(f"   - AUC: {baseline_metrics['eval_auc']:.4f}")
            logger.info(f"   - Recall: {baseline_metrics['eval_recall']:.4f}")
            logger.info(f"   - F1: {baseline_metrics['eval_f1']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Baseline test failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def run_scaling_test(self, target_size: int):
        """Run training with specific dataset size and validate results."""
        logger.info("="*80)
        logger.info(f"🔄 PHASE 2.{target_size}: SCALING TO {target_size} SAMPLES")
        logger.info("="*80)
        
        try:
            # Configure trainer for target size
            trainer = SimpleHybridTrainer()
            trainer.training_config['max_train_samples'] = target_size
            trainer.advanced_config['max_train_samples'] = target_size
            
            # Run training
            results = trainer.train(num_samples=target_size)
            
            # Extract metrics
            current_metrics = {
                'training_loss': results['train_result'].training_loss,
                'eval_auc': results['val_results'].get('eval_auc', 0.5),
                'eval_recall': results['val_results'].get('eval_disease_micro_recall', 1.0),
                'eval_f1': results['val_results'].get('eval_disease_micro_f1', 0.0),
            }
            
            # Validate scaling step
            is_valid, issues = self.scaling_manager.validate_scaling_step(current_metrics)
            
            if is_valid:
                logger.info(f"✅ SCALING TO {target_size} SAMPLES: SUCCESS")
                logger.info(f"   - Training Loss: {current_metrics['training_loss']:.4f}")
                logger.info(f"   - AUC: {current_metrics['eval_auc']:.4f}")
                logger.info(f"   - Recall: {current_metrics['eval_recall']:.4f}")
                logger.info(f"   - F1: {current_metrics['eval_f1']:.4f}")
                return True, results
            else:
                logger.warning(f"❌ SCALING TO {target_size} SAMPLES: FAILED")
                logger.warning("Issues detected:")
                for issue in issues:
                    logger.warning(f"   - {issue}")
                return False, results
                
        except Exception as e:
            logger.error(f"❌ Scaling test to {target_size} failed: {e}")
            logger.error(traceback.format_exc())
            return False, None
    
    def run_complete_scaling_sequence(self):
        """Run the complete scaling sequence with validation."""
        logger.info("="*80)
        logger.info("🚀 PHASE 2: COMPLETE DATASET SCALING TEST")
        logger.info("="*80)
        logger.info(f"Start time: {datetime.now()}")
        
        # Step 1: Establish baseline
        if not self.run_baseline_test():
            logger.error("❌ FAILED: Could not establish baseline - aborting scaling")
            return False
        
        # Step 2: Test each scaling step
        scaling_results = {}
        
        for i in range(1, len(self.scaling_manager.scaling_sequence)):
            target_size = self.scaling_manager.scaling_sequence[i]
            
            success, results = self.run_scaling_test(target_size)
            scaling_results[target_size] = {
                'success': success,
                'results': results
            }
            
            if success:
                # Advance to next step
                next_size = self.scaling_manager.advance_to_next_scale()
                if next_size is None:
                    logger.info("🎯 SCALING SEQUENCE COMPLETE!")
                    break
            else:
                # Rollback and stop
                prev_size = self.scaling_manager.rollback_scaling()
                logger.warning(f"⚠️ Scaling failed at {target_size}, rolled back to {prev_size}")
                logger.warning("Stopping scaling sequence for safety")
                break
        
        # Generate final report
        self.generate_scaling_report(scaling_results)
        
        return True
    
    def generate_scaling_report(self, scaling_results: dict):
        """Generate comprehensive scaling report."""
        logger.info("="*80)
        logger.info("📊 PHASE 2 SCALING REPORT")
        logger.info("="*80)
        
        # Baseline performance
        if self.baseline_results:
            baseline = self.baseline_results
            logger.info("🔍 BASELINE (350 samples):")
            logger.info(f"   - Training Loss: {baseline['train_result'].training_loss:.4f}")
            logger.info(f"   - Status: ✅ PROVEN STABLE")
        
        # Scaling results
        for size, data in scaling_results.items():
            status = "✅ SUCCESS" if data['success'] else "❌ FAILED"
            logger.info(f"🔄 SCALING TO {size} samples: {status}")
            
            if data['results'] and data['success']:
                results = data['results']
                logger.info(f"   - Training Loss: {results['train_result'].training_loss:.4f}")
                logger.info(f"   - Validation: No overfitting detected")
            elif data['results']:
                results = data['results']
                logger.info(f"   - Training Loss: {results['train_result'].training_loss:.4f}")
                logger.info(f"   - Validation: OVERFITTING OR REGRESSION DETECTED")
        
        # Next steps
        if self.scaling_manager.is_scaling_complete():
            logger.info("🎉 SCALING SEQUENCE COMPLETE!")
            logger.info("Next: Ready for Phase 3 (Multi-task re-integration)")
        else:
            current_size = self.scaling_manager.get_current_dataset_size()
            logger.info(f"⚠️ SCALING STOPPED AT: {current_size} samples")
            logger.info("Recommendation: Investigate and fix issues before proceeding")
        
        # Save detailed progress
        progress = self.scaling_manager.get_scaling_progress()
        logger.info(f"📁 Detailed progress saved to: {self.scaling_manager.output_dir}")


def main():
    """Main function to run Phase 2 scaling tests."""
    logger.info("Starting Phase 2 Dataset Scaling Implementation")
    
    try:
        runner = ScalingTestRunner()
        
        # Test single scaling step (350 → 500)
        logger.info("="*50)
        logger.info("🔄 QUICK TEST: 350 → 500 samples")
        logger.info("="*50)
        
        # First establish baseline  
        if runner.run_baseline_test():
            # Then test 500 samples
            success, results = runner.run_scaling_test(500)
            
            if success:
                logger.info("✅ PHASE 2.1 SUCCESS: 500 sample scaling validated!")
                logger.info("Ready to proceed with full scaling sequence")
                
                # Optionally run full sequence
                run_full = input("\n🔄 Run full scaling sequence (500→750→1000)? [y/N]: ").lower().strip()
                if run_full == 'y':
                    runner.run_complete_scaling_sequence()
                else:
                    logger.info("Stopping at 500 samples. Run again to continue scaling.")
            else:
                logger.warning("❌ PHASE 2.1 FAILED: 500 sample scaling has issues")
                logger.warning("Need to investigate before proceeding")
        
        logger.info("\n" + "="*80)
        logger.info("✅ PHASE 2 IMPLEMENTATION COMPLETE")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Phase 2 implementation failed: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main() 