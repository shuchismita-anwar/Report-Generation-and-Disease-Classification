"""
Training Script for Hybrid RadBERT + MedCLIP Label & Keyword Prediction
======================================================================

This script implements a simple training loop for multi-task learning that combines:
- Disease classification (existing functionality)
- Global keyword prediction
- Region-specific keyword prediction
- Multi-task loss optimization

Key Features:
- Hybrid model integration
- Enhanced data collation with keyword labels
- Comprehensive metrics computation
- Memory-optimized training
- Checkpoint management
- Error recovery and logging
"""

import os
import sys
import json
import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import (
    TrainingArguments, Trainer, EarlyStoppingCallback, 
    EvalPrediction, set_seed
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Import our components
from config.training_config import *
from config.model_config import MODEL_CONFIG
from config.data_config import DATA_CONFIG
from training.data_collator import HybridDataCollator, create_data_collator
from training.metrics import HybridMetrics, MetricsConfig, create_hybrid_metrics
from models.hybrid_model import HybridMedicalModel
from utils.memory_utils import *

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
set_seed(TRAINING_CONFIG["seed"])

class SimpleHybridTrainer:
    """
    Simple trainer for hybrid multi-task learning.
    
    Integrates disease classification + keyword prediction with:
    - HybridMedicalModel
    - HybridDataCollator  
    - HybridMetrics
    - Multi-task loss optimization
    """
    
    def __init__(
        self,
        model_config: Dict = None,
        training_config: Dict = None,
        data_config: Dict = None
    ):
        self.model_config = model_config or MODEL_CONFIG
        self.training_config = training_config or TRAINING_CONFIG
        self.data_config = data_config or DATA_CONFIG
        
        # Initialize components
        self.model = None
        self.data_collator = None
        self.metrics_computer = None
        self.trainer = None
        
        # Training state
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        logger.info("SimpleHybridTrainer initialized")
        logger.info(f"Model config: {self.model_config}")
        logger.info(f"Training config: {self.training_config}")
    
    def setup_model(self) -> HybridMedicalModel:
        """Setup the hybrid model with keyword prediction capabilities."""
        logger.info("Setting up hybrid model...")
        
        try:
            # Create hybrid model
            self.model = HybridMedicalModel(
                num_diseases=self.model_config.get('num_diseases', 8),
                vocab_size=self.model_config.get('vocab_size', 35),
                num_regions=self.model_config.get('num_regions', 17),
                hidden_dim=self.model_config.get('hidden_dim', 768),
                enable_keywords=True,  # Enable keyword prediction
                low_memory=self.training_config.get('low_memory', False)
            )
            
            # Move to device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"✅ Hybrid model created successfully")
            logger.info(f"   - Total parameters: {total_params:,}")
            logger.info(f"   - Trainable parameters: {trainable_params:,}")
            logger.info(f"   - Keywords enabled: {self.model.enable_keywords}")
            logger.info(f"   - Device: {device}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Failed to setup model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def setup_data_collator(self) -> HybridDataCollator:
        """Setup the enhanced data collator for multi-task training."""
        logger.info("Setting up hybrid data collator...")
        
        try:
            self.data_collator = create_data_collator(
                vocab_size=self.model_config.get('vocab_size', 35),
                num_regions=self.model_config.get('num_regions', 17),
                max_fixation_length=self.data_config.get('max_fixation_length', None)
            )
            
            logger.info(f"✅ Hybrid data collator created successfully")
            logger.info(f"   - Vocabulary size: {self.data_collator.vocab_size}")
            logger.info(f"   - Number of regions: {self.data_collator.num_regions}")
            
            return self.data_collator
            
        except Exception as e:
            logger.error(f"❌ Failed to setup data collator: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def setup_metrics(self) -> HybridMetrics:
        """Setup the hybrid metrics computer."""
        logger.info("Setting up hybrid metrics computer...")
        
        try:
            # Configure metrics
            metrics_config = MetricsConfig(
                disease_threshold_range=THRESHOLD_CONFIG.get('threshold_search_range', (0.38, 0.78)),
                keyword_threshold=0.5,
                precision_at_k_values=[5, 10, 20],
                num_regions=self.model_config.get('num_regions', 17)
            )
            
            # Create metrics computer
            disease_conditions = [
                'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Pneumonia'
            ]
            keyword_vocabulary = [f'keyword_{i}' for i in range(self.model_config.get('vocab_size', 35))]
            
            self.metrics_computer = create_hybrid_metrics(
                disease_conditions=disease_conditions,
                keyword_vocabulary=keyword_vocabulary,
                config=metrics_config
            )
            
            logger.info(f"✅ Hybrid metrics computer created successfully")
            logger.info(f"   - Disease conditions: {len(disease_conditions)}")
            logger.info(f"   - Keyword vocabulary: {len(keyword_vocabulary)}")
            logger.info(f"   - Precision@k values: {metrics_config.precision_at_k_values}")
            
            return self.metrics_computer
            
        except Exception as e:
            logger.error(f"❌ Failed to setup metrics: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def create_dummy_datasets(self, num_samples: int = 100) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create dummy datasets for testing the training pipeline."""
        logger.info(f"Creating dummy datasets with {num_samples} samples each...")
        
        try:
            from training.data_collator import create_dummy_features
            
            # Create train/val/test splits
            train_size = int(0.7 * num_samples)
            val_size = int(0.15 * num_samples)
            test_size = num_samples - train_size - val_size
            
            # Generate dummy features with keywords
            train_features = [
                create_dummy_features(batch_size=1, include_keywords=True)[0] 
                for _ in range(train_size)
            ]
            val_features = [
                create_dummy_features(batch_size=1, include_keywords=True)[0] 
                for _ in range(val_size)
            ]
            test_features = [
                create_dummy_features(batch_size=1, include_keywords=True)[0] 
                for _ in range(test_size)
            ]
            
            logger.info(f"✅ Dummy datasets created successfully")
            logger.info(f"   - Train: {len(train_features)} samples")
            logger.info(f"   - Validation: {len(val_features)} samples")
            logger.info(f"   - Test: {len(test_features)} samples")
            
            return train_features, val_features, test_features
            
        except Exception as e:
            logger.error(f"❌ Failed to create dummy datasets: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def compute_metrics_wrapper(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Wrapper function for computing metrics during training."""
        try:
            # Extract predictions and labels
            predictions_raw = eval_pred.predictions
            labels_raw = eval_pred.label_ids
            
            # Handle different prediction formats (model may return dict or tuple)
            if isinstance(predictions_raw, dict):
                # Model returns dictionary with multiple outputs
                predictions = {
                    'disease_logits': predictions_raw.get('disease_logits', predictions_raw.get('logits')),
                    'global_keywords': predictions_raw.get('global_keyword_logits'),
                    'region_keywords': predictions_raw.get('region_keyword_logits')
                }
            elif isinstance(predictions_raw, (tuple, list)):
                # Model returns tuple/list (disease_logits, global_keywords, region_keywords)
                predictions = {
                    'disease_logits': predictions_raw[0],
                    'global_keywords': predictions_raw[1] if len(predictions_raw) > 1 else None,
                    'region_keywords': predictions_raw[2] if len(predictions_raw) > 2 else None
                }
            else:
                # Single tensor output (disease only)
                predictions = {'disease_logits': predictions_raw}
            
            # Handle different label formats
            if isinstance(labels_raw, dict):
                targets = labels_raw
            else:
                # Assume labels_raw is disease labels only
                targets = {'disease_labels': labels_raw}
            
            # Compute metrics using our hybrid metrics computer
            metrics = self.metrics_computer.compute(predictions, targets)
            
            # Add primary metrics for HuggingFace trainer compatibility
            if 'disease_micro_f1' in metrics:
                metrics['f1'] = metrics['disease_micro_f1']
                metrics['precision'] = metrics.get('disease_micro_precision', 0.0)
                metrics['recall'] = metrics.get('disease_micro_recall', 0.0)
                metrics['accuracy'] = metrics.get('disease_exact_match_accuracy', 0.0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in compute_metrics_wrapper: {e}")
            logger.error(traceback.format_exc())
            # Return dummy metrics to avoid training crash
            return {
                'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0,
                'disease_micro_f1': 0.0, 'keyword_global_f1': 0.0
            }
    
    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments for HuggingFace Trainer."""
        logger.info("Setting up training arguments...")
        
        try:
            # Create output directory
            output_dir = OUTPUT_CONFIG['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(OUTPUT_CONFIG['logging_dir'], exist_ok=True)
            
            # Configure for small-scale testing
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.training_config.get('num_epochs', 3),  # Small for testing
                per_device_train_batch_size=BATCH_CONFIG.get('batch_size', 4),  # Small batch size
                per_device_eval_batch_size=BATCH_CONFIG.get('eval_batch_size', 4),
                gradient_accumulation_steps=BATCH_CONFIG.get('gradient_accumulation_steps', 2),
                learning_rate=self.training_config.get('learning_rate', 3e-5),
                weight_decay=self.training_config.get('weight_decay', 0.01),
                warmup_steps=self.training_config.get('warmup_steps', 10),
                
                # Evaluation and logging
                eval_strategy=EVALUATION_CONFIG.get('eval_strategy', 'steps'),
                eval_steps=EVALUATION_CONFIG.get('eval_steps', 20),  # Frequent evaluation
                logging_strategy=LOGGING_CONFIG.get('logging_strategy', 'steps'),
                logging_steps=LOGGING_CONFIG.get('logging_steps', 10),
                
                # Checkpointing
                save_strategy=CHECKPOINT_CONFIG.get('save_strategy', 'steps'),
                save_steps=CHECKPOINT_CONFIG.get('save_steps', 20),
                save_total_limit=CHECKPOINT_CONFIG.get('save_total_limit', 2),
                load_best_model_at_end=CHECKPOINT_CONFIG.get('load_best_model_at_end', True),
                
                # Metrics and optimization
                metric_for_best_model=EVALUATION_CONFIG.get('metric_for_best_model', 'eval_f1'),
                greater_is_better=EVALUATION_CONFIG.get('greater_is_better', True),
                
                # Hardware and memory optimization
                dataloader_num_workers=BATCH_CONFIG.get('dataloader_num_workers', 0),  # 0 for stability
                remove_unused_columns=False,  # Keep all columns for multi-task
                fp16=MEMORY_CONFIG.get('fp16_training', False),  # Disable for stability
                dataloader_pin_memory=MEMORY_CONFIG.get('pin_memory', False),
                
                # Logging and reporting
                logging_dir=OUTPUT_CONFIG['logging_dir'],
                report_to=LOGGING_CONFIG.get('report_to', ['tensorboard']),
                run_name=OUTPUT_CONFIG.get('run_name', 'hybrid_training'),
                
                # Advanced settings
                prediction_loss_only=False,  # We need predictions for metrics
                include_inputs_for_metrics=False,
                ignore_data_skip=True,
                
                # Gradient optimization
                max_grad_norm=self.training_config.get('max_grad_norm', 1.0),
                optim='adamw_torch',
                lr_scheduler_type='cosine',
                
                # Early stopping will be handled by callback
            )
            
            logger.info(f"✅ Training arguments configured")
            logger.info(f"   - Epochs: {training_args.num_train_epochs}")
            logger.info(f"   - Batch size: {training_args.per_device_train_batch_size}")
            logger.info(f"   - Learning rate: {training_args.learning_rate}")
            logger.info(f"   - Output dir: {training_args.output_dir}")
            
            return training_args
            
        except Exception as e:
            logger.error(f"❌ Failed to setup training arguments: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def train(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            num_samples: Number of samples per dataset split for testing
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("🚀 Starting hybrid multi-task training...")
        logger.info(f"Configuration:")
        logger.info(f"  - Samples per split: {num_samples}")
        logger.info(f"  - Model: HybridMedicalModel")
        logger.info(f"  - Tasks: Disease classification + Keyword prediction")
        
        try:
            # 1. Setup all components
            logger.info("\n" + "="*60)
            logger.info("STEP 1: Setting up training components")
            logger.info("="*60)
            
            model = self.setup_model()
            data_collator = self.setup_data_collator()
            metrics_computer = self.setup_metrics()
            
            # 2. Create datasets
            logger.info("\n" + "="*60)
            logger.info("STEP 2: Creating datasets")
            logger.info("="*60)
            
            train_features, val_features, test_features = self.create_dummy_datasets(num_samples)
            
            # 3. Setup training
            logger.info("\n" + "="*60)
            logger.info("STEP 3: Setting up trainer")
            logger.info("="*60)
            
            training_args = self.setup_training_arguments()
            
            # Create early stopping callback
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_CONFIG.get('patience', 3),
                early_stopping_threshold=EARLY_STOPPING_CONFIG.get('min_delta', 0.001)
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_features,
                eval_dataset=val_features,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics_wrapper,
                callbacks=[early_stopping]
            )
            
            logger.info(f"✅ Trainer created successfully")
            
            # 4. Run training
            logger.info("\n" + "="*60)
            logger.info("STEP 4: Running training")
            logger.info("="*60)
            
            # Check for existing checkpoints
            checkpoint_dir = None
            if os.path.exists(training_args.output_dir):
                checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith('checkpoint-')]
                if checkpoints:
                    checkpoint_dir = os.path.join(training_args.output_dir, sorted(checkpoints)[-1])
                    logger.info(f"Found checkpoint: {checkpoint_dir}")
            
            # Start training
            train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
            
            logger.info(f"✅ Training completed successfully")
            logger.info(f"   - Final train loss: {train_result.training_loss:.4f}")
            logger.info(f"   - Training steps: {train_result.global_step}")
            
            # 5. Evaluate on test set
            logger.info("\n" + "="*60)
            logger.info("STEP 5: Testing on held-out test set")
            logger.info("="*60)
            
            test_results = trainer.evaluate(test_features, metric_key_prefix="test")
            
            logger.info(f"✅ Test evaluation completed")
            logger.info(f"Test Results:")
            for key, value in test_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  - {key}: {value:.4f}")
            
            # 6. Save final model
            logger.info("\n" + "="*60)
            logger.info("STEP 6: Saving final model")
            logger.info("="*60)
            
            final_model_dir = os.path.join(training_args.output_dir, 'final_model')
            trainer.save_model(final_model_dir)
            
            logger.info(f"✅ Final model saved to: {final_model_dir}")
            
            # 7. Compile results
            results = {
                'train_result': train_result,
                'test_results': test_results,
                'model_path': final_model_dir,
                'config': {
                    'model_config': self.model_config,
                    'training_config': self.training_config,
                    'data_config': self.data_config
                }
            }
            
            # Save results to file
            results_file = os.path.join(training_args.output_dir, 'training_results.json')
            with open(results_file, 'w') as f:
                # Convert non-serializable objects to strings
                serializable_results = {}
                for key, value in results.items():
                    if key == 'train_result':
                        serializable_results[key] = {
                            'training_loss': float(value.training_loss),
                            'global_step': int(value.global_step),
                            'epoch': float(value.epoch)
                        }
                    elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                        serializable_results[key] = value
                    else:
                        serializable_results[key] = str(value)
                
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"✅ Results saved to: {results_file}")
            
            # 8. Summary
            logger.info("\n" + "="*80)
            logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"Summary:")
            logger.info(f"  - Training samples: {len(train_features)}")
            logger.info(f"  - Validation samples: {len(val_features)}")
            logger.info(f"  - Test samples: {len(test_features)}")
            logger.info(f"  - Training epochs: {train_result.epoch:.1f}")
            logger.info(f"  - Final train loss: {train_result.training_loss:.4f}")
            
            # Key test metrics
            if 'test_f1' in test_results:
                logger.info(f"  - Test F1: {test_results['test_f1']:.4f}")
            if 'test_disease_micro_f1' in test_results:
                logger.info(f"  - Test Disease F1: {test_results['test_disease_micro_f1']:.4f}")
            if 'test_keyword_global_f1' in test_results:
                logger.info(f"  - Test Keyword F1: {test_results['test_keyword_global_f1']:.4f}")
            
            logger.info(f"  - Model saved: {final_model_dir}")
            logger.info(f"  - Results saved: {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.error(traceback.format_exc())
            raise


def main():
    """Main training function."""
    logger.info("="*80)
    logger.info("🚀 HYBRID RADBERT + MEDCLIP TRAINING")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Create trainer
        trainer = SimpleHybridTrainer()
        
        # Run training with small dataset for testing
        results = trainer.train(num_samples=50)  # Small for quick testing
        
        logger.info("\n" + "="*80)
        logger.info("✅ SUCCESS: Training pipeline completed successfully!")
        logger.info("="*80)
        logger.info("The hybrid model training demonstrates:")
        logger.info("  ✅ Multi-task learning (disease + keywords)")
        logger.info("  ✅ Enhanced data collation with keyword labels")
        logger.info("  ✅ Comprehensive metrics computation")
        logger.info("  ✅ Memory-optimized training pipeline")
        logger.info("  ✅ Checkpoint management and recovery")
        logger.info("  ✅ Integration of all Phase 4 components")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ FAILED: Training pipeline failed: {e}")
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n🎯 Ready to proceed to Phase 5: Integration & Testing")
    else:
        print(f"\n⚠️  Please resolve issues before continuing") 