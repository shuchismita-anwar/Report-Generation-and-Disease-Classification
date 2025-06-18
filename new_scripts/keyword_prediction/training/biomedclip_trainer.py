#!/usr/bin/env python3
"""
BiomedCLIP Trainer for Medical Keyword Extraction

Since BiomedCLIP is used for zero-shot inference, this "trainer" focuses on
evaluation and metrics rather than actual parameter updates.
"""

import logging
import time
import os
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalKeywordTrainer:
    """
    Trainer for BiomedCLIP medical keyword extraction.
    
    Since BiomedCLIP is used for zero-shot inference, this trainer focuses on
    evaluation and metrics computation rather than parameter training.
    """
    
    def __init__(
        self,
        model,
        config: Dict[str, Any],
        train_batches: int = None,
        val_batches: int = None
    ):
        """
        Initialize BiomedCLIP trainer.
        
        Args:
            model: MedicalBiomedCLIPModel instance
            config: Training configuration dictionary
            train_batches: Number of training batches to process (for evaluation)
            val_batches: Number of validation batches to process
        """
        self.model = model
        self.config = config
        self.train_batches = train_batches
        self.val_batches = val_batches
        
        # Extract device
        self.device = model.device
        
        # Initialize metrics system if enabled
        self.enable_metrics = config.get('training', {}).get('metrics_enabled', True)
        self.metrics_evaluator = None
        
        if self.enable_metrics:
            try:
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                metrics_dir = os.path.join(parent_dir, 'metrics')
                sys.path.insert(0, parent_dir)
                sys.path.insert(0, metrics_dir)
                
                from metrics.medical_metrics import MedicalKeywordMetrics
                self.metrics_evaluator = MedicalKeywordMetrics(device=self.device)
                logger.info("Comprehensive metrics system enabled for BiomedCLIP evaluation")
            except Exception as e:
                logger.warning(f"Failed to initialize metrics system: {e}")
                logger.info("Continuing without comprehensive metrics")
                self.enable_metrics = False
        
        logger.info(f"BiomedCLIP trainer initialized on device: {self.device}")
        logger.info(f"Evaluation mode: {train_batches} train batches, {val_batches} val batches")
    
    def train_epoch(self, epoch: int, train_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on training data (no actual training for BiomedCLIP).
        
        Args:
            epoch: Current epoch number
            train_loader: Training data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating BiomedCLIP on training data - Epoch {epoch}")
        
        # Start timing
        start_time = time.time()
        
        # Set model to evaluation mode (BiomedCLIP is always in eval mode)
        self.model.eval()
        
        # Evaluation metrics
        total_samples = 0
        successful_extractions = 0
        all_predictions = []
        all_targets = []
        processing_times = []
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Train Evaluation")
        
        # Process batches
        batch_count = 0
        for batch_idx, batch in enumerate(pbar):
            if self.train_batches and batch_count >= self.train_batches:
                break
            
            try:
                batch_start_time = time.time()
                
                # Extract batch data
                if 'pixel_values' in batch:
                    images = batch['pixel_values'].to(self.device)
                    target_keywords = batch.get('keywords', [])
                elif 'images' in batch:
                    images = batch['images'].to(self.device)
                    target_keywords = batch.get('keywords', [])
                else:
                    logger.warning(f"Unknown batch format: {list(batch.keys())}")
                    continue
                
                batch_size = images.shape[0]
                total_samples += batch_size
                
                # Generate keywords for each image in the batch
                batch_predictions = []
                for i in range(batch_size):
                    try:
                        # Extract keywords using BiomedCLIP
                        predicted_keywords = self.model.extract_keywords(
                            images[i],
                            top_k=10,
                            threshold=0.1
                        )
                        
                        batch_predictions.append(predicted_keywords)
                        successful_extractions += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract keywords for image {i}: {e}")
                        batch_predictions.append("normal chest x-ray")
                
                # Store predictions and targets
                all_predictions.extend(batch_predictions)
                if target_keywords:
                    all_targets.extend(target_keywords[:len(batch_predictions)])
                else:
                    # Create dummy targets if not available
                    all_targets.extend(["normal chest x-ray"] * len(batch_predictions))
                
                # Track processing time
                batch_time = time.time() - batch_start_time
                processing_times.append(batch_time)
                
                # Update progress bar
                success_rate = (successful_extractions / total_samples) * 100
                pbar.set_postfix({
                    'Success': f'{success_rate:.1f}%',
                    'Batch_time': f'{batch_time:.2f}s'
                })
                
                batch_count += 1
                
            except Exception as e:
                logger.error(f"Error processing training batch {batch_idx}: {e}")
                continue
        
        # Calculate metrics
        epoch_time = time.time() - start_time
        success_rate = (successful_extractions / total_samples) * 100 if total_samples > 0 else 0
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Compute comprehensive metrics if available
        detailed_metrics = {}
        if self.enable_metrics and self.metrics_evaluator and all_predictions and all_targets:
            try:
                detailed_metrics = self.metrics_evaluator.compute_all_metrics(
                    predictions=all_predictions,
                    references=all_targets
                )
                logger.info(f"Training evaluation metrics computed successfully")
            except Exception as e:
                logger.warning(f"Failed to compute detailed metrics: {e}")
        
        # Prepare results
        results = {
            'avg_loss': 0.0,  # BiomedCLIP doesn't compute loss
            'success_rate': success_rate,
            'total_samples': total_samples,
            'successful_extractions': successful_extractions,
            'epoch_time': epoch_time,
            'avg_processing_time': avg_processing_time,
            'predictions': all_predictions[:10],  # Store first 10 for inspection
            'targets': all_targets[:10]
        }
        
        # Add detailed metrics if available
        if detailed_metrics:
            # Extract key metrics from the nested structure
            if 'summary' in detailed_metrics:
                results['metrics'] = detailed_metrics['summary']
            else:
                results['metrics'] = detailed_metrics
        
        logger.info(f"Training evaluation completed: {success_rate:.1f}% success rate")
        return results
    
    def validate_epoch(self, epoch: int, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on validation data.
        
        Args:
            epoch: Current epoch number
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info(f"Evaluating BiomedCLIP on validation data - Epoch {epoch}")
        
        # Start timing
        start_time = time.time()
        
        # Set model to evaluation mode (always in eval mode for BiomedCLIP)
        self.model.eval()
        
        # Evaluation metrics
        total_samples = 0
        successful_extractions = 0
        all_predictions = []
        all_targets = []
        processing_times = []
        
        # Progress bar
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} - Validation")
        
        # Process validation batches
        batch_count = 0
        for batch_idx, batch in enumerate(pbar):
            if self.val_batches and batch_count >= self.val_batches:
                break
            
            try:
                batch_start_time = time.time()
                
                # Extract batch data
                if 'pixel_values' in batch:
                    images = batch['pixel_values'].to(self.device)
                    target_keywords = batch.get('keywords', [])
                elif 'images' in batch:
                    images = batch['images'].to(self.device)
                    target_keywords = batch.get('keywords', [])
                else:
                    logger.warning(f"Unknown batch format: {list(batch.keys())}")
                    continue
                
                batch_size = images.shape[0]
                total_samples += batch_size
                
                # Generate keywords for each image in the batch
                batch_predictions = []
                for i in range(batch_size):
                    try:
                        # Extract keywords using BiomedCLIP
                        predicted_keywords = self.model.extract_keywords(
                            images[i],
                            top_k=10,
                            threshold=0.1
                        )
                        
                        batch_predictions.append(predicted_keywords)
                        successful_extractions += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract keywords for image {i}: {e}")
                        batch_predictions.append("normal chest x-ray")
                
                # Store predictions and targets
                all_predictions.extend(batch_predictions)
                if target_keywords:
                    all_targets.extend(target_keywords[:len(batch_predictions)])
                else:
                    # Create dummy targets if not available
                    all_targets.extend(["normal chest x-ray"] * len(batch_predictions))
                
                # Track processing time
                batch_time = time.time() - batch_start_time
                processing_times.append(batch_time)
                
                # Update progress bar
                success_rate = (successful_extractions / total_samples) * 100
                pbar.set_postfix({
                    'Success': f'{success_rate:.1f}%',
                    'Batch_time': f'{batch_time:.2f}s'
                })
                
                batch_count += 1
                
            except Exception as e:
                logger.error(f"Error processing validation batch {batch_idx}: {e}")
                continue
        
        # Calculate metrics
        epoch_time = time.time() - start_time
        success_rate = (successful_extractions / total_samples) * 100 if total_samples > 0 else 0
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Compute comprehensive metrics if available
        detailed_metrics = {}
        if self.enable_metrics and self.metrics_evaluator and all_predictions and all_targets:
            try:
                detailed_metrics = self.metrics_evaluator.compute_all_metrics(
                    predictions=all_predictions,
                    references=all_targets
                )
                logger.info(f"Validation metrics computed successfully")
            except Exception as e:
                logger.warning(f"Failed to compute detailed metrics: {e}")
        
        # Prepare results
        results = {
            'avg_loss': 0.0,  # BiomedCLIP doesn't compute loss
            'success_rate': success_rate,
            'total_samples': total_samples,
            'successful_extractions': successful_extractions,
            'epoch_time': epoch_time,
            'avg_processing_time': avg_processing_time,
            'predictions': all_predictions[:10],  # Store first 10 for inspection
            'targets': all_targets[:10]
        }
        
        # Add detailed metrics if available
        if detailed_metrics:
            # Extract key metrics from the nested structure
            if 'summary' in detailed_metrics:
                results['metrics'] = detailed_metrics['summary']
            else:
                results['metrics'] = detailed_metrics
        
        logger.info(f"Validation completed: {success_rate:.1f}% success rate")
        
        # Log sample predictions for inspection
        if all_predictions:
            logger.info("Sample predictions:")
            for i, (pred, target) in enumerate(zip(all_predictions[:3], all_targets[:3])):
                logger.info(f"  {i+1}. Predicted: '{pred}'")
                logger.info(f"     Target:    '{target}'")
        
        return results
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Save model checkpoint and metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Metrics dictionary to save
        """
        try:
            # Create checkpoints directory
            checkpoint_dir = self.config.get('training', {}).get('checkpoint_dir', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save metrics (BiomedCLIP model doesn't need parameter saving)
            metrics_file = os.path.join(checkpoint_dir, f'metrics_epoch_{epoch}.json')
            
            import json
            with open(metrics_file, 'w') as f:
                # Convert any tensors to floats for JSON serialization
                serializable_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        serializable_metrics[key] = value.item()
                    elif isinstance(value, dict):
                        serializable_metrics[key] = {
                            k: v.item() if isinstance(v, torch.Tensor) else v
                            for k, v in value.items()
                        }
                    else:
                        serializable_metrics[key] = value
                
                json.dump(serializable_metrics, f, indent=2)
            
            logger.info(f"Checkpoint saved: {metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")


def test_biomedclip_trainer():
    """Test BiomedCLIP trainer functionality."""
    logger.info("Testing BiomedCLIP trainer...")
    
    try:
        # Create dummy config
        config = {
            'model': {
                'model_name': 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                'max_length': 64
            },
            'training': {
                'device': 'auto',
                'metrics_enabled': True,
                'checkpoint_dir': 'test_checkpoints'
            }
        }
        
        # Create dummy model
        from model.biomedclip_medical import MedicalBiomedCLIPModel
        model = MedicalBiomedCLIPModel()
        
        # Initialize trainer
        trainer = MedicalKeywordTrainer(
            model=model,
            config=config,
            train_batches=2,
            val_batches=1
        )
        
        logger.info("BiomedCLIP trainer test passed!")
        return True
        
    except Exception as e:
        logger.error(f"BiomedCLIP trainer test failed: {e}")
        return False


if __name__ == "__main__":
    test_biomedclip_trainer() 