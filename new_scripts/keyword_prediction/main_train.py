#!/usr/bin/env python3
"""
Medical Keyword Prediction Training Pipeline using BiomedCLIP
Updated to use Microsoft BiomedCLIP instead of BLIP-2 for robust medical image analysis
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import time

# Configure logging to handle Unicode
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random

# Import configuration and data loaders
from config.config_loader import load_config, update_config_from_args
from data.data_loader import create_data_loaders

# Import BiomedCLIP model (replaces BLIP-2)
from model.biomedclip_medical import MedicalBiomedCLIPModel

# Import BiomedCLIP trainer
from training.biomedclip_trainer import MedicalKeywordTrainer

def setup_logging():
    """Setup logging configuration."""
    os.makedirs('logs', exist_ok=True)
    logger.info("Logging setup complete. Log file: logs/training.log")

def setup_reproducibility(seed=42):
    """Setup reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable CuDNN optimizations
    cudnn.benchmark = True
    logger.info("CuDNN benchmark enabled for speed")
    logger.info(f"Random seeds set to {seed}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Medical Keyword Prediction Training with BiomedCLIP')
    
    # Model configuration
    parser.add_argument('--model-name', type=str, 
                       default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                       help='BiomedCLIP model identifier')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                       help='Learning rate for training')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=20,
                       help='Number of training epochs')
    
    # Training configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--metrics', action='store_true', default=True,
                       help='Enable comprehensive metrics evaluation')
    parser.add_argument('--checkpointing', action='store_true', default=True,
                       help='Enable model checkpointing')
    
    # Short training mode for testing
    parser.add_argument('--short-training', action='store_true',
                       help='Use limited data for quick testing')
    parser.add_argument('--epochs', type=int, help='Override epochs (alias for --num-epochs)')
    
    return parser.parse_args()

def display_config_summary(config, args):
    """Display training configuration summary."""
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model Name               : {config['model']['model_name']}")
    logger.info(f"Learning Rate            : {config['training']['learning_rate']}")
    logger.info(f"Batch Size               : {config['training']['batch_size']}")
    logger.info(f"Num Epochs               : {config['training']['num_epochs']}")
    logger.info(f"Device                   : {config['training']['device']}")
    logger.info(f"Metrics Enabled          : {config['training']['metrics_enabled']}")
    logger.info(f"Checkpointing Enabled    : {config['training']['checkpointing_enabled']}")
    logger.info(f"Training Mode            : {'short' if args.short_training else 'full'}")
    logger.info("=" * 60)

def main():
    """Main training pipeline."""
    setup_logging()
    setup_reproducibility()
    
    logger.info("Medical keyword training pipeline initialized")
    
    # Parse arguments and load configuration
    args = parse_arguments()
    config = load_config()
    
    # Override epochs if specified
    if args.epochs:
        args.num_epochs = args.epochs
    
    # Update configuration from command line arguments
    config_changes = update_config_from_args(config, args)
    display_config_summary(config, args)
    
    if config_changes > 0:
        logger.info(f"Applied {config_changes} configuration overrides from command line")
    
    logger.info("Starting Medical Keyword Prediction Training Pipeline")
    
    try:
        # Initialize BiomedCLIP model
        logger.info(f"Loading BiomedCLIP model: {config['model']['model_name']}")
        model = MedicalBiomedCLIPModel(
            model_name=config['model']['model_name'],
            device=config['training']['device'],
            max_length=config['model']['max_length']
        )
        
        logger.info(f"Model loaded: {config['model']['model_name']}")
        
        # Count trainable parameters (BiomedCLIP doesn't need training, it's zero-shot)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        
        # Setup data loaders
        logger.info("Setting up data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            config=config,
            model=model  # Pass BiomedCLIP model instead of processor
        )
        
        logger.info("Data loaders created:")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")
        
        # Setup trainer
        logger.info("Setting up trainer...")
        
        # Adjust batch limits for short training
        if args.short_training:
            train_batches = min(100, len(train_loader))
            val_batches = min(25, len(val_loader))
            logger.info(f"Short training mode: Limited to {train_batches} train batches, {val_batches} val batches")
        else:
            train_batches = len(train_loader)
            val_batches = len(val_loader)
        
        trainer = MedicalKeywordTrainer(
            model=model,
            config=config,
            train_batches=train_batches,
            val_batches=val_batches
        )
        
        logger.info("Trainer setup complete")
        
        # Start training
        logger.info("Starting training process...")
        num_epochs = config['training']['num_epochs']
        logger.info(f"Training for {num_epochs} epochs")
        
        training_start_time = time.time()
        best_quality = 0.0
        
        for epoch in range(1, num_epochs + 1):
            logger.info("=" * 60)
            logger.info(f"EPOCH {epoch}/{num_epochs}")
            logger.info("=" * 60)
            
            # Training phase
            train_results = trainer.train_epoch(epoch, train_loader)
            
            # Validation phase
            val_results = trainer.validate_epoch(epoch, val_loader)
            
            # Log epoch results
            logger.info(f"Epoch {epoch} Results:")
            logger.info(f"  Training loss: {train_results['avg_loss']:.4f}")
            logger.info(f"  Validation loss: {val_results['avg_loss']:.4f}")
            logger.info(f"  Training success rate: {train_results['success_rate']:.1f}%")
            
            if 'metrics' in val_results:
                logger.info(f"  Comprehensive Metrics:")
                logger.info(f"     Overall Quality: {val_results['metrics']['overall_quality']:.4f}")
                logger.info(f"     BLEU-4: {val_results['metrics']['bleu_4']:.4f}")
                logger.info(f"     ROUGE-1 F1: {val_results['metrics']['rouge1_f1']:.4f}")
                logger.info(f"     Medical Coverage: {val_results['metrics']['medical_coverage']:.4f}")
            
            logger.info(f"  Epoch time: {val_results['epoch_time']:.2f}s")
            
            # Check for best model
            current_quality = val_results.get('metrics', {}).get('overall_quality', 0.0)
            if current_quality > best_quality:
                best_quality = current_quality
                logger.info(f"NEW BEST MODEL! Quality: {best_quality:.4f}")
            
            # Save checkpoints
            if config['training']['checkpointing_enabled']:
                trainer.save_checkpoint(epoch, val_results)
        
        # Training completed
        total_time = time.time() - training_start_time
        avg_time_per_epoch = total_time / num_epochs
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total epochs: {num_epochs}")
        logger.info(f"Total training time: {total_time:.2f}s ({total_time/60:.1f}min)")
        logger.info(f"Average time per epoch: {avg_time_per_epoch:.2f}s")
        logger.info(f"Best validation quality: {best_quality:.4f}")
        logger.info(f"Training mode: {'Short' if args.short_training else 'Full'}")
        logger.info("=" * 60)
        
        logger.info("Training pipeline completed successfully!")
        
        print(f"\nTraining completed successfully!")
        print(f"Best validation quality: {best_quality:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise e

if __name__ == "__main__":
    main() 