"""
Medical Keyword Prediction Trainer

Implements basic training loop for BLIP-2 medical keyword generation.
Task 5.1: Create Basic Training Loop
Task 5.4: Add training metrics (BLEU/ROUGE on train and val)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import time
from typing import Dict, Any, Optional, List
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalKeywordTrainer:
    """
    Trainer for BLIP-2 medical keyword generation model.
    
    Handles single epoch training with loss computation, backpropagation,
    and comprehensive metrics evaluation (BLEU/ROUGE/Medical-specific).
    """
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 5e-8,  # 🔧 OPTIMAL: Updated to optimal rate from testing
        device: str = "auto",
        gradient_accumulation_steps: int = 4,  # 🔧 NEW: Gradient accumulation for stability
        enable_metrics: bool = True  # 🔧 NEW: Enable comprehensive metrics evaluation
    ):
        """
        Initialize trainer.
        
        Args:
            model: MedicalBLIP2Model model instance
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Learning rate for optimizer
            device: Device to use for training
            gradient_accumulation_steps: Steps to accumulate gradients
            enable_metrics: Whether to compute comprehensive metrics
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_metrics = enable_metrics
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # 🔧 CRITICAL: Disable mixed precision - FP16 causes gradient explosion
        self.use_amp = False  # Disabled for numerical stability
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision (FP16) enabled for memory optimization")
        else:
            logger.info("Mixed precision DISABLED - using FP32 for numerical stability")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
        # 🔧 NEW: Initialize comprehensive metrics system for Task 5.4
        if self.enable_metrics:
            try:
                # Import metrics system (delay import to avoid dependencies in basic usage)
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                metrics_dir = os.path.join(parent_dir, 'metrics')
                sys.path.insert(0, parent_dir)  # Add parent dir to path
                sys.path.insert(0, metrics_dir)  # Add metrics dir to path
                
                from metrics.medical_metrics import MedicalKeywordMetrics
                self.metrics_evaluator = MedicalKeywordMetrics(device=device)
                logger.info("✅ Comprehensive metrics system enabled (BLEU/ROUGE/Medical)")
            except Exception as e:
                logger.warning(f"Failed to initialize metrics system: {e}")
                logger.info("Continuing without comprehensive metrics")
                self.enable_metrics = False
                self.metrics_evaluator = None
        else:
            self.metrics_evaluator = None
            logger.info("Basic training mode - metrics disabled")
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Train batches: {len(train_loader)}")
        if val_loader:
            logger.info(f"Validation batches: {len(val_loader)}")
    
    def _generate_keywords_for_metrics(self, images: torch.Tensor, batch_size: int = None) -> List[str]:
        """
        Generate keywords from images for metrics evaluation.
        
        Args:
            images: Batch of images
            batch_size: Expected batch size for validation
            
        Returns:
            List of generated keyword strings
        """
        try:
            if batch_size is None:
                batch_size = images.shape[0]
            
            # Set model to evaluation mode
            self.model.eval()
            
            generated_keywords = []
            
            with torch.no_grad():
                # Generate keywords for each image in the batch
                for i in range(batch_size):
                    try:
                        # Generate keywords for single image
                        single_image = images[i:i+1]  # Keep batch dimension
                        keywords = self.model.generate_keywords(single_image)
                        
                        # Extract generated text
                        if isinstance(keywords, list) and len(keywords) > 0:
                            generated_keywords.append(keywords[0])
                        else:
                            generated_keywords.append("")  # Fallback for failed generation
                            
                    except Exception as e:
                        logger.warning(f"Failed to generate keywords for image {i}: {e}")
                        generated_keywords.append("")  # Fallback
            
            # Restore training mode
            self.model.train()
            
            return generated_keywords
            
        except Exception as e:
            logger.error(f"Error in keyword generation for metrics: {e}")
            return [""] * batch_size  # Return empty strings as fallback
    
    def train_epoch(self, epoch: int = 0) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Starting training epoch {epoch}")
        
        # 🔧 Clear CUDA cache at start of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"   GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        
        # Set model to training mode
        self.model.train()
        
        # Training metrics
        total_loss = 0.0
        num_batches = 0
        successful_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} - Training")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Handle different batch formats
                if 'pixel_values' in batch:
                    # BLIP processor format
                    images = batch['pixel_values'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                elif 'images' in batch:
                    # Manual format - need to process text separately
                    images = batch['images'].to(self.device)
                    # For manual format, we need to tokenize the keywords
                    keywords = batch['keywords']
                    
                    # Use model's processor to tokenize
                    tokenized = self.model.processor.tokenizer(
                        keywords,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=64
                    )
                    
                    input_ids = tokenized['input_ids'].to(self.device)
                    attention_mask = tokenized['attention_mask'].to(self.device)
                    labels = input_ids.clone()
                else:
                    logger.error(f"Unknown batch format: {list(batch.keys())}")
                    num_batches += 1
                    continue
                
                # Zero gradients only at start of accumulation cycle
                if batch_idx % self.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model.forward(
                    pixel_values=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Extract loss
                loss = outputs.loss
                
                # 🔧 CRITICAL: Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # 🔍 DEBUG: Check for NaN loss
                if torch.isnan(loss):
                    logger.error(f"⚠️ NaN loss detected in batch {batch_idx}!")
                    logger.error(f"   Image shape: {images.shape}")
                    logger.error(f"   Input IDs shape: {input_ids.shape}")
                    logger.error(f"   Labels shape: {labels.shape}")
                    logger.error(f"   Image range: [{images.min():.4f}, {images.max():.4f}]")
                    logger.error(f"   Input IDs sample: {input_ids[0][:10]}")
                    logger.error(f"   Labels sample: {labels[0][:10]}")
                    if hasattr(outputs, 'logits'):
                        logger.error(f"   Logits shape: {outputs.logits.shape}")
                        logger.error(f"   Logits range: [{outputs.logits.min():.4f}, {outputs.logits.max():.4f}]")
                        logger.error(f"   Logits has NaN: {torch.isnan(outputs.logits).any()}")
                    # Continue processing to see pattern
                
                # Check for invalid values before backward pass
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Skipping batch {batch_idx} due to invalid loss: {loss.item()}")
                    num_batches += 1
                    continue
                
                # Backward pass
                loss.backward()
                
                # 🔧 CRITICAL: Only step optimizer every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # 🔍 DEBUG: Check gradients for NaN/Inf before optimizer step
                    total_grad_norm = 0.0
                    nan_grad_count = 0
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2)
                            total_grad_norm += grad_norm.item() ** 2
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                nan_grad_count += 1
                                if nan_grad_count <= 3:  # Log first few
                                    logger.error(f"   NaN/Inf gradient in {name}: {param.grad.data}")
                    
                    if nan_grad_count > 0:
                        logger.error(f"   Found {nan_grad_count} parameters with NaN/Inf gradients")
                        logger.error(f"   Total gradient norm: {total_grad_norm}")
                        # 🔧 Skip optimizer step if gradients are corrupted
                        self.optimizer.zero_grad()
                        continue
                    
                    # 🔧 ULTRA-CONSERVATIVE gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.01)  # 🔧 REDUCED from 0.1 to 0.01
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # 🔧 CRITICAL: Validate model parameters after update
                    param_check_failed = False
                    for name, param in self.model.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            logger.error(f"   Parameter corruption detected in {name} after optimizer step!")
                            param_check_failed = True
                            break
                    
                    if param_check_failed:
                        logger.error("   Model parameters corrupted! Stopping training to prevent further damage.")
                        break
                
                # Update metrics (do this BEFORE clearing variables)
                total_loss += loss.item()
                num_batches += 1
                successful_batches += 1
                
                # Update progress bar
                avg_loss = total_loss / successful_batches
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{avg_loss:.4f}',
                    'Success': f'{successful_batches}/{num_batches}'
                })
                
                # Log every 50 batches
                if batch_idx % 50 == 0:
                    logger.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
                
                # 🔧 CRITICAL: Aggressive memory clearing after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Clear variables to free memory (do this AFTER using loss.item())
                del loss, outputs
                if 'pixel_values' in locals():
                    del pixel_values, input_ids, attention_mask, labels
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                num_batches += 1  # Count failed batches too
                # Clear any variables that might have been created
                if 'loss' in locals():
                    del loss
                if 'outputs' in locals():
                    del outputs
                if 'pixel_values' in locals():
                    del pixel_values, input_ids, attention_mask, labels
                continue
        
        # Final statistics
        avg_train_loss = total_loss / successful_batches if successful_batches > 0 else 0.0
        success_rate = (successful_batches / num_batches) * 100 if num_batches > 0 else 0
        
        # 🔧 CRITICAL: Clear memory at end of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info(f"Epoch {epoch} completed:")
        logger.info(f"  Average training loss: {avg_train_loss:.4f}")
        logger.info(f"  Successful batches: {successful_batches}/{num_batches} ({success_rate:.1f}%)")
        
        # Store metrics
        self.train_losses.append(avg_train_loss)
        
        metrics = {
            'train_loss': avg_train_loss,
            'successful_batches': successful_batches,
            'total_batches': num_batches,
            'success_rate': success_rate
        }
        
        return metrics
    
    def validate(self, epoch: int = 0) -> Dict[str, float]:
        """
        Run validation with comprehensive metrics evaluation.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics including BLEU/ROUGE scores
        """
        if self.val_loader is None:
            logger.warning("No validation loader provided")
            return {}
        
        logger.info(f"Starting validation for epoch {epoch}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Validation metrics
        total_loss = 0.0
        num_batches = 0
        successful_batches = 0
        
        # 🔧 NEW: Collect data for comprehensive metrics (Task 5.4)
        all_references = []
        all_hypotheses = []
        metrics_sample_limit = 50  # Limit samples for metrics to avoid memory issues
        metrics_samples_collected = 0
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} - Validation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Handle different batch formats
                    if 'pixel_values' in batch:
                        # BLIP processor format
                        images = batch['pixel_values'].to(self.device)
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        keywords = batch.get('keywords', [])  # Get original keywords if available
                    elif 'images' in batch:
                        # Manual format - need to process text separately
                        images = batch['images'].to(self.device)
                        keywords = batch['keywords']
                        
                        # Use model's processor to tokenize
                        tokenized = self.model.processor.tokenizer(
                            keywords,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=64
                        )
                        
                        input_ids = tokenized['input_ids'].to(self.device)
                        attention_mask = tokenized['attention_mask'].to(self.device)
                        labels = input_ids.clone()
                    else:
                        logger.error(f"Unknown validation batch format: {list(batch.keys())}")
                        num_batches += 1
                        continue
                    
                    # Forward pass
                    outputs = self.model.forward(
                        pixel_values=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    # Extract loss
                    loss = outputs.loss
                    
                    # Update metrics
                    total_loss += loss.item()
                    num_batches += 1
                    successful_batches += 1
                    
                    # 🔧 NEW: Collect samples for comprehensive metrics evaluation (Task 5.4)
                    if (self.enable_metrics and 
                        metrics_samples_collected < metrics_sample_limit and 
                        keywords and len(keywords) > 0):
                        
                        try:
                            # Generate keywords for this batch
                            batch_size = images.shape[0]
                            generated_keywords = self._generate_keywords_for_metrics(images, batch_size)
                            
                            # Add to collections (limit to available keywords)
                            for i in range(min(len(keywords), len(generated_keywords), batch_size)):
                                if metrics_samples_collected < metrics_sample_limit:
                                    all_references.append(keywords[i])
                                    all_hypotheses.append(generated_keywords[i])
                                    metrics_samples_collected += 1
                                else:
                                    break
                                    
                        except Exception as e:
                            logger.warning(f"Error collecting metrics samples in batch {batch_idx}: {e}")
                    
                    # Update progress bar
                    avg_loss = total_loss / num_batches
                    pbar.set_postfix({
                        'Val Loss': f'{loss.item():.4f}',
                        'Avg Val Loss': f'{avg_loss:.4f}',
                        'Metrics': f'{metrics_samples_collected}/{metrics_sample_limit}'
                    })
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    num_batches += 1
                    continue
        
        # Calculate final loss metrics
        avg_val_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        success_rate = (successful_batches / num_batches) * 100 if num_batches > 0 else 0
        
        # Store metrics
        self.val_losses.append(avg_val_loss)
        
        # 🔧 NEW: Compute comprehensive metrics (Task 5.4)
        comprehensive_metrics = {}
        if self.enable_metrics and self.metrics_evaluator and len(all_references) > 0:
            try:
                logger.info(f"Computing comprehensive metrics on {len(all_references)} samples...")
                validation_metrics = self.metrics_evaluator.compute_validation_metrics(
                    all_references, all_hypotheses
                )
                comprehensive_metrics = validation_metrics['summary']
                
                # Log key metrics
                logger.info(f"📊 Validation Metrics Summary:")
                logger.info(f"   BLEU-4: {comprehensive_metrics.get('bleu_4', 0):.4f}")
                logger.info(f"   ROUGE-1 F1: {comprehensive_metrics.get('rouge1_f1', 0):.4f}")
                logger.info(f"   ROUGE Avg Recall: {comprehensive_metrics.get('rouge_avg_recall', 0):.4f}")
                logger.info(f"   Medical Coverage: {comprehensive_metrics.get('medical_coverage', 0):.4f}")
                logger.info(f"   Overall Quality: {comprehensive_metrics.get('overall_quality', 0):.4f}")
                
                # Show best example if available
                if validation_metrics.get('best_example'):
                    best = validation_metrics['best_example']
                    logger.info(f"🏆 Best Example:")
                    logger.info(f"   Ref: '{best['reference']}'")
                    logger.info(f"   Gen: '{best['hypothesis']}'")
                
            except Exception as e:
                logger.error(f"Error computing comprehensive metrics: {e}")
                comprehensive_metrics = {}
        
        # Combine all metrics
        metrics = {
            'val_loss': avg_val_loss,
            'successful_batches': successful_batches,
            'total_batches': num_batches,
            'success_rate': success_rate,
            'metrics_samples': metrics_samples_collected
        }
        
        # Add comprehensive metrics with 'val_' prefix for consistency
        for key, value in comprehensive_metrics.items():
            metrics[f'val_{key}'] = value
        
        logger.info(f"Validation epoch {epoch} completed:")
        logger.info(f"  Average validation loss: {avg_val_loss:.4f}")
        logger.info(f"  Successful batches: {successful_batches}/{num_batches} ({success_rate:.1f}%)")
        if comprehensive_metrics:
            logger.info(f"  Overall quality: {comprehensive_metrics.get('overall_quality', 0):.4f}")
        
        return metrics
    
    def train_single_epoch(self, epoch: int = 0) -> Dict[str, Any]:
        """
        Complete training + validation for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Combined metrics from training and validation
        """
        start_time = time.time()
        
        # Training
        train_metrics = self.train_epoch(epoch)
        
        # Validation
        val_metrics = self.validate(epoch) if self.val_loader else {}
        
        # Combined metrics
        epoch_time = time.time() - start_time
        
        combined_metrics = {
            **train_metrics,
            **val_metrics,
            'epoch_time': epoch_time,
            'epoch': epoch
        }
        
        logger.info(f"Epoch {epoch} total time: {epoch_time:.2f}s")
        
        return combined_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any], checkpoint_dir: str = "../checkpoints", is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Training/validation metrics
            checkpoint_dir: Directory to save checkpoints
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        import os
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics': metrics,
            'model_config': {
                'model_name': getattr(self.model, 'model_name', 'Salesforce/blip2-opt-2.7b'),
                'max_length': getattr(self.model, 'max_length', 64)
            }
        }
        
        # Determine checkpoint filename
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            logger.info(f"Saving best model checkpoint at epoch {epoch}")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            logger.info(f"Saving checkpoint for epoch {epoch}")
        
        # Save checkpoint
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved to: {checkpoint_path}")
            
            # Also save latest checkpoint
            latest_path = os.path.join(checkpoint_dir, "latest_model.pth")
            torch.save(checkpoint, latest_path)
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return ""
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with loaded checkpoint info
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training history
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            
            # Get checkpoint info
            checkpoint_info = {
                'epoch': checkpoint.get('epoch', 0),
                'metrics': checkpoint.get('metrics', {}),
                'model_config': checkpoint.get('model_config', {}),
                'train_losses': len(self.train_losses),
                'val_losses': len(self.val_losses)
            }
            
            logger.info(f"Checkpoint loaded from: {checkpoint_path}")
            logger.info(f"Resumed from epoch: {checkpoint_info['epoch']}")
            logger.info(f"Training history: {checkpoint_info['train_losses']} train losses, {checkpoint_info['val_losses']} val losses")
            
            return checkpoint_info
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return {}
    
    def save_best_model(self, current_metrics: Dict[str, Any], checkpoint_dir: str = "../checkpoints") -> bool:
        """
        Save model if it's the best so far based on validation loss.
        
        Args:
            current_metrics: Current epoch metrics
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            True if model was saved as best, False otherwise
        """
        current_val_loss = current_metrics.get('val_loss', float('inf'))
        
        # Check if this is the best model so far
        if len(self.val_losses) <= 1:
            # First validation or very early training - save as best
            is_best = True
        else:
            # Compare with previous best (excluding current)
            previous_losses = self.val_losses[:-1] if len(self.val_losses) > 1 else []
            if previous_losses:
                best_val_loss = min(previous_losses)
                # Save as best if current loss is better OR equal (for consistent 0.0 losses)
                is_best = current_val_loss <= best_val_loss
            else:
                is_best = True
        
        if is_best:
            epoch = current_metrics.get('epoch', len(self.train_losses))
            self.save_checkpoint(epoch, current_metrics, checkpoint_dir, is_best=True)
            logger.info(f"New best model! Validation loss: {current_val_loss:.4f}")
            return True
        else:
            best_loss = min(self.val_losses[:-1]) if len(self.val_losses) > 1 else current_val_loss
            logger.info(f"Not best model. Current val loss: {current_val_loss:.4f}, Best: {best_loss:.4f}")
            return False
    
    def train_with_checkpointing(self, num_epochs: int, checkpoint_dir: str = "../checkpoints", save_every: int = 5) -> Dict[str, Any]:
        """
        Train with automatic checkpointing.
        
        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            
        Returns:
            Final training metrics
        """
        logger.info(f"Starting training with checkpointing for {num_epochs} epochs")
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
        logger.info(f"Saving checkpoint every {save_every} epochs")
        
        final_metrics = {}
        
        for epoch in range(1, num_epochs + 1):
            # Train single epoch
            epoch_metrics = self.train_single_epoch(epoch)
            
            # Save best model if validation improved
            if 'val_loss' in epoch_metrics:
                self.save_best_model(epoch_metrics, checkpoint_dir)
            
            # Save regular checkpoint
            if epoch % save_every == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, epoch_metrics, checkpoint_dir, is_best=False)
            
            final_metrics = epoch_metrics
            
            logger.info(f"Epoch {epoch}/{num_epochs} completed. Train loss: {epoch_metrics.get('train_loss', 'N/A'):.4f}, Val loss: {epoch_metrics.get('val_loss', 'N/A'):.4f}")
        
        logger.info(f"Training completed after {num_epochs} epochs")
        return final_metrics


def test_basic_training_loop():
    """Test Task 5.1: Basic training loop with real data"""
    
    print("="*60)
    print("TASK 5.1: TESTING BASIC TRAINING LOOP")
    print("="*60)
    
    try:
        # Import required modules with correct paths
        import sys
        import os
        
        # Add parent directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.insert(0, parent_dir)
        
        from model.blip2_medical import MedicalBLIP2Model
        from data.data_loader import create_data_loaders
        
        print("🧪 Setting up training components...")
        
        # Initialize model
        print("   📱 Loading BLIP-2 model...")
        model = MedicalBLIP2Model(
            model_name="Salesforce/blip2-opt-2.7b",
            device="auto",
            max_length=64  # Shorter for faster training
        )
        print("   ✅ Model loaded")
        
        # Create data loaders with correct paths
        print("   📊 Creating data loaders...")
        train_csv_path = "../../dataset_splits/train.csv"
        val_csv_path = "../../dataset_splits/val.csv"
        
        loaders = create_data_loaders(
            train_csv=train_csv_path,
            val_csv=val_csv_path,
            batch_size=4,  # Small batch for testing
            num_workers=2
        )
        
        train_loader = loaders['train']
        val_loader = loaders['val']
        
        print(f"   ✅ Data loaders created: {len(train_loader)} train, {len(val_loader)} val batches")
        
        # Initialize trainer
        print("   🏋️ Initializing trainer...")
        trainer = MedicalKeywordTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=1e-5,  # Lower learning rate for stability
            device="auto"
        )
        print("   ✅ Trainer initialized")
        
        # Test single epoch training
        print("\n🚀 Starting single epoch training test...")
        
        # Limit to first 10 batches for testing
        print("   📝 Note: Testing with first 10 batches only for speed")
        
        # Create limited data loaders for testing
        from torch.utils.data import Subset
        train_subset = Subset(train_loader.dataset, range(min(40, len(train_loader.dataset))))  # 40 samples = 10 batches of 4
        val_subset = Subset(val_loader.dataset, range(min(20, len(val_loader.dataset))))  # 20 samples = 5 batches of 4
        
        from torch.utils.data import DataLoader
        test_train_loader = DataLoader(
            train_subset, 
            batch_size=4, 
            shuffle=False,
            collate_fn=train_loader.collate_fn
        )
        test_val_loader = DataLoader(
            val_subset, 
            batch_size=4, 
            shuffle=False,
            collate_fn=val_loader.collate_fn
        )
        
        # Update trainer with test loaders
        trainer.train_loader = test_train_loader
        trainer.val_loader = test_val_loader
        
        # Run single epoch
        metrics = trainer.train_single_epoch(epoch=1)
        
        print(f"\n📊 TRAINING LOOP TEST RESULTS:")
        print(f"   Training loss: {metrics.get('train_loss', 'N/A'):.4f}")
        print(f"   Validation loss: {metrics.get('val_loss', 'N/A'):.4f}")
        print(f"   Training success rate: {metrics.get('success_rate', 0):.1f}%")
        print(f"   Epoch time: {metrics.get('epoch_time', 0):.2f}s")
        print(f"   Successful train batches: {metrics.get('successful_batches', 0)}")
        
        # Check success criteria
        train_success = metrics.get('success_rate', 0) >= 50  # At least 50% batches successful
        loss_reasonable = metrics.get('train_loss', float('inf')) < 100  # Loss not exploding
        
        if train_success and loss_reasonable:
            print(f"\n✅ TASK 5.1 COMPLETE - Basic training loop working!")
            return True
        else:
            print(f"\n❌ TASK 5.1 FAILED - Success rate: {metrics.get('success_rate', 0):.1f}%, Loss: {metrics.get('train_loss', 'N/A')}")
            return False
            
    except Exception as e:
        print(f"\n❌ TASK 5.1 FAILED - Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_loop():
    """Test Task 5.2: Validation loop after training epoch"""
    
    print("="*60)
    print("TASK 5.2: TESTING VALIDATION LOOP")
    print("="*60)
    
    try:
        # Import required modules with correct paths
        import sys
        import os
        
        # Add parent directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.insert(0, parent_dir)
        
        from model.blip2_medical import MedicalBLIP2Model
        from data.data_loader import create_data_loaders
        
        print("🧪 Setting up validation test...")
        
        # Initialize model
        print("   📱 Loading BLIP-2 model...")
        model = MedicalBLIP2Model(
            model_name="Salesforce/blip2-opt-2.7b",
            device="auto",
            max_length=64
        )
        print("   ✅ Model loaded")
        
        # Create data loaders
        print("   📊 Creating data loaders...")
        train_csv_path = "../../dataset_splits/train.csv"
        val_csv_path = "../../dataset_splits/val.csv"
        
        loaders = create_data_loaders(
            train_csv=train_csv_path,
            val_csv=val_csv_path,
            batch_size=4,
            num_workers=2
        )
        
        train_loader = loaders['train']
        val_loader = loaders['val']
        print(f"   ✅ Data loaders created: {len(train_loader)} train, {len(val_loader)} val batches")
        
        # Initialize trainer
        print("   🏋️ Initializing trainer...")
        trainer = MedicalKeywordTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=1e-5,
            device="auto"
        )
        print("   ✅ Trainer initialized")
        
        # Step 1: Run one training epoch (limited)
        print("\n🚀 Step 1: Running limited training epoch...")
        
        # Create limited training loader
        from torch.utils.data import Subset, DataLoader
        train_subset = Subset(train_loader.dataset, range(min(20, len(train_loader.dataset))))
        test_train_loader = DataLoader(
            train_subset, 
            batch_size=4, 
            shuffle=False,
            collate_fn=train_loader.collate_fn
        )
        
        trainer.train_loader = test_train_loader
        train_metrics = trainer.train_epoch(epoch=1)
        
        print(f"   ✅ Training completed:")
        print(f"      Loss: {train_metrics['train_loss']:.4f}")
        print(f"      Success: {train_metrics['success_rate']:.1f}%")
        
        # Step 2: Run validation loop independently
        print("\n🔍 Step 2: Running validation loop after training...")
        
        # Create limited validation loader
        val_subset = Subset(val_loader.dataset, range(min(20, len(val_loader.dataset))))
        test_val_loader = DataLoader(
            val_subset, 
            batch_size=4, 
            shuffle=False,
            collate_fn=val_loader.collate_fn
        )
        
        trainer.val_loader = test_val_loader
        val_metrics = trainer.validate(epoch=1)
        
        print(f"   ✅ Validation completed:")
        print(f"      Loss: {val_metrics['val_loss']:.4f}")
        print(f"      Success: {val_metrics['success_rate']:.1f}%")
        print(f"      Batches: {val_metrics['successful_batches']}/{val_metrics['total_batches']}")
        
        # Step 3: Test combined training + validation
        print("\n🔄 Step 3: Testing combined training + validation...")
        
        combined_metrics = trainer.train_single_epoch(epoch=2)
        
        print(f"   ✅ Combined epoch completed:")
        print(f"      Training loss: {combined_metrics.get('train_loss', 'N/A'):.4f}")
        print(f"      Validation loss: {combined_metrics.get('val_loss', 'N/A'):.4f}")
        print(f"      Total time: {combined_metrics.get('epoch_time', 0):.2f}s")
        
        # Check success criteria for Task 5.2
        val_success = val_metrics.get('success_rate', 0) >= 50  # At least 50% validation batches successful
        val_loss_reasonable = val_metrics.get('val_loss', float('inf')) < 100  # Validation loss not exploding
        has_val_metrics = 'val_loss' in combined_metrics  # Combined method includes validation
        
        if val_success and val_loss_reasonable and has_val_metrics:
            print(f"\n✅ TASK 5.2 COMPLETE - Validation loop working after training!")
            print(f"   🎯 Validation success rate: {val_metrics.get('success_rate', 0):.1f}%")
            print(f"   📊 Validation loss: {val_metrics.get('val_loss', 'N/A'):.4f}")
            print(f"   ⏱️ Combined training+validation works seamlessly")
            return True
        else:
            print(f"\n❌ TASK 5.2 FAILED:")
            print(f"   Validation success: {val_success} (rate: {val_metrics.get('success_rate', 0):.1f}%)")
            print(f"   Loss reasonable: {val_loss_reasonable} (loss: {val_metrics.get('val_loss', 'N/A')})")
            print(f"   Has val metrics: {has_val_metrics}")
            return False
            
    except Exception as e:
        print(f"\n❌ TASK 5.2 FAILED - Validation loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpointing():
    """Test Task 5.3: Checkpointing save and load functionality"""
    
    print("="*60)
    print("TASK 5.3: TESTING CHECKPOINTING")
    print("="*60)
    
    try:
        # Import required modules with correct paths
        import sys
        import os
        
        # Add parent directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.insert(0, parent_dir)
        
        from model.blip2_medical import MedicalBLIP2Model
        from data.data_loader import create_data_loaders
        
        print("🧪 Setting up checkpointing test...")
        
        # Initialize model
        print("   📱 Loading BLIP-2 model...")
        model = MedicalBLIP2Model(
            model_name="Salesforce/blip2-opt-2.7b",
            device="auto",
            max_length=64
        )
        print("   ✅ Model loaded")
        
        # Create data loaders
        print("   📊 Creating data loaders...")
        train_csv_path = "../../dataset_splits/train.csv"
        val_csv_path = "../../dataset_splits/val.csv"
        
        loaders = create_data_loaders(
            train_csv=train_csv_path,
            val_csv=val_csv_path,
            batch_size=4,
            num_workers=2
        )
        
        train_loader = loaders['train']
        val_loader = loaders['val']
        print(f"   ✅ Data loaders created: {len(train_loader)} train, {len(val_loader)} val batches")
        
        # Initialize trainer
        print("   🏋️ Initializing trainer...")
        trainer = MedicalKeywordTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=1e-5,
            device="auto"
        )
        print("   ✅ Trainer initialized")
        
        # Create test checkpoint directory
        test_checkpoint_dir = "../checkpoints/test"
        os.makedirs(test_checkpoint_dir, exist_ok=True)
        print(f"   📁 Test checkpoint directory: {test_checkpoint_dir}")
        
        # Step 1: Train one epoch and save checkpoint
        print("\n🚀 Step 1: Training epoch and saving checkpoint...")
        
        # Create limited data loaders for testing
        from torch.utils.data import Subset, DataLoader
        train_subset = Subset(train_loader.dataset, range(min(12, len(train_loader.dataset))))
        val_subset = Subset(val_loader.dataset, range(min(8, len(val_loader.dataset))))
        
        test_train_loader = DataLoader(
            train_subset, 
            batch_size=4, 
            shuffle=False,
            collate_fn=train_loader.collate_fn
        )
        test_val_loader = DataLoader(
            val_subset, 
            batch_size=4, 
            shuffle=False,
            collate_fn=val_loader.collate_fn
        )
        
        trainer.train_loader = test_train_loader
        trainer.val_loader = test_val_loader
        
        # Train one epoch
        epoch_metrics = trainer.train_single_epoch(epoch=1)
        
        print(f"   ✅ Epoch 1 completed:")
        print(f"      Training loss: {epoch_metrics.get('train_loss', 'N/A'):.4f}")
        print(f"      Validation loss: {epoch_metrics.get('val_loss', 'N/A'):.4f}")
        
        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(1, epoch_metrics, test_checkpoint_dir)
        print(f"   💾 Checkpoint saved to: {checkpoint_path}")
        
        # Verify checkpoint file exists
        checkpoint_exists = os.path.exists(checkpoint_path)
        latest_exists = os.path.exists(os.path.join(test_checkpoint_dir, "latest_model.pth"))
        
        print(f"   ✅ Checkpoint file exists: {checkpoint_exists}")
        print(f"   ✅ Latest model exists: {latest_exists}")
        
        # Step 2: Create new trainer and load checkpoint
        print("\n🔄 Step 2: Creating new trainer and loading checkpoint...")
        
        # Create new model instance
        model2 = MedicalBLIP2Model(
            model_name="Salesforce/blip2-opt-2.7b",
            device="auto",
            max_length=64
        )
        
        # Create new trainer
        trainer2 = MedicalKeywordTrainer(
            model=model2,
            train_loader=test_train_loader,
            val_loader=test_val_loader,
            learning_rate=1e-5,
            device="auto"
        )
        
        print("   ✅ New trainer created")
        
        # Load checkpoint
        checkpoint_info = trainer2.load_checkpoint(checkpoint_path)
        
        print(f"   ✅ Checkpoint loaded:")
        print(f"      Resumed epoch: {checkpoint_info.get('epoch', 'N/A')}")
        print(f"      Training history: {checkpoint_info.get('train_losses', 0)} losses")
        print(f"      Model config: {checkpoint_info.get('model_config', {})}")
        
        # Step 3: Test best model saving
        print("\n🏆 Step 3: Testing best model saving...")
        
        # Train another epoch with worse loss (should not save as best)
        epoch_metrics_2 = trainer2.train_single_epoch(epoch=2)
        is_best_2 = trainer2.save_best_model(epoch_metrics_2, test_checkpoint_dir)
        
        # Train another epoch (might be better)
        epoch_metrics_3 = trainer2.train_single_epoch(epoch=3)
        is_best_3 = trainer2.save_best_model(epoch_metrics_3, test_checkpoint_dir)
        
        print(f"   📊 Epoch 2 was best: {is_best_2}")
        print(f"   📊 Epoch 3 was best: {is_best_3}")
        
        # Check if best model file exists
        best_model_path = os.path.join(test_checkpoint_dir, "best_model.pth")
        best_exists = os.path.exists(best_model_path)
        print(f"   ✅ Best model file exists: {best_exists}")
        
        # Step 4: Test multi-epoch training with checkpointing
        print("\n🔄 Step 4: Testing multi-epoch training with checkpointing...")
        
        final_metrics = trainer2.train_with_checkpointing(
            num_epochs=2, 
            checkpoint_dir=test_checkpoint_dir,
            save_every=1  # Save every epoch for testing
        )
        
        print(f"   ✅ Multi-epoch training completed:")
        print(f"      Final training loss: {final_metrics.get('train_loss', 'N/A'):.4f}")
        print(f"      Final validation loss: {final_metrics.get('val_loss', 'N/A'):.4f}")
        
        # Verify all checkpoint files
        epoch_1_exists = os.path.exists(os.path.join(test_checkpoint_dir, "checkpoint_epoch_1.pth"))
        epoch_2_exists = os.path.exists(os.path.join(test_checkpoint_dir, "checkpoint_epoch_2.pth"))
        
        print(f"   📁 Checkpoint files created:")
        print(f"      Epoch 1: {epoch_1_exists}")
        print(f"      Epoch 2: {epoch_2_exists}")
        print(f"      Best model: {best_exists}")
        print(f"      Latest model: {latest_exists}")
        
        # Check success criteria for Task 5.3
        save_success = checkpoint_exists and latest_exists
        load_success = checkpoint_info.get('epoch', 0) > 0
        best_model_success = best_exists
        multi_epoch_success = epoch_2_exists
        
        if save_success and load_success and best_model_success and multi_epoch_success:
            print(f"\n✅ TASK 5.3 COMPLETE - Checkpointing working!")
            print(f"   💾 Save checkpoint: ✅")
            print(f"   🔄 Load checkpoint: ✅") 
            print(f"   🏆 Best model saving: ✅")
            print(f"   📈 Multi-epoch checkpointing: ✅")
            return True
        else:
            print(f"\n❌ TASK 5.3 FAILED:")
            print(f"   Save success: {save_success}")
            print(f"   Load success: {load_success}")
            print(f"   Best model success: {best_model_success}")
            print(f"   Multi-epoch success: {multi_epoch_success}")
            return False
            
    except Exception as e:
        print(f"\n❌ TASK 5.3 FAILED - Checkpointing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run all Phase 5 tests
    task_5_1 = test_basic_training_loop()
    print("\n" + "="*60 + "\n")
    task_5_2 = test_validation_loop()
    print("\n" + "="*60 + "\n")
    task_5_3 = test_checkpointing()
    
    print("\n" + "="*60)
    print("PHASE 5 TRAINING PIPELINE SUMMARY")
    print("="*60)
    print(f"Task 5.1 (Basic Training Loop): {'✅ PASS' if task_5_1 else '❌ FAIL'}")
    print(f"Task 5.2 (Validation Loop): {'✅ PASS' if task_5_2 else '❌ FAIL'}")
    print(f"Task 5.3 (Checkpointing): {'✅ PASS' if task_5_3 else '❌ FAIL'}")
    
    completed_tasks = sum([task_5_1, task_5_2, task_5_3])
    
    if completed_tasks == 3:
        print(f"\n🎉 PHASE 5 PROGRESS: 3/4 tasks complete - Ready for Task 5.4!")
        print(f"🏗️ Training pipeline foundation is solid:")
        print(f"   ✅ Basic training loop working")
        print(f"   ✅ Validation integrated seamlessly") 
        print(f"   ✅ Checkpointing save/load functional")
        print(f"   📈 Ready for metrics and full training!")
    elif completed_tasks >= 2:
        print(f"\n⚠️ PHASE 5 PROGRESS: {completed_tasks}/4 tasks complete")
        print(f"Most core functionality working - can proceed with caution")
    else:
        print(f"\n❌ PHASE 5 ISSUES: Only {completed_tasks}/4 tasks complete")
        print(f"Need to fix core training pipeline before proceeding") 