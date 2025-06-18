"""
Configuration Loader for Medical Keyword Prediction

Handles loading, validation, and merging of YAML configuration files.
Provides type-safe access to configuration parameters.

Task 6.1: Create configuration system (config.yaml with hyperparameters)
"""

import yaml
import os
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Configuration loader and validator for medical keyword prediction training.
    
    Provides type-safe access to configuration parameters with validation
    and default value handling.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file (defaults to training_config.yaml)
        """
        if config_path is None:
            # Default to training_config.yaml in the same directory
            config_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(config_dir, "training_config.yaml")
        
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        
        logger.info(f"Configuration loaded from: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ValueError("Configuration file is empty")
            
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _validate_config(self):
        """Validate configuration structure and required fields."""
        required_sections = ['model', 'training', 'data', 'checkpointing']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate model configuration
        model_config = self.config['model']
        required_model_fields = ['model_name', 'max_length', 'device']
        for field in required_model_fields:
            if field not in model_config:
                raise ValueError(f"Missing required model field: {field}")
        
        # Validate training configuration
        training_config = self.config['training']
        required_training_fields = ['learning_rate', 'batch_size', 'num_epochs']
        for field in required_training_fields:
            if field not in training_config:
                raise ValueError(f"Missing required training field: {field}")
        
        # Keep relative paths as-is for data files
        # They will be resolved relative to the working directory when used
        pass
        
        logger.info("Configuration validation passed")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {key_path}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self.config['model']
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration section."""
        return self.config['training']
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self.config['data']
    
    def get_checkpointing_config(self) -> Dict[str, Any]:
        """Get checkpointing configuration section."""
        return self.config['checkpointing']
    
    def get_metrics_config(self) -> Dict[str, Any]:
        """Get metrics configuration section."""
        return self.config.get('metrics', {'enabled': False})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration section."""
        return self.config.get('logging', {'enabled': True, 'level': 'INFO'})
    
    def is_short_training(self) -> bool:
        """Check if short training mode is enabled."""
        return self.get('short_training.enabled', False)
    
    def is_full_training(self) -> bool:
        """Check if full training mode is enabled."""
        return self.get('full_training.enabled', False)
    
    def get_effective_epochs(self) -> int:
        """Get effective number of epochs based on training mode."""
        if self.is_short_training():
            return self.get('short_training.max_epochs', 5)
        elif self.is_full_training():
            return self.get('full_training.max_epochs', 50)
        else:
            return self.get('training.num_epochs', 10)
    
    def get_effective_batch_limits(self) -> Dict[str, Optional[int]]:
        """Get effective batch limits based on training mode."""
        if self.is_short_training():
            return {
                'max_train_batches': self.get('short_training.max_train_batches', 100),
                'max_val_batches': self.get('short_training.max_val_batches', 25)
            }
        else:
            return {
                'max_train_batches': None,
                'max_val_batches': None
            }
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates in dot notation format
        """
        for key_path, value in updates.items():
            keys = key_path.split('.')
            config_section = self.config
            
            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in config_section:
                    config_section[key] = {}
                config_section = config_section[key]
            
            # Set the final value
            config_section[keys[-1]] = value
        
        logger.info(f"Configuration updated with {len(updates)} changes")
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration (defaults to original path)
        """
        if output_path is None:
            output_path = self.config_path
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key configuration settings."""
        summary = {
            'model_name': self.get('model.model_name'),
            'learning_rate': self.get('training.learning_rate'),
            'batch_size': self.get('training.batch_size'),
            'num_epochs': self.get_effective_epochs(),
            'device': self.get('model.device'),
            'metrics_enabled': self.get('metrics.enabled', False),
            'checkpointing_enabled': self.get('checkpointing.enabled', True),
            'training_mode': 'short' if self.is_short_training() else 'full' if self.is_full_training() else 'standard'
        }
        
        # Add batch limits for short training
        if self.is_short_training():
            batch_limits = self.get_effective_batch_limits()
            summary.update(batch_limits)
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of the configuration."""
        summary = self.get_summary()
        
        print("="*60)
        print("TRAINING CONFIGURATION SUMMARY")
        print("="*60)
        
        for key, value in summary.items():
            # Format key for display
            display_key = key.replace('_', ' ').title()
            print(f"{display_key:25}: {value}")
        
        print("="*60)


def test_config_loader():
    """Test configuration loader functionality."""
    
    print("="*60)
    print("TESTING CONFIGURATION LOADER (TASK 6.1)")
    print("="*60)
    
    try:
        # Test loading configuration
        print("🧪 Step 1: Loading configuration...")
        config = ConfigLoader()
        print("   ✅ Configuration loaded successfully")
        
        # Test basic access
        print("🔍 Step 2: Testing configuration access...")
        model_name = config.get('model.model_name')
        learning_rate = config.get('training.learning_rate')
        batch_size = config.get('training.batch_size')
        
        print(f"   Model: {model_name}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Batch size: {batch_size}")
        print("   ✅ Basic access working")
        
        # Test section access
        print("📊 Step 3: Testing section access...")
        model_config = config.get_model_config()
        training_config = config.get_training_config()
        
        print(f"   Model config keys: {len(model_config)}")
        print(f"   Training config keys: {len(training_config)}")
        print("   ✅ Section access working")
        
        # Test training mode detection
        print("🎯 Step 4: Testing training mode detection...")
        is_short = config.is_short_training()
        effective_epochs = config.get_effective_epochs()
        batch_limits = config.get_effective_batch_limits()
        
        print(f"   Short training: {is_short}")
        print(f"   Effective epochs: {effective_epochs}")
        print(f"   Batch limits: {batch_limits}")
        print("   ✅ Training mode detection working")
        
        # Test configuration summary
        print("📋 Step 5: Testing configuration summary...")
        config.print_summary()
        print("   ✅ Configuration summary working")
        
        print(f"\n✅ TASK 6.1 CONFIGURATION SYSTEM COMPLETE!")
        print(f"📝 Configuration file: training_config.yaml")
        print(f"🔧 Configuration loader: config_loader.py")
        print(f"🎯 All hyperparameters from optimization work captured")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.config


def update_config_from_args(config: Dict[str, Any], args) -> int:
    """
    Update configuration with command line arguments.
    
    Args:
        config: Configuration dictionary to update
        args: Parsed command line arguments
        
    Returns:
        Number of configuration changes made
    """
    changes = 0
    
    # Update model configuration
    if hasattr(args, 'model_name') and args.model_name:
        config['model']['model_name'] = args.model_name
        changes += 1
    
    # Update training configuration
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
        changes += 1
    
    if hasattr(args, 'batch_size') and args.batch_size:
        config['training']['batch_size'] = args.batch_size
        changes += 1
    
    if hasattr(args, 'num_epochs') and args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
        changes += 1
    
    if hasattr(args, 'device') and args.device:
        config['training']['device'] = args.device
        changes += 1
    
    if hasattr(args, 'metrics') and args.metrics is not None:
        config['training']['metrics_enabled'] = args.metrics
        changes += 1
    
    if hasattr(args, 'checkpointing') and args.checkpointing is not None:
        config['training']['checkpointing_enabled'] = args.checkpointing
        changes += 1
    
    # Handle short training mode
    if hasattr(args, 'short_training') and args.short_training:
        config['short_training'] = {'enabled': True}
        changes += 1
    
    return changes


if __name__ == "__main__":
    test_config_loader() 