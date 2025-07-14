import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuration - Easy to modify for different experiments
EXPERIMENTS = {
    "with_gaze": {
        "name": "With Gaze - Train+Val Data",
        "path": "output/0.((train+val)full+enhanced_gaze)_training_mimic_on_chexpert_optimized",
        "filename_prefix": "with_gaze_train+val_data",
        "color": "#2E86C1",  # Blue
        "line_style": "-"
    },
    "without_gaze": {
        "name": "Without Gaze - Fixation Removed", 
        "path": "output/2.(fixation_removed)_training_mimic_on_chexpert_optimized",
        "filename_prefix": "without_gaze_fixation_removed",
        "color": "#E74C3C",  # Red
        "line_style": "-"
    }
}

# Output directory
OUTPUT_DIR = "graphs"

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")

def find_trainer_state_files(experiment_path):
    """Find all trainer_state.json files in checkpoint folders"""
    checkpoint_pattern = os.path.join(experiment_path, "checkpoint-*", "trainer_state.json")
    trainer_files = glob.glob(checkpoint_pattern)
    
    if not trainer_files:
        print(f"⚠️  No trainer_state.json files found in {experiment_path}")
        return []
    
    # Sort by checkpoint number
    def get_checkpoint_num(path):
        try:
            checkpoint_folder = os.path.basename(os.path.dirname(path))
            return int(checkpoint_folder.split('-')[1])
        except (ValueError, IndexError):
            return 0
    
    trainer_files.sort(key=get_checkpoint_num)
    print(f"✅ Found {len(trainer_files)} trainer_state.json files in {experiment_path}")
    return trainer_files

def load_training_history(trainer_files):
    """Load and combine training history from all trainer_state.json files"""
    all_logs = []
    
    for file_path in trainer_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                log_history = data.get('log_history', [])
                all_logs.extend(log_history)
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
            continue
    
    if not all_logs:
        print("❌ No training logs found!")
        return None, None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_logs)
    
    # Separate training and evaluation logs
    train_logs = df[df['loss'].notna()].copy()
    eval_logs = df[df['eval_loss'].notna()].copy()
    
    # Remove duplicates and sort by step
    train_logs = train_logs.drop_duplicates(subset=['step']).sort_values('step')
    eval_logs = eval_logs.drop_duplicates(subset=['step']).sort_values('step')
    
    print(f"📊 Loaded {len(train_logs)} training steps, {len(eval_logs)} evaluation steps")
    
    return train_logs, eval_logs

def load_test_predictions(experiment_path):
    """Load test metrics from metrics JSON file"""
    metrics_file = os.path.join(experiment_path, "metrics_training_mimic_on_chexpert.json")
    
    if not os.path.exists(metrics_file):
        print(f"⚠️  Metrics file not found: {metrics_file}")
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        test_f1 = metrics.get('eval_f1', None)
        test_auc = metrics.get('eval_auc', None)
        
        print(f"📊 Test Metrics from JSON:")
        print(f"  Test F1: {test_f1:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        
        return test_f1
    except Exception as e:
        print(f"❌ Error loading test metrics: {e}")
        return None

def create_f1_training_testing_history_plot(experiment_key, experiment_config, eval_logs):
    """Create F1 score training and testing history plot"""
    if eval_logs is None or eval_logs.empty:
        print(f"❌ No evaluation data for {experiment_key}")
        return
    
    # Load test F1 score
    test_f1 = load_test_predictions(experiment_config['path'])
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot training F1 score progression (eval_f1 represents training performance)
    plt.plot(eval_logs['step'], eval_logs['eval_f1'], 
             color=experiment_config['color'], 
             linewidth=2.5, 
             marker='o', 
             markersize=4, 
             alpha=0.8,
             label='Training F1 Score')
    
    # Add test F1 as horizontal line if available
    if test_f1 is not None:
        plt.axhline(y=test_f1, color='orange', linestyle='--', linewidth=2.5, 
                    alpha=0.8, label=f'Test F1 Score ({test_f1:.4f})')
    
    # Formatting (no title)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Set y-axis limits for better visualization
    f1_min = eval_logs['eval_f1'].min()
    f1_max = eval_logs['eval_f1'].max()
    if test_f1 is not None:
        f1_min = min(f1_min, test_f1)
        f1_max = max(f1_max, test_f1)
    
    margin = (f1_max - f1_min) * 0.1
    plt.ylim(max(0, f1_min - margin), min(1, f1_max + margin))
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{experiment_config['filename_prefix']}_training_testing_history.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved F1 training/testing history: {filepath}")

def create_loss_plot(experiment_key, experiment_config, train_logs, eval_logs):
    """Create training and validation loss plot"""
    if train_logs is None or eval_logs is None:
        print(f"❌ No training/evaluation data for {experiment_key}")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    if not train_logs.empty:
        plt.plot(train_logs['step'], train_logs['loss'], 
                color=experiment_config['color'], 
                linewidth=2, 
                alpha=0.7,
                label='Training Loss')
    
    # Plot validation loss
    if not eval_logs.empty:
        plt.plot(eval_logs['step'], eval_logs['eval_loss'], 
                color=experiment_config['color'], 
                linewidth=2.5, 
                linestyle='--',
                marker='s', 
                markersize=3,
                label='Validation Loss')
    
    # Formatting (no title)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{experiment_config['filename_prefix']}_loss_plot.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved loss plot: {filepath}")

def process_experiment(experiment_key, experiment_config):
    """Process a single experiment and generate plots"""
    print(f"\n🔬 Processing experiment: {experiment_config['name']}")
    print(f"📂 Path: {experiment_config['path']}")
    
    # Find trainer state files
    trainer_files = find_trainer_state_files(experiment_config['path'])
    if not trainer_files:
        print(f"❌ Skipping {experiment_key} - no trainer files found")
        return
    
    # Load training history
    train_logs, eval_logs = load_training_history(trainer_files)
    if train_logs is None and eval_logs is None:
        print(f"❌ Skipping {experiment_key} - no valid data found")
        return
    
    # Create plots
    create_f1_training_testing_history_plot(experiment_key, experiment_config, eval_logs)
    create_loss_plot(experiment_key, experiment_config, train_logs, eval_logs)

def main():
    """Main function to generate all plots"""
    print("🎨 Starting graph generation for training experiments")
    print("=" * 60)
    
    # Ensure output directory exists
    ensure_output_dir()
    
    # Process each experiment
    for experiment_key, experiment_config in EXPERIMENTS.items():
        process_experiment(experiment_key, experiment_config)
    
    print("\n" + "=" * 60)
    print("🎉 Graph generation complete!")
    print(f"📁 All plots saved to: {OUTPUT_DIR}/")
    
    # List generated files
    generated_files = glob.glob(os.path.join(OUTPUT_DIR, "*.png"))
    if generated_files:
        print("\n📊 Generated files:")
        for file in sorted(generated_files):
            print(f"   - {os.path.basename(file)}")

if __name__ == "__main__":
    main()
