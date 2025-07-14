import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

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

def create_f1_training_history_plot(experiment_key, experiment_config, eval_logs):
    """Create F1 score training history plot"""
    if eval_logs is None or eval_logs.empty:
        print(f"❌ No evaluation data for {experiment_key}")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot F1 score progression
    plt.plot(eval_logs['step'], eval_logs['eval_f1'], 
             color=experiment_config['color'], 
             linewidth=2.5, 
             marker='o', 
             markersize=4, 
             alpha=0.8,
             label='F1 Score')
    
    # Add baseline reference line (if applicable)
    baseline_f1 = 0.6242  # ViT-only baseline mentioned in the code
    plt.axhline(y=baseline_f1, color='gray', linestyle='--', alpha=0.7, 
                label=f'Baseline F1 ({baseline_f1})')
    
    # Formatting
    plt.title(f'{experiment_config["name"]} - Training History', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add statistics text box
    if not eval_logs['eval_f1'].empty:
        final_f1 = eval_logs['eval_f1'].iloc[-1]
        best_f1 = eval_logs['eval_f1'].max()
        best_step = eval_logs.loc[eval_logs['eval_f1'].idxmax(), 'step']
        
        stats_text = f'Final F1: {final_f1:.4f}\nBest F1: {best_f1:.4f} (Step {best_step})'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set y-axis limits for better visualization
    f1_min = eval_logs['eval_f1'].min()
    f1_max = eval_logs['eval_f1'].max()
    margin = (f1_max - f1_min) * 0.1
    plt.ylim(max(0, f1_min - margin), min(1, f1_max + margin))
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{experiment_config['filename_prefix']}_training_history.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved F1 training history: {filepath}")

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
    
    # Formatting
    plt.title(f'{experiment_config["name"]} - Loss Plot', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add statistics text box
    stats_text = ""
    if not train_logs.empty:
        final_train_loss = train_logs['loss'].iloc[-1]
        stats_text += f'Final Train Loss: {final_train_loss:.4f}\n'
    
    if not eval_logs.empty:
        final_val_loss = eval_logs['eval_loss'].iloc[-1]
        best_val_loss = eval_logs['eval_loss'].min()
        best_step = eval_logs.loc[eval_logs['eval_loss'].idxmin(), 'step']
        stats_text += f'Final Val Loss: {final_val_loss:.4f}\n'
        stats_text += f'Best Val Loss: {best_val_loss:.4f} (Step {best_step})'
    
    if stats_text:
        plt.text(0.02, 0.98, stats_text.strip(), transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
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
    create_f1_training_history_plot(experiment_key, experiment_config, eval_logs)
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
