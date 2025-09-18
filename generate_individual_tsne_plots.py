"""
Generate Individual t-SNE Plots for Each Medical Condition
========================================================

This script creates separate t-SNE visualizations for each of the 8 medical conditions,
showing how each condition clusters in the 2D t-SNE space.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def create_output_dir():
    """Create output directory for individual t-SNE plots"""
    output_dir = "tsne_individual_plots"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_and_prepare_data():
    """Load and prepare the dataset for t-SNE analysis"""
    print("Loading dataset...")
    
    # Load the dataset
    df = pd.read_csv("final_dataset.csv")
    print(f"Loaded {len(df):,} samples")
    
    # Define the main conditions
    main_conditions = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Support Devices'
    ]
    
    # Create binary features for each condition
    for condition in main_conditions:
        condition_col = f'has_{condition.replace(" ", "_").lower()}'
        df[condition_col] = df['condition'].str.contains(condition, case=False, na=False).astype(int)
    
    # Get condition feature columns
    condition_features = [f'has_{condition.replace(" ", "_").lower()}' 
                        for condition in main_conditions]
    
    return df, condition_features, main_conditions

def perform_tsne_analysis(df, condition_features, main_conditions):
    """Perform t-SNE analysis on the condition data"""
    print("Performing t-SNE analysis...")
    
    # Prepare the data for t-SNE
    X = df[condition_features].values
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA first to reduce dimensionality if needed
    if X_scaled.shape[1] > 50:
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X_scaled)
        print(f"Applied PCA to reduce dimensions from {X_scaled.shape[1]} to {X_pca.shape[1]}")
        X_for_tsne = X_pca
    else:
        X_for_tsne = X_scaled
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(X_for_tsne)
    
    print("t-SNE analysis completed!")
    return tsne_result, X

def create_individual_condition_plot(tsne_result, df, condition, condition_col, output_dir, color_map):
    """Create individual t-SNE plot for a specific condition"""
    
    # Create figure with larger size for individual plots
    plt.figure(figsize=(12, 10))
    
    # Get mask for this condition
    mask = df[condition_col] == 1
    condition_count = mask.sum()
    
    # Plot all points in light gray as background
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                c='lightgray', alpha=0.3, s=15, label=f'Other samples ({len(df) - condition_count})')
    
    # Plot condition-specific points
    if condition_count > 0:
        plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                   c=color_map[condition], alpha=0.8, s=60, 
                   label=f'{condition} ({condition_count})')
    
    # Customize the plot
    plt.title(f't-SNE Analysis: {condition}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    percentage = (condition_count / len(df)) * 100
    stats_text = f'Total samples: {len(df):,}\n{condition}: {condition_count:,} ({percentage:.1f}%)'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set axis limits to match the original plot
    plt.xlim(-200, 150)
    plt.ylim(-200, 150)
    
    # Save the individual plot
    filename = f'tsne_{condition.replace(" ", "_").lower()}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {filename}")
    return filepath

def generate_all_individual_plots():
    """Generate individual t-SNE plots for each condition"""
    print("Generating individual t-SNE plots for each condition...")
    print("="*60)
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Load and prepare data
    df, condition_features, main_conditions = load_and_prepare_data()
    
    print(f"Dataset loaded with {len(df):,} samples")
    print(f"Generating plots for {len(main_conditions)} conditions")
    
    # Perform t-SNE analysis
    tsne_result, X = perform_tsne_analysis(df, condition_features, main_conditions)
    
    # Define colors for each condition (matching the original plot)
    color_map = {
        'Atelectasis': '#1f77b4',      # Blue
        'Cardiomegaly': '#ff7f0e',     # Orange
        'Consolidation': '#d62728',     # Red
        'Edema': '#2ca02c',            # Green
        'Lung Opacity': '#9467bd',     # Purple
        'No Finding': '#8c564b',       # Brown
        'Pleural Effusion': '#e377c2', # Pink
        'Support Devices': '#7f7f7f'   # Gray
    }
    
    # Generate individual plots for each condition
    generated_files = []
    for condition in main_conditions:
        condition_col = f'has_{condition.replace(" ", "_").lower()}'
        filepath = create_individual_condition_plot(
            tsne_result, df, condition, condition_col, output_dir, color_map
        )
        generated_files.append(filepath)
    
    # Create a summary file listing all generated plots
    summary_content = f"""# Individual t-SNE Plots Generated

## Overview
Generated individual t-SNE visualizations for each of the {len(main_conditions)} medical conditions.

## Generated Files

"""
    
    for i, condition in enumerate(main_conditions):
        filename = f'tsne_{condition.replace(" ", "_").lower()}.png'
        condition_count = df[f'has_{condition.replace(" ", "_").lower()}'].sum()
        percentage = (condition_count / len(df)) * 100
        summary_content += f"{i+1}. **{condition}**: `{filename}` ({condition_count:,} samples, {percentage:.1f}%)\n"
    
    summary_content += f"""
## Technical Details

- **Total samples**: {len(df):,}
- **t-SNE parameters**: perplexity=30, n_iter=1000, random_state=42
- **Color scheme**: Each condition has a distinct color for easy identification
- **Background**: All other samples shown in light gray for context

## Usage

Each plot shows:
- Light gray dots: All other samples (for context)
- Colored dots: Samples with the specific condition
- Statistics: Sample counts and percentages
- Grid: For better spatial reference

## Files Location
All files are saved in: `{output_dir}/`
"""
    
    # Save summary
    summary_path = os.path.join(output_dir, "README.md")
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    
    print("\n" + "="*60)
    print("INDIVIDUAL t-SNE PLOTS GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"Generated {len(generated_files)} individual plots:")
    for filepath in generated_files:
        print(f"  ✓ {os.path.basename(filepath)}")
    print(f"\nAll files saved in: {output_dir}/")
    print(f"Summary: {summary_path}")
    print("="*60)
    
    return generated_files

if __name__ == "__main__":
    generate_all_individual_plots() 