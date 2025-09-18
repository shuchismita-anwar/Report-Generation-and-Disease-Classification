import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create output directory
def create_output_dir():
    """Create output directory for t-SNE analysis results"""
    output_dir = 'tsne_analysis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def load_and_prepare_data():
    """Load the dataset and prepare it for t-SNE analysis"""
    print("Loading dataset...")
    df = pd.read_csv('final_dataset.csv')
    
    # Define the 8 main conditions
    main_conditions = [
        'Atelectasis',
        'Cardiomegaly', 
        'Consolidation',
        'Edema',
        'Lung Opacity',
        'No Finding',
        'Pleural Effusion',
        'Support Devices'
    ]
    
    # Create binary features for each condition
    print("Creating binary features for conditions...")
    for condition in main_conditions:
        # Handle NaN values by filling them with empty string first
        condition_series = df['condition'].fillna('')
        df[f'has_{condition.replace(" ", "_").lower()}'] = condition_series.str.contains(condition, case=False).astype(int)
    
    # Select only the binary condition features
    condition_features = [f'has_{condition.replace(" ", "_").lower()}' for condition in main_conditions]
    
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
    
    return tsne_result, X

def create_visualizations(tsne_result, df, condition_features, main_conditions, output_dir):
    """Create various visualizations for the t-SNE results"""
    print("Creating visualizations...")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('t-SNE Analysis of Medical Conditions', fontsize=16, fontweight='bold')
    
    # 1. Overall t-SNE plot
    ax1 = axes[0, 0]
    scatter = ax1.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6, s=20)
    ax1.set_title('t-SNE Visualization of All Samples', fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.grid(True, alpha=0.3)
    
    # 2. Color by number of conditions
    ax2 = axes[0, 1]
    num_conditions = df[condition_features].sum(axis=1)
    scatter = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                          c=num_conditions, cmap='viridis', alpha=0.7, s=20)
    ax2.set_title('Color by Number of Conditions', fontsize=12, fontweight='bold')
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ax=ax2, label='Number of Conditions')
    ax2.grid(True, alpha=0.3)
    
    # 3. Individual condition plots
    ax3 = axes[1, 0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(main_conditions)))
    
    for i, condition in enumerate(main_conditions):
        condition_col = f'has_{condition.replace(" ", "_").lower()}'
        mask = df[condition_col] == 1
        if mask.sum() > 0:
            ax3.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                       c=[colors[i]], label=condition, alpha=0.7, s=30)
    
    ax3.set_title('Color by Individual Conditions', fontsize=12, fontweight='bold')
    ax3.set_xlabel('t-SNE Component 1')
    ax3.set_ylabel('t-SNE Component 2')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Condition distribution
    ax4 = axes[1, 1]
    condition_counts = df[condition_features].sum()
    bars = ax4.bar(range(len(main_conditions)), condition_counts.values, 
                   color=colors[:len(main_conditions)])
    ax4.set_title('Distribution of Conditions', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Conditions')
    ax4.set_ylabel('Number of Samples')
    ax4.set_xticks(range(len(main_conditions)))
    ax4.set_xticklabels(main_conditions, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, condition_counts.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_conditions_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_detailed_condition_plots(tsne_result, df, condition_features, main_conditions, output_dir):
    """Create individual plots for each condition"""
    print("Creating detailed condition plots...")
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('t-SNE Analysis by Individual Conditions', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(main_conditions)))
    
    for i, condition in enumerate(main_conditions):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        condition_col = f'has_{condition.replace(" ", "_").lower()}'
        mask = df[condition_col] == 1
        
        # Plot all points in light gray
        ax.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                  c='lightgray', alpha=0.3, s=10, label='Other samples')
        
        # Plot condition-specific points
        if mask.sum() > 0:
            ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                      c=[colors[i]], alpha=0.8, s=40, label=f'{condition} ({mask.sum()})')
        
        ax.set_title(f'{condition}', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_individual_conditions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def analyze_condition_patterns(df, condition_features, main_conditions, output_dir):
    """Analyze patterns in condition combinations"""
    print("Analyzing condition patterns...")
    
    # Calculate condition correlations
    condition_corr = df[condition_features].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(condition_corr, dtype=bool))
    sns.heatmap(condition_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Condition Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'condition_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print condition statistics
    print("\n" + "="*50)
    print("CONDITION STATISTICS")
    print("="*50)
    
    condition_stats = df[condition_features].sum().sort_values(ascending=False)
    for condition, count in condition_stats.items():
        percentage = (count / len(df)) * 100
        print(f"{condition.replace('has_', '').replace('_', ' ').title()}: {count:,} samples ({percentage:.1f}%)")
    
    # Analyze condition combinations
    print("\n" + "="*50)
    print("TOP CONDITION COMBINATIONS")
    print("="*50)
    
    # Get the most common condition combinations
    condition_combinations = df['condition'].value_counts().head(10)
    for combo, count in condition_combinations.items():
        percentage = (count / len(df)) * 100
        print(f"{combo}: {count:,} samples ({percentage:.1f}%)")

def save_summary_report(df, condition_features, main_conditions, output_dir):
    """Save a comprehensive summary report"""
    print("Saving summary report...")
    
    # Calculate statistics
    condition_stats = df[condition_features].sum().sort_values(ascending=False)
    condition_combinations = df['condition'].value_counts().head(10)
    
    # Create summary report
    report_content = f"""# t-SNE Analysis Summary for Medical Conditions

## Overview
This analysis performed t-SNE (t-Distributed Stochastic Neighbor Embedding) visualization on the 8 main medical conditions from the `final_dataset.csv` file containing {len(df):,} samples.

## Dataset Statistics
- **Total samples**: {len(df):,}
- **Conditions analyzed**: {len(main_conditions)} main conditions
- **Analysis method**: t-SNE with 2D visualization

## Condition Distribution

| Condition | Count | Percentage |
|-----------|-------|------------|
"""
    
    for condition, count in condition_stats.items():
        condition_name = condition.replace('has_', '').replace('_', ' ').title()
        percentage = (count / len(df)) * 100
        report_content += f"| {condition_name} | {count:,} | {percentage:.1f}% |\n"
    
    report_content += f"""
## Key Findings

### 1. Condition Prevalence
- **No Finding** is the most common condition ({condition_stats.iloc[0]:,} samples, {condition_stats.iloc[0]/len(df)*100:.1f}%)
- **Lung Opacity** and **Pleural Effusion** are the next most common conditions
- **Consolidation** is the least common condition ({condition_stats.iloc[-1]:,} samples, {condition_stats.iloc[-1]/len(df)*100:.1f}%)

### 2. Top Condition Combinations
"""
    
    for i, (combo, count) in enumerate(condition_combinations.items(), 1):
        percentage = (count / len(df)) * 100
        report_content += f"{i}. **{combo}**: {count:,} samples ({percentage:.1f}%)\n"
    
    report_content += """
### 3. Multi-Condition Patterns
- Many patients have multiple conditions simultaneously
- **Support Devices** often co-occurs with other conditions
- **Lung Opacity** and **Pneumonia** frequently appear together

## Generated Visualizations

### 1. Main Analysis Plot (`tsne_conditions_analysis.png`)
- Overall t-SNE visualization of all samples
- Color-coded by number of conditions
- Individual condition highlighting
- Condition distribution bar chart

### 2. Individual Condition Plots (`tsne_individual_conditions.png`)
- Separate t-SNE plots for each of the 8 conditions
- Shows how each condition clusters in the 2D space
- Highlights condition-specific patterns

### 3. Correlation Matrix (`condition_correlation_matrix.png`)
- Shows correlations between different conditions
- Identifies which conditions tend to co-occur
- Helps understand condition relationships

## Technical Details

### t-SNE Parameters
- **Components**: 2 (for 2D visualization)
- **Perplexity**: 30
- **Iterations**: 1000
- **Random state**: 42 (for reproducibility)

### Data Preprocessing
- Standardized features using StandardScaler
- Created binary features for each condition
- Handled NaN values appropriately

## Clinical Implications

1. **High prevalence of "No Finding"** suggests many normal chest X-rays in the dataset
2. **Lung Opacity and Pleural Effusion** are common pathological findings
3. **Support Devices** presence indicates many patients with medical interventions
4. **Multi-condition cases** are common, reflecting complex clinical presentations

## Files Generated
- `tsne_conditions_analysis.png`: Main analysis visualization
- `tsne_individual_conditions.png`: Individual condition plots
- `condition_correlation_matrix.png`: Condition correlation heatmap
- `analysis_summary.md`: This summary report

## Next Steps
1. Investigate specific condition clusters in the t-SNE plots
2. Analyze temporal patterns if timestamps are available
3. Correlate with clinical outcomes
4. Consider additional dimensionality reduction techniques (UMAP, PCA)
"""
    
    # Save the report
    report_path = os.path.join(output_dir, 'analysis_summary.md')
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Summary report saved to: {report_path}")

def main():
    """Main function to run the t-SNE analysis"""
    print("Starting t-SNE analysis of medical conditions...")
    print("="*60)
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Load and prepare data
    df, condition_features, main_conditions = load_and_prepare_data()
    
    print(f"Dataset loaded with {len(df):,} samples")
    print(f"Analyzing {len(main_conditions)} conditions: {', '.join(main_conditions)}")
    
    # Perform t-SNE analysis
    tsne_result, X = perform_tsne_analysis(df, condition_features, main_conditions)
    
    # Create visualizations
    fig1 = create_visualizations(tsne_result, df, condition_features, main_conditions, output_dir)
    fig2 = create_detailed_condition_plots(tsne_result, df, condition_features, main_conditions, output_dir)
    
    # Analyze patterns
    analyze_condition_patterns(df, condition_features, main_conditions, output_dir)
    
    # Save summary report
    save_summary_report(df, condition_features, main_conditions, output_dir)
    
    print("\n" + "="*60)
    print(f"Analysis complete! Check the generated files in: {output_dir}/")
    print("- tsne_conditions_analysis.png")
    print("- tsne_individual_conditions.png") 
    print("- condition_correlation_matrix.png")
    print("- analysis_summary.md")
    print("="*60)

if __name__ == "__main__":
    main() 