#!/usr/bin/env python3
"""
Keyword Counter for Medical Conditions
Counts keywords per condition across different confidence levels
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def load_keywords_data(file_path):
    """Load the cleaned keywords data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return None

def count_keywords_by_condition(data):
    """Count keywords for each condition by confidence level"""
    if not data or 'conditions' not in data:
        print("Error: Invalid data structure")
        return None
    
    conditions = data['conditions']
    keyword_counts = {}
    
    # Confidence levels to check
    confidence_levels = ['high_confidence', 'medium_confidence', 'low_confidence']
    
    for condition_name, condition_data in conditions.items():
        keyword_counts[condition_name] = {}
        
        for confidence_level in confidence_levels:
            if confidence_level in condition_data:
                count = len(condition_data[confidence_level])
                keyword_counts[condition_name][confidence_level] = count
            else:
                keyword_counts[condition_name][confidence_level] = 0
        
        # Calculate total keywords for this condition
        total_keywords = sum(keyword_counts[condition_name].values())
        keyword_counts[condition_name]['total'] = total_keywords
    
    return keyword_counts

def create_summary_dataframe(keyword_counts):
    """Create a pandas DataFrame for easy analysis"""
    if not keyword_counts:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(keyword_counts, orient='index')
    
    # Reset index to make condition names a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'condition'}, inplace=True)
    
    # Sort by total keywords (descending)
    df = df.sort_values('total', ascending=False)
    
    return df

def print_detailed_summary(keyword_counts, data):
    """Print detailed summary of keyword counts"""
    print("="*80)
    print("MEDICAL CONDITION KEYWORD COUNT SUMMARY")
    print("="*80)
    
    # Print metadata
    if 'metadata' in data:
        metadata = data['metadata']
        print(f"Data Version: {metadata.get('version', 'Unknown')}")
        print(f"Total Patients Analyzed: {metadata.get('total_patients_analyzed', 'Unknown')}")
        print(f"Successful Extractions: {metadata.get('successful_extractions', 'Unknown')}")
        if 'ai_cleanup' in metadata:
            cleanup = metadata['ai_cleanup']
            print(f"Total Keywords Processed: {cleanup.get('total_keywords_processed', 'Unknown')}")
            print(f"Total Keywords Kept: {cleanup.get('total_keywords_kept', 'Unknown')}")
            print(f"Removal Rate: {cleanup.get('removal_rate', 'Unknown')}")
        print("-"*80)
    
    # Print condition-wise counts
    total_all_keywords = 0
    total_high_conf = 0
    total_medium_conf = 0
    total_low_conf = 0
    
    print(f"{'Condition':<20} {'High':<8} {'Medium':<8} {'Low':<8} {'Total':<8}")
    print("-"*60)
    
    for condition, counts in keyword_counts.items():
        high = counts['high_confidence']
        medium = counts['medium_confidence']
        low = counts['low_confidence']
        total = counts['total']
        
        print(f"{condition:<20} {high:<8} {medium:<8} {low:<8} {total:<8}")
        
        total_all_keywords += total
        total_high_conf += high
        total_medium_conf += medium
        total_low_conf += low
    
    print("-"*60)
    print(f"{'TOTAL':<20} {total_high_conf:<8} {total_medium_conf:<8} {total_low_conf:<8} {total_all_keywords:<8}")
    print("="*80)

def create_visualizations(df, output_dir):
    """Create visualizations for keyword counts"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Bar chart of total keywords per condition
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['condition'], df['total'])
    plt.title('Total Keywords per Medical Condition', fontsize=16, fontweight='bold')
    plt.xlabel('Medical Condition', fontsize=12)
    plt.ylabel('Number of Keywords', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'total_keywords_per_condition.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Stacked bar chart by confidence levels
    plt.figure(figsize=(14, 8))
    
    # Create stacked bar chart
    x_pos = range(len(df))
    
    plt.bar(x_pos, df['high_confidence'], label='High Confidence', alpha=0.8)
    plt.bar(x_pos, df['medium_confidence'], bottom=df['high_confidence'], 
            label='Medium Confidence', alpha=0.8)
    plt.bar(x_pos, df['low_confidence'], 
            bottom=df['high_confidence'] + df['medium_confidence'], 
            label='Low Confidence', alpha=0.8)
    
    plt.title('Keywords by Confidence Level per Medical Condition', fontsize=16, fontweight='bold')
    plt.xlabel('Medical Condition', fontsize=12)
    plt.ylabel('Number of Keywords', fontsize=12)
    plt.xticks(x_pos, df['condition'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'keywords_by_confidence_level.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Pie chart of total keywords distribution
    plt.figure(figsize=(10, 8))
    plt.pie(df['total'], labels=df['condition'], autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Keywords Across Medical Conditions', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_dir / 'keywords_distribution_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap of confidence levels
    plt.figure(figsize=(10, 8))
    
    # Prepare data for heatmap
    heatmap_data = df[['high_confidence', 'medium_confidence', 'low_confidence']].T
    heatmap_data.columns = df['condition']
    
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Number of Keywords'})
    plt.title('Keyword Count Heatmap by Confidence Level', fontsize=16, fontweight='bold')
    plt.xlabel('Medical Condition', fontsize=12)
    plt.ylabel('Confidence Level', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'keywords_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

def save_results(df, keyword_counts, output_dir):
    """Save results to CSV and JSON files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save DataFrame to CSV
    csv_path = output_dir / 'keyword_counts_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}")
    
    # Save detailed counts to JSON
    json_path = output_dir / 'keyword_counts_detailed.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(keyword_counts, f, indent=2, ensure_ascii=False)
    print(f"Detailed counts saved to: {json_path}")

def main():
    """Main function to run the keyword counting analysis"""
    # File paths
    input_file = Path(__file__).parent / 'extracted_keywords_result_cleaned.json'
    output_dir = Path(__file__).parent / 'keyword_count_results'
    
    print("Loading keywords data...")
    data = load_keywords_data(input_file)
    
    if data is None:
        return
    
    print("Counting keywords by condition and confidence level...")
    keyword_counts = count_keywords_by_condition(data)
    
    if keyword_counts is None:
        return
    
    print("Creating summary DataFrame...")
    df = create_summary_dataframe(keyword_counts)
    
    if df is None:
        return
    
    # Print detailed summary
    print_detailed_summary(keyword_counts, data)
    
    # Save results
    print("\nSaving results...")
    save_results(df, keyword_counts, output_dir)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
