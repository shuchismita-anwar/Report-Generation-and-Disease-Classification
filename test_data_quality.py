#!/usr/bin/env python3
"""
Simple Data Quality Test for Task 5.2
Tests core data quality analysis functionality
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def test_data_quality_analysis():
    """Test data quality analysis functionality"""
    print("🔬 Task 5.2: Data Quality Analysis Test")
    print("=" * 50)
    
    # Test 1: Dataset Loading
    print("\n📊 Test 1: Dataset Loading")
    dataset_splits_dir = Path("../../dataset_splits")
    
    splits_info = {}
    for split in ['train', 'val', 'test']:
        split_path = dataset_splits_dir / f"{split}.csv"
        if split_path.exists():
            try:
                df = pd.read_csv(split_path)
                splits_info[split] = {
                    'total_samples': len(df),
                    'columns': list(df.columns),
                    'file_loaded': True
                }
                print(f"  ✅ {split}: {len(df)} samples, {len(df.columns)} columns")
            except Exception as e:
                splits_info[split] = {'file_loaded': False, 'error': str(e)}
                print(f"  ❌ {split}: Error - {e}")
        else:
            splits_info[split] = {'file_loaded': False, 'error': 'File not found'}
            print(f"  ⚠️  {split}: File not found")
    
    # Test 2: Sample Data Analysis
    print("\n📝 Test 2: Sample Data Structure Analysis")
    if splits_info.get('train', {}).get('file_loaded', False):
        train_df = pd.read_csv(dataset_splits_dir / "train.csv")
        sample_row = train_df.iloc[0]
        
        print("  📋 Sample data structure:")
        key_columns = ['patient_folder', 'condition', 'image_path', 'transcript_path', 'bbox_path']
        for col in key_columns:
            if col in sample_row:
                value = str(sample_row[col])[:100]  # Truncate for display
                print(f"    • {col}: {value}")
        
        # Analyze conditions
        if 'condition' in train_df.columns:
            conditions = train_df['condition'].dropna()
            unique_conditions = set()
            for condition_str in conditions:
                if '|' in str(condition_str):
                    unique_conditions.update(str(condition_str).split('|'))
                else:
                    unique_conditions.add(str(condition_str))
            
            print(f"  🏥 Medical conditions found: {len(unique_conditions)} unique")
            print(f"    • Sample conditions: {list(unique_conditions)[:5]}")
    
    # Test 3: File Existence Check (Small Sample)
    print("\n🔍 Test 3: File Existence Check (Sample of 5)")
    if splits_info.get('train', {}).get('file_loaded', False):
        train_df = pd.read_csv(dataset_splits_dir / "train.csv")
        sample_df = train_df.head(5)  # Small sample for testing
        
        file_types = ['image_path', 'transcript_path', 'bbox_path']
        existence_stats = {ft: {'exists': 0, 'missing': 0} for ft in file_types}
        
        base_dir = Path("../../")
        for idx, row in sample_df.iterrows():
            for file_type in file_types:
                if file_type in row and pd.notna(row[file_type]):
                    file_path = base_dir / row[file_type]
                    if file_path.exists():
                        existence_stats[file_type]['exists'] += 1
                    else:
                        existence_stats[file_type]['missing'] += 1
                        
        for file_type, stats in existence_stats.items():
            total = stats['exists'] + stats['missing']
            if total > 0:
                exist_pct = (stats['exists'] / total) * 100
                print(f"  📁 {file_type}: {stats['exists']}/{total} exist ({exist_pct:.1f}%)")
    
    # Test 4: Data Integrity Check
    print("\n🔍 Test 4: Data Integrity Analysis")
    all_patients = set()
    cross_split_issues = []
    
    for split_name, split_info in splits_info.items():
        if split_info.get('file_loaded', False):
            df = pd.read_csv(dataset_splits_dir / f"{split_name}.csv")
            
            if 'patient_folder' in df.columns:
                split_patients = set(df['patient_folder'].dropna())
                
                # Check for cross-split patient overlap
                overlap = all_patients.intersection(split_patients)
                if overlap:
                    cross_split_issues.append({
                        'split': split_name,
                        'overlapping_patients': list(overlap)[:5]  # Show first 5
                    })
                
                all_patients.update(split_patients)
                print(f"  👥 {split_name}: {len(split_patients)} unique patients")
    
    if cross_split_issues:
        print("  ⚠️  CRITICAL: Patient overlap detected between splits!")
        for issue in cross_split_issues:
            print(f"    • {issue['split']}: {len(issue['overlapping_patients'])} overlapping patients")
    else:
        print("  ✅ No patient overlap between splits")
    
    # Test 5: Vocabulary Analysis (Expected to find none yet)
    print("\n📚 Test 5: Vocabulary File Analysis")
    vocab_paths = [
        "output/vocabulary/simple_vocab.json",
        "output/vocabulary/medical_vocab.json",
        "output/keywords/basic_keywords.json"
    ]
    
    vocab_found = False
    for vocab_path in vocab_paths:
        if Path(vocab_path).exists():
            vocab_found = True
            print(f"  📖 Found: {vocab_path}")
        else:
            print(f"  📄 Not found: {vocab_path} (expected before keyword extraction)")
    
    if not vocab_found:
        print("  ℹ️  No vocabulary files found - this is expected before Task 6.1")
    
    # Test 6: Generate Quality Report
    print("\n💡 Test 6: Data Quality Summary")
    
    total_splits = len([s for s in splits_info.values() if s.get('file_loaded', False)])
    total_patients = len(all_patients)
    critical_issues = len(cross_split_issues)
    
    quality_score = 100
    if critical_issues > 0:
        quality_score -= 50  # Major penalty for data leakage
    
    if total_splits < 3:
        quality_score -= 20  # Penalty for missing splits
        
    # Generate recommendations
    recommendations = []
    
    if critical_issues > 0:
        recommendations.append({
            'priority': 'CRITICAL',
            'issue': 'Patient overlap between splits detected',
            'action': 'Fix data splits immediately to prevent overfitting'
        })
    
    if total_splits < 3:
        recommendations.append({
            'priority': 'HIGH',
            'issue': f'Only {total_splits}/3 dataset splits loaded successfully',
            'action': 'Verify dataset split files and paths'
        })
    
    if not vocab_found:
        recommendations.append({
            'priority': 'LOW',
            'issue': 'No vocabulary files found',
            'action': 'Expected - proceed with keyword extraction in Task 6.1'
        })
    
    print(f"  📊 Overall Data Quality Score: {quality_score}/100")
    print(f"  📈 Dataset Statistics:")
    print(f"    • Total splits loaded: {total_splits}/3")
    print(f"    • Total unique patients: {total_patients}")
    print(f"    • Critical issues: {critical_issues}")
    
    if recommendations:
        print(f"\n  📋 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. [{rec['priority']}] {rec['issue']}")
            print(f"       Action: {rec['action']}")
    
    # Save simple report
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    report = {
        'task': '5.2 - Data Quality Analysis',
        'timestamp': pd.Timestamp.now().isoformat(),
        'splits_info': splits_info,
        'total_patients': total_patients,
        'quality_score': quality_score,
        'critical_issues': critical_issues,
        'recommendations': recommendations
    }
    
    with open(output_dir / "data_quality_report_simple.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Simple report saved to: output/data_quality_report_simple.json")
    
    # Final result
    print("\n" + "=" * 50)
    if quality_score >= 80:
        print("✅ Task 5.2 COMPLETED: Data quality analysis finished successfully!")
        print(f"🎉 Data quality is {'excellent' if quality_score >= 90 else 'good'} (Score: {quality_score}/100)")
    else:
        print("⚠️  Task 5.2 COMPLETED with issues: Data quality analysis finished")
        print(f"🔧 Data quality needs improvement (Score: {quality_score}/100)")
    
    if critical_issues == 0:
        print("🚀 Ready to proceed to Task 5.3: Create Inference Pipeline")
    else:
        print("⛔ Address critical issues before proceeding to next task")

if __name__ == "__main__":
    test_data_quality_analysis() 