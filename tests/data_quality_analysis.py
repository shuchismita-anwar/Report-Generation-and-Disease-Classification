#!/usr/bin/env python3
"""
Data Quality Analysis for Hybrid RadBERT + MedCLIP System
Task 5.2: Validate Data Quality

This script performs comprehensive analysis of:
1. Transcript keyword extraction quality
2. Vocabulary distribution and relevance
3. Region availability patterns
4. Data leakage and consistency checks
5. Dataset split integrity
6. File existence and accessibility
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional
import warnings
warnings.filterwarnings('ignore')

class DataQualityAnalyzer:
    """Comprehensive data quality analysis for medical imaging dataset"""
    
    def __init__(self, base_dir: str = "../../"):
        self.base_dir = Path(base_dir)
        self.dataset_splits_dir = self.base_dir / "dataset_splits"
        self.mimic_eye_dir = self.base_dir / "mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0"
        
        # Results storage
        self.results = {
            'file_existence': {},
            'transcript_quality': {},
            'vocabulary_analysis': {},
            'region_patterns': {},
            'data_integrity': {},
            'split_analysis': {},
            'recommendations': []
        }
        
        # Load dataset splits
        self.load_dataset_splits()
        
    def load_dataset_splits(self) -> None:
        """Load all dataset splits"""
        print("📊 Loading dataset splits...")
        
        splits = ['train', 'val', 'test']
        self.dataframes = {}
        
        for split in splits:
            split_path = self.dataset_splits_dir / f"{split}.csv"
            if split_path.exists():
                try:
                    df = pd.read_csv(split_path)
                    self.dataframes[split] = df
                    print(f"  ✅ {split}: {len(df)} samples")
                    self.results['split_analysis'][split] = {
                        'total_samples': len(df),
                        'file_loaded': True
                    }
                except Exception as e:
                    print(f"  ❌ {split}: Error loading - {e}")
                    self.results['split_analysis'][split] = {
                        'file_loaded': False,
                        'error': str(e)
                    }
            else:
                print(f"  ⚠️  {split}: File not found at {split_path}")
                self.results['split_analysis'][split] = {
                    'file_loaded': False,
                    'error': 'File not found'
                }
    
    def analyze_file_existence(self, sample_size: int = 50) -> None:
        """Analyze file existence and accessibility"""
        print(f"\n🔍 Analyzing file existence (sample size: {sample_size})...")
        
        file_types = ['image_path', 'fixations_path', 'bbox_path', 'transcript_path', 'report_path']
        results = {ft: {'exists': 0, 'missing': 0, 'accessible': 0} for ft in file_types}
        
        # Sample from train split
        if 'train' not in self.dataframes:
            print("  ❌ No train split available for analysis")
            return
            
        df = self.dataframes['train']
        sample_df = df.sample(min(sample_size, len(df)), random_state=42)
        
        for idx, row in sample_df.iterrows():
            for file_type in file_types:
                if file_type in row and pd.notna(row[file_type]):
                    file_path = self.base_dir / row[file_type]
                    
                    if file_path.exists():
                        results[file_type]['exists'] += 1
                        
                        # Check if file is accessible/readable
                        try:
                            if file_type == 'transcript_path' and file_path.suffix == '.json':
                                with open(file_path, 'r') as f:
                                    json.load(f)
                            elif file_type in ['image_path']:
                                # Just check if file can be opened
                                with open(file_path, 'rb') as f:
                                    f.read(1024)  # Read first 1KB
                            elif file_type in ['fixations_path', 'bbox_path'] and file_path.suffix == '.csv':
                                pd.read_csv(file_path, nrows=1)
                            elif file_type == 'report_path':
                                with open(file_path, 'r') as f:
                                    f.read(100)  # Read first 100 chars
                            
                            results[file_type]['accessible'] += 1
                        except Exception as e:
                            print(f"    ⚠️  {file_type}: File exists but not accessible - {e}")
                    else:
                        results[file_type]['missing'] += 1
                        
        # Store results
        self.results['file_existence'] = results
        
        # Print summary
        for file_type, stats in results.items():
            total = stats['exists'] + stats['missing']
            if total > 0:
                exist_pct = (stats['exists'] / total) * 100
                access_pct = (stats['accessible'] / stats['exists']) * 100 if stats['exists'] > 0 else 0
                print(f"  📁 {file_type}: {exist_pct:.1f}% exist, {access_pct:.1f}% accessible")
    
    def analyze_transcript_quality(self, sample_size: int = 20) -> None:
        """Analyze transcript content and quality"""
        print(f"\n📝 Analyzing transcript quality (sample size: {sample_size})...")
        
        if 'train' not in self.dataframes:
            print("  ❌ No train split available for analysis")
            return
            
        df = self.dataframes['train']
        sample_df = df.sample(min(sample_size, len(df)), random_state=42)
        
        transcript_stats = {
            'total_analyzed': 0,
            'valid_json': 0,
            'has_text_content': 0,
            'avg_length': 0,
            'lengths': [],
            'word_counts': [],
            'medical_terms': Counter(),
            'common_words': Counter(),
            'sample_contents': []
        }
        
        medical_keywords = [
            'lung', 'heart', 'chest', 'pneumonia', 'atelectasis', 'effusion',
            'consolidation', 'opacity', 'nodule', 'mass', 'fracture', 'edema',
            'cardiac', 'pulmonary', 'thorax', 'pleural', 'mediastinal', 'rib',
            'clavicle', 'diaphragm', 'costophrenic', 'hilum', 'aorta'
        ]
        
        for idx, row in sample_df.iterrows():
            if 'transcript_path' in row and pd.notna(row['transcript_path']):
                transcript_path = self.base_dir / row['transcript_path']
                
                if transcript_path.exists():
                    try:
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                            transcript_data = json.load(f)
                        
                        transcript_stats['valid_json'] += 1
                        
                        # Extract text content (format may vary)
                        text_content = ""
                        if isinstance(transcript_data, dict):
                            # Common transcript JSON structures
                            text_content = (
                                transcript_data.get('transcript', '') or 
                                transcript_data.get('text', '') or
                                transcript_data.get('content', '') or
                                str(transcript_data)
                            )
                        elif isinstance(transcript_data, list):
                            text_content = ' '.join([str(item) for item in transcript_data])
                        else:
                            text_content = str(transcript_data)
                        
                        if text_content and len(text_content.strip()) > 0:
                            transcript_stats['has_text_content'] += 1
                            
                            # Basic text analysis
                            clean_text = text_content.lower().strip()
                            transcript_stats['lengths'].append(len(clean_text))
                            
                            words = clean_text.split()
                            transcript_stats['word_counts'].append(len(words))
                            
                            # Count medical terms
                            for word in words:
                                if word in medical_keywords:
                                    transcript_stats['medical_terms'][word] += 1
                                transcript_stats['common_words'][word] += 1
                            
                            # Store sample content
                            if len(transcript_stats['sample_contents']) < 3:
                                transcript_stats['sample_contents'].append({
                                    'patient': row.get('patient_folder', 'unknown'),
                                    'length': len(clean_text),
                                    'word_count': len(words),
                                    'preview': clean_text[:200] + "..." if len(clean_text) > 200 else clean_text
                                })
                        
                    except Exception as e:
                        print(f"    ⚠️  Error processing transcript {transcript_path}: {e}")
                
                transcript_stats['total_analyzed'] += 1
        
        # Calculate averages
        if transcript_stats['lengths']:
            transcript_stats['avg_length'] = np.mean(transcript_stats['lengths'])
            transcript_stats['avg_word_count'] = np.mean(transcript_stats['word_counts'])
            transcript_stats['std_length'] = np.std(transcript_stats['lengths'])
            transcript_stats['std_word_count'] = np.std(transcript_stats['word_counts'])
        
        self.results['transcript_quality'] = transcript_stats
        
        # Print summary
        print(f"  📊 Analysis Summary:")
        print(f"    • Total analyzed: {transcript_stats['total_analyzed']}")
        print(f"    • Valid JSON: {transcript_stats['valid_json']}")
        print(f"    • Has text content: {transcript_stats['has_text_content']}")
        
        if transcript_stats['lengths']:
            print(f"    • Avg length: {transcript_stats['avg_length']:.1f} ± {transcript_stats['std_length']:.1f} chars")
            print(f"    • Avg words: {transcript_stats['avg_word_count']:.1f} ± {transcript_stats['std_word_count']:.1f}")
            print(f"    • Medical terms found: {len(transcript_stats['medical_terms'])} unique")
            
            # Show top medical terms
            top_medical = transcript_stats['medical_terms'].most_common(5)
            if top_medical:
                print(f"    • Top medical terms: {', '.join([f'{term}({count})' for term, count in top_medical])}")
    
    def analyze_vocabulary_distribution(self) -> None:
        """Analyze vocabulary patterns from extracted keywords (if available)"""
        print(f"\n📚 Analyzing vocabulary distribution...")
        
        # Check if vocabulary files exist from previous extractions
        vocab_paths = [
            "output/vocabulary/simple_vocab.json",
            "output/vocabulary/medical_vocab.json",
            "output/keywords/basic_keywords.json"
        ]
        
        vocab_found = False
        for vocab_path in vocab_paths:
            full_path = Path(vocab_path)
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        vocab_data = json.load(f)
                    
                    vocab_stats = {
                        'source_file': vocab_path,
                        'total_entries': len(vocab_data),
                        'structure': type(vocab_data).__name__
                    }
                    
                    if isinstance(vocab_data, dict):
                        if 'word2idx' in vocab_data:
                            vocab_stats['vocabulary_size'] = len(vocab_data['word2idx'])
                            vocab_stats['sample_words'] = list(vocab_data['word2idx'].keys())[:10]
                        elif 'vocabulary' in vocab_data:
                            vocab_stats['vocabulary_size'] = len(vocab_data['vocabulary'])
                            vocab_stats['sample_words'] = vocab_data['vocabulary'][:10]
                        else:
                            vocab_stats['vocabulary_size'] = len(vocab_data)
                            vocab_stats['sample_words'] = list(vocab_data.keys())[:10]
                    
                    self.results['vocabulary_analysis'][vocab_path] = vocab_stats
                    
                    print(f"  📖 Found vocabulary: {vocab_path}")
                    print(f"    • Size: {vocab_stats.get('vocabulary_size', 'unknown')}")
                    print(f"    • Sample: {', '.join(vocab_stats.get('sample_words', [])[:5])}")
                    
                    vocab_found = True
                    
                except Exception as e:
                    print(f"    ⚠️  Error loading {vocab_path}: {e}")
        
        if not vocab_found:
            print("  ℹ️  No vocabulary files found - this is expected before keyword extraction")
            self.results['vocabulary_analysis']['status'] = 'No vocabulary files found'
    
    def analyze_region_availability(self, sample_size: int = 30) -> None:
        """Analyze region availability patterns from bbox files"""
        print(f"\n🗺️  Analyzing region availability patterns (sample size: {sample_size})...")
        
        if 'train' not in self.dataframes:
            print("  ❌ No train split available for analysis")
            return
            
        df = self.dataframes['train']
        sample_df = df.sample(min(sample_size, len(df)), random_state=42)
        
        region_stats = {
            'total_analyzed': 0,
            'bbox_files_found': 0,
            'region_counts': Counter(),
            'region_patterns': defaultdict(int),
            'avg_regions_per_image': 0,
            'region_availability_matrix': []
        }
        
        for idx, row in sample_df.iterrows():
            if 'bbox_path' in row and pd.notna(row['bbox_path']):
                bbox_path = self.base_dir / row['bbox_path']
                
                if bbox_path.exists():
                    try:
                        bbox_df = pd.read_csv(bbox_path)
                        region_stats['bbox_files_found'] += 1
                        
                        # Analyze region information
                        if 'region' in bbox_df.columns:
                            regions = bbox_df['region'].unique()
                            region_stats['region_counts'].update(regions)
                            
                            # Track region patterns
                            region_pattern = '|'.join(sorted(regions))
                            region_stats['region_patterns'][region_pattern] += 1
                            
                            # Track availability matrix
                            region_stats['region_availability_matrix'].append(list(regions))
                        
                        elif len(bbox_df) > 0:
                            # If no region column, count as generic regions
                            num_regions = len(bbox_df)
                            region_stats['region_counts'][f'generic_region_{num_regions}'] += 1
                        
                    except Exception as e:
                        print(f"    ⚠️  Error processing bbox file {bbox_path}: {e}")
                
                region_stats['total_analyzed'] += 1
        
        # Calculate statistics
        if region_stats['region_availability_matrix']:
            all_regions = set()
            for regions in region_stats['region_availability_matrix']:
                all_regions.update(regions)
            
            region_stats['unique_regions'] = list(all_regions)
            region_stats['total_unique_regions'] = len(all_regions)
            
            # Calculate average regions per image
            region_counts_per_image = [len(regions) for regions in region_stats['region_availability_matrix']]
            region_stats['avg_regions_per_image'] = np.mean(region_counts_per_image)
            region_stats['std_regions_per_image'] = np.std(region_counts_per_image)
        
        self.results['region_patterns'] = region_stats
        
        # Print summary
        print(f"  📊 Region Analysis Summary:")
        print(f"    • Total analyzed: {region_stats['total_analyzed']}")
        print(f"    • BBox files found: {region_stats['bbox_files_found']}")
        
        if region_stats.get('total_unique_regions', 0) > 0:
            print(f"    • Unique regions found: {region_stats['total_unique_regions']}")
            print(f"    • Avg regions per image: {region_stats['avg_regions_per_image']:.1f} ± {region_stats['std_regions_per_image']:.1f}")
            
            # Show most common regions
            top_regions = region_stats['region_counts'].most_common(5)
            if top_regions:
                print(f"    • Most common regions: {', '.join([f'{region}({count})' for region, count in top_regions])}")
    
    def analyze_data_integrity(self) -> None:
        """Check for data leakage and consistency issues"""
        print(f"\n🔍 Analyzing data integrity and consistency...")
        
        integrity_issues = {
            'duplicate_patients': [],
            'duplicate_images': [],
            'missing_conditions': [],
            'inconsistent_paths': [],
            'cross_split_leakage': []
        }
        
        # Analyze across all splits
        all_patients = set()
        all_images = set()
        all_studies = set()
        
        for split_name, df in self.dataframes.items():
            if df is None:
                continue
                
            # Check for missing conditions
            if 'condition' in df.columns:
                missing_conditions = df['condition'].isna().sum()
                if missing_conditions > 0:
                    integrity_issues['missing_conditions'].append({
                        'split': split_name,
                        'count': missing_conditions,
                        'percentage': (missing_conditions / len(df)) * 100
                    })
            
            # Track patients and images for cross-split analysis
            if 'patient_folder' in df.columns:
                split_patients = set(df['patient_folder'].dropna())
                
                # Check for duplicates within split
                duplicate_patients = df['patient_folder'].value_counts()
                duplicates = duplicate_patients[duplicate_patients > 1]
                if len(duplicates) > 0:
                    integrity_issues['duplicate_patients'].append({
                        'split': split_name,
                        'duplicates': duplicates.to_dict()
                    })
                
                # Check for cross-split patient leakage
                overlap = all_patients.intersection(split_patients)
                if overlap:
                    integrity_issues['cross_split_leakage'].append({
                        'split': split_name,
                        'overlapping_patients': list(overlap)
                    })
                
                all_patients.update(split_patients)
            
            # Similar analysis for images
            if 'dicom_id' in df.columns:
                split_images = set(df['dicom_id'].dropna())
                
                overlap = all_images.intersection(split_images)
                if overlap:
                    integrity_issues['cross_split_leakage'].append({
                        'split': split_name,
                        'overlapping_images': list(overlap)
                    })
                
                all_images.update(split_images)
        
        self.results['data_integrity'] = integrity_issues
        
        # Print summary
        print(f"  📊 Data Integrity Summary:")
        
        issues_found = False
        for issue_type, issues in integrity_issues.items():
            if issues:
                issues_found = True
                print(f"    ⚠️  {issue_type}: {len(issues)} instances")
                
                if issue_type == 'cross_split_leakage':
                    for issue in issues:
                        if 'overlapping_patients' in issue:
                            print(f"      • {issue['split']}: {len(issue['overlapping_patients'])} overlapping patients")
                        if 'overlapping_images' in issue:
                            print(f"      • {issue['split']}: {len(issue['overlapping_images'])} overlapping images")
        
        if not issues_found:
            print(f"    ✅ No major integrity issues detected")
    
    def generate_recommendations(self) -> None:
        """Generate actionable recommendations based on analysis"""
        print(f"\n💡 Generating recommendations...")
        
        recommendations = []
        
        # File existence recommendations
        file_exists = self.results.get('file_existence', {})
        for file_type, stats in file_exists.items():
            total = stats.get('exists', 0) + stats.get('missing', 0)
            if total > 0:
                exist_rate = stats.get('exists', 0) / total
                access_rate = stats.get('accessible', 0) / max(stats.get('exists', 1), 1)
                
                if exist_rate < 0.9:
                    recommendations.append({
                        'priority': 'HIGH',
                        'category': 'File Existence',
                        'issue': f'{file_type} files missing in {(1-exist_rate)*100:.1f}% of cases',
                        'action': f'Verify file paths and data integrity for {file_type}'
                    })
                
                if access_rate < 0.95:
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'category': 'File Access',
                        'issue': f'{file_type} files not accessible in {(1-access_rate)*100:.1f}% of cases',
                        'action': f'Check file permissions and formats for {file_type}'
                    })
        
        # Transcript quality recommendations
        transcript_quality = self.results.get('transcript_quality', {})
        if transcript_quality.get('total_analyzed', 0) > 0:
            text_content_rate = transcript_quality.get('has_text_content', 0) / transcript_quality.get('total_analyzed', 1)
            
            if text_content_rate < 0.8:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Transcript Quality',
                    'issue': f'Only {text_content_rate*100:.1f}% of transcripts have extractable text content',
                    'action': 'Improve transcript parsing logic and verify JSON structure'
                })
            
            avg_length = transcript_quality.get('avg_length', 0)
            if avg_length < 100:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Transcript Content',
                    'issue': f'Average transcript length is only {avg_length:.1f} characters',
                    'action': 'Verify transcript completeness and extraction method'
                })
        
        # Data integrity recommendations
        integrity_issues = self.results.get('data_integrity', {})
        if integrity_issues.get('cross_split_leakage'):
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Data Leakage',
                'issue': 'Patient or image overlap detected between train/val/test splits',
                'action': 'Immediately fix data splits to prevent model overfitting'
            })
        
        # Vocabulary recommendations
        vocab_analysis = self.results.get('vocabulary_analysis', {})
        if not vocab_analysis or vocab_analysis.get('status') == 'No vocabulary files found':
            recommendations.append({
                'priority': 'LOW',
                'category': 'Vocabulary',
                'issue': 'No vocabulary files found',
                'action': 'This is expected before keyword extraction - proceed with Task 6.1'
            })
        
        # Store and display recommendations
        self.results['recommendations'] = recommendations
        
        # Group by priority
        critical = [r for r in recommendations if r['priority'] == 'CRITICAL']
        high = [r for r in recommendations if r['priority'] == 'HIGH']
        medium = [r for r in recommendations if r['priority'] == 'MEDIUM']
        low = [r for r in recommendations if r['priority'] == 'LOW']
        
        for priority, recs in [('CRITICAL', critical), ('HIGH', high), ('MEDIUM', medium), ('LOW', low)]:
            if recs:
                print(f"\n  🚨 {priority} Priority:")
                for i, rec in enumerate(recs, 1):
                    print(f"    {i}. [{rec['category']}] {rec['issue']}")
                    print(f"       Action: {rec['action']}")
    
    def save_report(self, output_path: str = "output/data_quality_report.json") -> None:
        """Save complete analysis report"""
        print(f"\n💾 Saving data quality report...")
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        self.results['metadata'] = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'base_directory': str(self.base_dir),
            'total_splits': len(self.dataframes),
            'analysis_version': '1.0'
        }
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"  ✅ Report saved to: {output_path}")
    
    def run_full_analysis(self) -> None:
        """Run complete data quality analysis"""
        print("🔬 Starting comprehensive data quality analysis...")
        print("=" * 60)
        
        try:
            self.analyze_file_existence(sample_size=50)
            self.analyze_transcript_quality(sample_size=20)
            self.analyze_vocabulary_distribution()
            self.analyze_region_availability(sample_size=30)
            self.analyze_data_integrity()
            self.generate_recommendations()
            
            # Save report
            self.save_report()
            
            print("\n" + "=" * 60)
            print("✅ Data quality analysis completed successfully!")
            
            # Final summary
            total_issues = len([r for r in self.results.get('recommendations', []) 
                              if r['priority'] in ['CRITICAL', 'HIGH']])
            
            if total_issues == 0:
                print("🎉 No critical issues found - data quality looks good!")
            else:
                print(f"⚠️  Found {total_issues} critical/high priority issues to address")
            
        except Exception as e:
            print(f"\n❌ Analysis failed with error: {e}")
            raise


def main():
    """Main function to run data quality analysis"""
    print("🔬 Data Quality Analysis for Hybrid RadBERT + MedCLIP System")
    print("Task 5.2: Validate Data Quality")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DataQualityAnalyzer()
    
    # Run analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()