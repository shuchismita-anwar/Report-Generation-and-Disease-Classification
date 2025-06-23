#!/usr/bin/env python3
"""
Keyword Deduplication Script
=============================

This script removes exact duplicate keywords across confidence levels within each condition.
When a keyword appears in multiple confidence levels, it keeps only the highest confidence version.

Priority order: high_confidence > medium_confidence > low_confidence

Usage:
    python deduplicate_keywords.py [input_file] [output_file]
    
Default files:
    Input:  extracted_keywords_result_cleaned.json
    Output: extracted_keywords_result_deduplicated.json
"""

import json
import sys
from datetime import datetime
from typing import Dict, List, Set


def deduplicate_condition_keywords(condition_data: Dict) -> Dict:
    """
    Remove duplicate keywords across confidence levels for a single condition.
    Keeps highest confidence version of each keyword.
    
    Args:
        condition_data: Dictionary with high_confidence, medium_confidence, low_confidence lists
        
    Returns:
        Deduplicated condition data with same structure
    """
    # Normalize keywords for comparison (lowercase, stripped)
    def normalize_keyword(keyword: str) -> str:
        return keyword.lower().strip()
    
    # Track seen keywords and their original forms
    seen_keywords: Set[str] = set()
    
    # Result structure
    result = {
        'high_confidence': [],
        'medium_confidence': [],
        'low_confidence': []
    }
    
    # Copy extraction_stats if it exists
    if 'extraction_stats' in condition_data:
        result['extraction_stats'] = condition_data['extraction_stats']
    
    # Process in priority order: high -> medium -> low
    confidence_levels = ['high_confidence', 'medium_confidence', 'low_confidence']
    
    for confidence_level in confidence_levels:
        if confidence_level in condition_data:
            unique_keywords = []
            
            for keyword in condition_data[confidence_level]:
                normalized = normalize_keyword(keyword)
                
                if normalized not in seen_keywords and normalized:  # Skip empty keywords
                    seen_keywords.add(normalized)
                    unique_keywords.append(keyword)  # Keep original formatting
            
            result[confidence_level] = sorted(unique_keywords)  # Sort for consistency
    
    return result


def deduplicate_all_conditions(data: Dict) -> Dict:
    """
    Deduplicate keywords across all conditions in the dataset.
    
    Args:
        data: Full dataset with conditions and metadata
        
    Returns:
        Deduplicated dataset with same structure
    """
    if 'conditions' not in data:
        raise ValueError("No 'conditions' section found in the data!")
    
    result = data.copy()
    result['conditions'] = {}
    
    # Track statistics
    total_original = 0
    total_deduplicated = 0
    condition_stats = {}
    
    print("🔄 Deduplicating keywords across confidence levels...")
    print("=" * 60)
    
    for condition_name, condition_data in data['conditions'].items():
        # Count original keywords
        original_count = 0
        for conf_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
            if conf_level in condition_data:
                original_count += len(condition_data[conf_level])
        
        # Deduplicate
        deduplicated_data = deduplicate_condition_keywords(condition_data)
        result['conditions'][condition_name] = deduplicated_data
        
        # Count deduplicated keywords
        deduplicated_count = 0
        for conf_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
            if conf_level in deduplicated_data:
                deduplicated_count += len(deduplicated_data[conf_level])
        
        # Track stats
        duplicates_removed = original_count - deduplicated_count
        condition_stats[condition_name] = {
            'original': original_count,
            'deduplicated': deduplicated_count,
            'duplicates_removed': duplicates_removed,
            'duplicate_rate': f"{(duplicates_removed/original_count*100):.1f}%" if original_count > 0 else "0.0%"
        }
        
        total_original += original_count
        total_deduplicated += deduplicated_count
        
        # Print progress
        if duplicates_removed > 0:
            print(f"📝 {condition_name}:")
            print(f"   Original: {original_count} → Deduplicated: {deduplicated_count}")
            print(f"   Removed: {duplicates_removed} duplicates ({condition_stats[condition_name]['duplicate_rate']})")
        else:
            print(f"✅ {condition_name}: {original_count} keywords (no duplicates)")
    
    print("=" * 60)
    print(f"📊 DEDUPLICATION SUMMARY:")
    print(f"   Total original keywords: {total_original}")
    print(f"   Total deduplicated keywords: {total_deduplicated}")
    print(f"   Total duplicates removed: {total_original - total_deduplicated}")
    print(f"   Overall duplicate rate: {((total_original - total_deduplicated)/total_original*100):.1f}%")
    
    return result, condition_stats


def update_metadata(data: Dict, condition_stats: Dict) -> Dict:
    """Update metadata with deduplication information"""
    
    if 'metadata' not in data:
        data['metadata'] = {}
    
    data['metadata']['deduplication'] = {
        'deduplication_date': datetime.now().isoformat(),
        'method': 'exact_match_across_confidence_levels',
        'priority_order': 'high_confidence > medium_confidence > low_confidence',
        'total_duplicates_removed': sum(stats['duplicates_removed'] for stats in condition_stats.values()),
        'condition_breakdown': condition_stats,
        'completed': True
    }
    
    return data


def main():
    """Main function to run deduplication"""
    
    # Parse command line arguments
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    else:
        input_file = "extracted_keywords_result_cleaned.json"
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = "extracted_keywords_result_deduplicated.json"
    
    print(f"🤖 Keyword Deduplication Script")
    print(f"📂 Input file: {input_file}")
    print(f"💾 Output file: {output_file}")
    print()
    
    try:
        # Load data
        print("📂 Loading keywords file...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data.get('conditions', {}))} conditions")
        print()
        
        # Deduplicate
        deduplicated_data, condition_stats = deduplicate_all_conditions(data)
        
        # Update metadata
        deduplicated_data = update_metadata(deduplicated_data, condition_stats)
        
        # Save result
        print(f"\n💾 Saving deduplicated keywords to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(deduplicated_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Deduplication completed successfully!")
        print(f"📁 Deduplicated file saved: {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in input file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during deduplication: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 