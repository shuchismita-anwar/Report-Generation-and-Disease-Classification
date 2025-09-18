import json
import re
from typing import Dict, List, Set

def load_keywords_file(file_path: str) -> Dict:
    """Load the keywords JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_keywords_file(data: Dict, file_path: str) -> None:
    """Save the cleaned keywords JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def clean_keywords(keywords: List[str]) -> List[str]:
    """
    Clean a list of keywords by removing stop words and duplicates.
    
    Args:
        keywords: List of keyword strings
        
    Returns:
        Cleaned list of keywords
    """
    # Define stop words to remove
    stop_words = {
        "the", "and", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "can", "of", "in", "on", "at", "to", "for", "with", "by",
        "from", "as", "or", "but", "not", "this", "that", "these", "those", "it",
        "its", "he", "she", "his", "her", "him", "they", "them", "their", "we",
        "us", "our", "you", "your", "i", "me", "my", "am"
    }
    
    cleaned_keywords = []
    seen_keywords = set()  # Track duplicates (case-insensitive)
    
    for keyword in keywords:
        if not isinstance(keyword, str):
            continue
            
        # Strip whitespace and convert to lowercase for comparison
        keyword_clean = keyword.strip()
        keyword_lower = keyword_clean.lower()
        
        # Skip empty keywords
        if not keyword_clean:
            continue
            
        # Skip stop words
        if keyword_lower in stop_words:
            continue
            
        # Skip duplicates (case-insensitive)
        if keyword_lower in seen_keywords:
            continue
            
        # Add to cleaned list and mark as seen
        cleaned_keywords.append(keyword_clean)
        seen_keywords.add(keyword_lower)
    
    return cleaned_keywords

def clean_conditions_keywords(conditions_data: Dict) -> Dict:
    """Clean keywords in conditions data."""
    cleaned_data = {}
    
    for condition_name, condition_data in conditions_data.items():
        cleaned_data[condition_name] = {}
        
        # Clean each confidence level
        for confidence_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
            if confidence_level in condition_data:
                cleaned_data[condition_name][confidence_level] = clean_keywords(condition_data[confidence_level])
    
    return cleaned_data

def print_cleanup_stats(original_data: Dict, cleaned_data: Dict) -> None:
    """Print statistics about the cleanup process."""
    print(f"\n=== CONDITIONS CLEANUP STATS ===")
    
    total_original = 0
    total_cleaned = 0
    
    for condition_name in original_data.keys():
        condition_original = 0
        condition_cleaned = 0
        
        print(f"\n{condition_name}:")
        
        for confidence_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
            if confidence_level in original_data[condition_name]:
                original_count = len(original_data[condition_name][confidence_level])
                cleaned_count = len(cleaned_data[condition_name][confidence_level]) if confidence_level in cleaned_data[condition_name] else 0
                removed_count = original_count - cleaned_count
                
                print(f"  {confidence_level}:")
                print(f"    Original: {original_count} keywords")
                print(f"    Cleaned:  {cleaned_count} keywords")
                print(f"    Removed:  {removed_count} keywords ({removed_count/original_count*100:.1f}%)")
                
                condition_original += original_count
                condition_cleaned += cleaned_count
        
        condition_removed = condition_original - condition_cleaned
        print(f"  TOTAL for {condition_name}:")
        print(f"    Original: {condition_original} keywords")
        print(f"    Cleaned:  {condition_cleaned} keywords") 
        print(f"    Removed:  {condition_removed} keywords ({condition_removed/condition_original*100:.1f}%)")
        
        total_original += condition_original
        total_cleaned += condition_cleaned
    
    total_removed = total_original - total_cleaned
    print(f"\n=== OVERALL STATS ===")
    print(f"Total Original: {total_original} keywords")
    print(f"Total Cleaned:  {total_cleaned} keywords")
    print(f"Total Removed:  {total_removed} keywords ({total_removed/total_original*100:.1f}%)")

def main():
    """Main function to perform static cleanup."""
    input_file = "extracted_keywords_result.json"
    output_file = "extracted_keywords_result_cleaned.json"
    
    print("Loading keywords file...")
    data = load_keywords_file(input_file)
    
    # Create a copy for cleaning
    cleaned_data = data.copy()
    
    # Clean conditions
    if 'conditions' in data:
        print("Cleaning conditions keywords...")
        original_conditions = data['conditions']
        cleaned_data['conditions'] = clean_conditions_keywords(original_conditions)
        print_cleanup_stats(original_conditions, cleaned_data['conditions'])
    else:
        print("No 'conditions' section found in the data!")
        return
    
    # Save cleaned data
    print(f"\nSaving cleaned keywords to {output_file}...")
    save_keywords_file(cleaned_data, output_file)
    
    print("Static cleanup completed!")

if __name__ == "__main__":
    main()
