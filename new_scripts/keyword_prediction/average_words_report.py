#!/usr/bin/env python3
"""
Average Words Report Analyzer
=============================

This script analyzes cleaned medical reports to find those that match specific criteria:
- Exactly 60-70 words
- Exactly 7-8 sentences

The output will be saved to a text file with matching report paths.

Author: Medical Report Analysis System
Date: 2025-01-19
"""

import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import logging

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('report_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def clean_text_for_analysis(text: str) -> str:
    """
    Clean text for accurate word and sentence counting.
    
    Args:
        text: Raw report text
        
    Returns:
        Cleaned text for analysis
    """
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common report headers/footers that shouldn't count
    text = re.sub(r'FINAL REPORT|PRELIMINARY REPORT|ADDENDUM|IMPRESSION:|FINDINGS:|TECHNIQUE:', '', text, flags=re.IGNORECASE)
    
    # Remove patient identifiers and dates
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)  # Dates
    text = re.sub(r'\b\d{8,}\b', '', text)  # Long numbers (likely IDs)
    
    # Remove extra punctuation that might affect sentence counting
    text = re.sub(r'\.{2,}', '.', text)  # Multiple periods
    text = re.sub(r'\s+', ' ', text.strip())  # Clean up whitespace again
    
    return text

def count_words_and_sentences(text: str) -> Tuple[int, int]:
    """
    Count words and sentences in the text using NLTK.
    
    Args:
        text: Text to analyze
        
    Returns:
        Tuple of (word_count, sentence_count)
    """
    try:
        # Clean text first
        cleaned_text = clean_text_for_analysis(text)
        
        # Count sentences using NLTK
        sentences = sent_tokenize(cleaned_text)
        sentence_count = len(sentences)
        
        # Count words using NLTK (more accurate than simple split)
        words = word_tokenize(cleaned_text)
        # Filter out punctuation-only tokens
        words = [word for word in words if re.match(r'\w', word)]
        word_count = len(words)
        
        return word_count, sentence_count
        
    except Exception as e:
        logging.error(f"Error counting words/sentences: {e}")
        return 0, 0

def analyze_single_report(file_path: Path) -> Dict:
    """
    Analyze a single report file.
    
    Args:
        file_path: Path to the report file
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Read the report
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content.strip():
            return {
                'path': str(file_path),
                'word_count': 0,
                'sentence_count': 0,
                'meets_criteria': False,
                'error': 'Empty file'
            }
        
        # Count words and sentences
        word_count, sentence_count = count_words_and_sentences(content)
        
        # Check if meets criteria (60-70 words, 7-8 sentences)
        meets_criteria = (60 <= word_count <= 70) and (7 <= sentence_count <= 8)
        
        return {
            'path': str(file_path),
            'word_count': word_count,
            'sentence_count': sentence_count,
            'meets_criteria': meets_criteria,
            'content_preview': content[:200] + "..." if len(content) > 200 else content
        }
        
    except Exception as e:
        return {
            'path': str(file_path),
            'word_count': 0,
            'sentence_count': 0,
            'meets_criteria': False,
            'error': str(e)
        }

def find_cleaned_reports_directory() -> Path:
    """
    Find the cleaned_reports directory.
    
    Returns:
        Path to cleaned_reports directory
    """
    # Try multiple possible locations
    possible_paths = [
        Path("../../cleaned_reports"),  # From new_scripts/keyword_prediction/
        Path("../cleaned_reports"),     # From new_scripts/
        Path("cleaned_reports"),        # From root
        Path("./cleaned_reports"),      # Current directory
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return path.resolve()
    
    raise FileNotFoundError("Could not find cleaned_reports directory")

def analyze_all_reports(cleaned_reports_dir: Path, logger) -> List[Dict]:
    """
    Analyze all reports in the cleaned_reports directory.
    
    Args:
        cleaned_reports_dir: Path to cleaned reports directory
        logger: Logger instance
        
    Returns:
        List of analysis results
    """
    results = []
    total_files = 0
    processed_files = 0
    
    logger.info(f"Scanning directory: {cleaned_reports_dir}")
    
    # Walk through all subdirectories to find .txt files
    for root, dirs, files in os.walk(cleaned_reports_dir):
        for file in files:
            if file.endswith('.txt'):
                total_files += 1
                file_path = Path(root) / file
                
                # Analyze the report
                result = analyze_single_report(file_path)
                results.append(result)
                processed_files += 1
                
                # Log progress every 100 files
                if processed_files % 100 == 0:
                    logger.info(f"Processed {processed_files}/{total_files} files...")
    
    logger.info(f"Analysis complete: {processed_files} files processed")
    return results

def save_matching_reports(results: List[Dict], output_file: str, logger):
    """
    Save reports that meet the criteria to JSON output file.
    
    Args:
        results: Analysis results
        output_file: Output file path
        logger: Logger instance
    """
    matching_reports = [r for r in results if r.get('meets_criteria', False)]
    
    logger.info(f"Found {len(matching_reports)} reports matching criteria (60-70 words, 7-8 sentences)")
    
    # Prepare JSON data structure
    json_data = {
        "metadata": {
            "title": "Medical Reports Matching Criteria (60-70 words, 7-8 sentences)",
            "total_matching_reports": len(matching_reports),
            "analysis_date": datetime.now().isoformat(),
            "criteria": {
                "word_count_range": [60, 70],
                "sentence_count_range": [7, 8]
            }
        },
        "matching_reports": []
    }
    
    # Process each matching report
    for i, report in enumerate(matching_reports, 1):
        # Extract the relative path in the desired format
        path_str = report['path']
        
        # Try to extract patient_XXXXXXXX/CXR-DICOM/sXXXXXXXX.txt format
        path_parts = Path(path_str).parts
        formatted_path = path_str
        patient_id = None
        study_id = None
        
        if len(path_parts) >= 3:
            # Look for patient folder pattern
            patient_folder = None
            cxr_folder = None
            filename = Path(path_str).name
            
            for part in path_parts:
                if part.startswith('patient_') or 'patient' in part.lower():
                    patient_folder = part
                    # Extract patient ID
                    if part.startswith('patient_'):
                        patient_id = part.replace('patient_', '')
                elif 'cxr' in part.lower() or 'dicom' in part.lower():
                    cxr_folder = part
            
            if patient_folder and cxr_folder:
                formatted_path = f"{patient_folder}/{cxr_folder}/{filename}"
            else:
                # Fallback to last 3 parts of path
                formatted_path = "/".join(path_parts[-3:])
            
            # Extract study ID from filename
            if filename.startswith('s') and filename.endswith('.txt'):
                study_id = filename[1:-4]  # Remove 's' prefix and '.txt' suffix
        
        # Create report entry
        report_entry = {
            "index": i,
            "path": formatted_path,
            "full_path": path_str,
            "patient_id": patient_id,
            "study_id": study_id,
            "word_count": report['word_count'],
            "sentence_count": report['sentence_count'],
            "content_preview": report.get('content_preview', 'N/A')[:200] + "..." if len(report.get('content_preview', '')) > 200 else report.get('content_preview', 'N/A')
        }
        
        json_data["matching_reports"].append(report_entry)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to JSON file: {output_file}")
    
    # Also save a summary text file for quick viewing
    summary_file = output_file.replace('.json', '_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("MEDICAL REPORTS MATCHING CRITERIA - SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total matching reports: {len(matching_reports)}\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for report in json_data["matching_reports"]:
            f.write(f"{report['index']:4d}. {report['path']}\n")
            f.write(f"      Words: {report['word_count']}, Sentences: {report['sentence_count']}\n")
            if report['patient_id']:
                f.write(f"      Patient ID: {report['patient_id']}, Study ID: {report['study_id']}\n")
            f.write("\n")
    
    logger.info(f"Summary also saved to: {summary_file}")

def generate_summary_statistics(results: List[Dict], logger):
    """
    Generate and log summary statistics.
    
    Args:
        results: Analysis results
        logger: Logger instance
    """
    total_reports = len(results)
    valid_reports = [r for r in results if not r.get('error')]
    matching_reports = [r for r in results if r.get('meets_criteria', False)]
    
    if valid_reports:
        word_counts = [r['word_count'] for r in valid_reports]
        sentence_counts = [r['sentence_count'] for r in valid_reports]
        
        logger.info("=" * 50)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total reports analyzed: {total_reports}")
        logger.info(f"Valid reports: {len(valid_reports)}")
        logger.info(f"Reports with errors: {total_reports - len(valid_reports)}")
        logger.info(f"Reports matching criteria (60-70 words, 7-8 sentences): {len(matching_reports)}")
        logger.info(f"Match percentage: {len(matching_reports)/len(valid_reports)*100:.2f}%")
        logger.info("")
        logger.info(f"Word count - Min: {min(word_counts)}, Max: {max(word_counts)}, Avg: {sum(word_counts)/len(word_counts):.1f}")
        logger.info(f"Sentence count - Min: {min(sentence_counts)}, Max: {max(sentence_counts)}, Avg: {sum(sentence_counts)/len(sentence_counts):.1f}")
        logger.info("=" * 50)

def main():
    """Main execution function"""
    logger = setup_logging()
    
    try:
        logger.info("Starting medical report analysis...")
        
        # Find cleaned reports directory
        cleaned_reports_dir = find_cleaned_reports_directory()
        logger.info(f"Found cleaned reports directory: {cleaned_reports_dir}")
        
        # Analyze all reports
        results = analyze_all_reports(cleaned_reports_dir, logger)
        
        # Generate summary statistics
        generate_summary_statistics(results, logger)
        
        # Save matching reports
        output_file = "reports_60_70_words_7_8_sentences.json"
        save_matching_reports(results, output_file, logger)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
