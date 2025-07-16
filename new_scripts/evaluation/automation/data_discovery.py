#!/usr/bin/env python3
"""
Enhanced Data Discovery System for MIMIC-Eye Evaluation
========================================================
Proper CSV-based discovery using the master dataset files.

Author: Medical Report Evaluation System
Version: 2.0 (Fixed)
"""

import json
import os
import pandas as pd
import logging
from typing import Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def discover_and_match_reports(
    llm_path: str = "../real_analysis_results/",
    dataset_csv_path: str = "../../dataset_splits/val.csv",
    use_all_splits: bool = True
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Enhanced discovery using CSV master files for precise mapping.
    
    Returns:
        Tuple of (matched_pairs, unmatched_llm_reports, unmatched_ground_truth)
    """
    logger.info("🔍 Starting CSV-based data discovery...")
    
    # Step 1: Load LLM reports
    llm_reports = _load_llm_reports(llm_path)
    logger.info(f"📊 Found {len(llm_reports)} LLM reports")
    
    # Step 2: Load and combine dataset CSVs
    dataset_df = _load_dataset_files(use_all_splits)
    logger.info(f"📊 Loaded {len(dataset_df)} entries from dataset CSV(s)")
    
    # Step 3: Match reports using CSV mapping
    matched_pairs, unmatched_llm, unmatched_gt = _match_reports_csv_based(
        llm_reports, dataset_df
    )
    
    logger.info(f"✅ Discovery complete: {len(matched_pairs)} matched, {len(unmatched_llm)} unmatched LLM, {len(unmatched_gt)} unmatched GT")
    
    return matched_pairs, unmatched_llm, unmatched_gt

def _load_llm_reports(llm_path: str) -> Dict[str, str]:
    """Load all LLM report files and extract study IDs."""
    llm_reports = {}
    llm_dir = Path(llm_path)
    
    if not llm_dir.exists():
        logger.warning(f"LLM directory not found: {llm_path}")
        return llm_reports
    
    for report_file in llm_dir.glob("analysis_*.json"):
        try:
            # Extract study ID from filename
            study_id = report_file.stem.replace("analysis_", "")
            llm_reports[study_id] = str(report_file)
            logger.debug(f"Found LLM report: {study_id}")
        except Exception as e:
            logger.warning(f"Error processing {report_file}: {e}")
    
    return llm_reports

def _load_dataset_files(use_all_splits: bool = True) -> pd.DataFrame:
    """Load the main dataset CSV file with all metadata for dicom_id mapping."""
    # Use the main dataset CSV file for mapping instead of split files
    dataset_csv_path = "../../final_dataset_fixed.csv"
    
    if not os.path.exists(dataset_csv_path):
        logger.error(f"Main dataset CSV not found: {dataset_csv_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(dataset_csv_path)
        logger.info(f"Loaded {len(df)} entries from {dataset_csv_path}")
        
        # Ensure required columns exist
        required_columns = ['dicom_id', 'patient_folder', 'study_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in dataset: {missing_columns}")
            return pd.DataFrame()
        
        logger.info(f"Dataset loaded successfully with columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {dataset_csv_path}: {e}")
        return pd.DataFrame()

def _match_reports_csv_based(
    llm_reports: Dict[str, str], 
    dataset_df: pd.DataFrame
) -> Tuple[List[Dict], List[str], List[str]]:
    """Match reports using CSV-based precise mapping."""
    matched_pairs = []
    unmatched_llm = []
    unmatched_gt = []
    
    # Track which ground truth entries we've matched
    matched_gt_ids = set()
    
    # Process each LLM report
    for study_id, llm_file in llm_reports.items():
        # Look for matching entry in dataset
        mask = dataset_df['dicom_id'] == study_id
        matching_rows = dataset_df[mask]
        
        if not matching_rows.empty:
            # Found a match!
            row = matching_rows.iloc[0]
            
            # Extract report path from CSV instead of transcript path
            report_path = row.get('report_path', '')
            
            if report_path and _validate_report_path(report_path):
                # Create matched pair
                matched_pair = {
                    'study_id': study_id,
                    'llm_report': llm_file,
                    'ground_truth': report_path,  # Now using actual radiology report
                    'patient_folder': row.get('patient_folder', ''),
                    'has_reflacx': row.get('ifReflacx', False),
                    'has_eyegaze': row.get('ifEyegaze', False),
                    'condition': row.get('condition', ''),
                    'report_path': report_path,
                    'image_path': row.get('image_path', '')
                }
                matched_pairs.append(matched_pair)
                matched_gt_ids.add(study_id)
                
                logger.debug(f"✅ Matched: {study_id}")
            else:
                logger.warning(f"⚠️ Invalid report path for {study_id}: {report_path}")
                unmatched_llm.append(study_id)
        else:
            logger.debug(f"❌ No CSV entry found for LLM report: {study_id}")
            unmatched_llm.append(study_id)
    
    # Find unmatched ground truth entries (those in CSV but not in LLM reports)
    for _, row in dataset_df.iterrows():
        study_id = row.get('dicom_id', '')
        if study_id and study_id not in matched_gt_ids and study_id not in llm_reports:
            report_path = row.get('report_path', '')
            if report_path and _validate_report_path(report_path):
                unmatched_gt.append(study_id)
    
    return matched_pairs, unmatched_llm, unmatched_gt

def _validate_report_path(report_path: str) -> bool:
    """Validate that a report path exists and is accessible."""
    if not report_path:
        return False
    
    # Try both absolute and relative paths
    if os.path.exists(report_path):
        return True
    
    # Try relative to current directory
    relative_path = os.path.join("..", "..", report_path)
    if os.path.exists(relative_path):
        return True
    
    return False

def load_ground_truth_report(dicom_id: str) -> Dict:
    """
    Load ground truth radiology report using dicom_id mapping.
    
    Args:
        dicom_id (str): DICOM ID to look up in final_dataset_fixed.csv
        
    Returns:
        Dict: Ground truth report data or None if not found
    """
    try:
        # Load the main dataset CSV file to get patient mapping
        dataset_csv_path = "../../final_dataset_fixed.csv"
        
        if not os.path.exists(dataset_csv_path):
            logger.error(f"Main dataset CSV not found: {dataset_csv_path}")
            return None
        
        # Read CSV and find the matching dicom_id
        import pandas as pd
        df = pd.read_csv(dataset_csv_path)
        
        # Find the row with matching dicom_id
        matching_rows = df[df['dicom_id'] == dicom_id]
        
        if matching_rows.empty:
            logger.warning(f"No matching entry found for dicom_id: {dicom_id}")
            return None
        
        # Get the first match (should be unique)
        row = matching_rows.iloc[0]
        patient_folder = row['patient_folder']
        study_id = row['study_id']
        
        # Construct the ground truth report path
        ground_truth_path = f"../../cleaned_reports/{patient_folder}/CXR-DICOM/s{study_id}.txt"
        
        if not os.path.exists(ground_truth_path):
            logger.warning(f"Ground truth report not found: {ground_truth_path}")
            return None
        
        # Load the ground truth report
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean the report text to extract final report
        cleaned_text = _clean_radiology_report(content)
        
        logger.info(f"Successfully loaded ground truth for {dicom_id} from {ground_truth_path}")
        
        return {
            'report_text': cleaned_text.strip(),
            'source_path': ground_truth_path,
            'file_type': 'radiology_report',
            'raw_text': content.strip(),
            'patient_folder': patient_folder,
            'study_id': study_id,
            'dicom_id': dicom_id
        }
        
    except Exception as e:
        logger.error(f"Error loading ground truth report for {dicom_id}: {e}")
        return None

def _clean_radiology_report(raw_text: str) -> str:
    """
    Clean radiology report text by extracting the final report section.
    
    Args:
        raw_text (str): Raw report text from .txt file
        
    Returns:
        str: Cleaned report text with final report section only
    """
    import re
    
    # Split by common section markers
    lines = raw_text.split('\n')
    
    # Look for FINAL REPORT section
    final_report_start = -1
    for i, line in enumerate(lines):
        if 'FINAL REPORT' in line.upper():
            final_report_start = i
            break
    
    # If no final report section found, look for main content after WET READ
    if final_report_start == -1:
        for i, line in enumerate(lines):
            if '___' in line and 'FINAL REPORT' not in line.upper():
                # Found separator line, content after this is likely the main report
                final_report_start = i + 1
                break
    
    # Extract text from final report section onward
    if final_report_start >= 0:
        report_lines = lines[final_report_start:]
    else:
        # No clear section found, use all content
        report_lines = lines
    
    # Join lines and clean up
    cleaned_text = '\n'.join(report_lines)
    
    # Remove common artifacts
    cleaned_text = re.sub(r'WET READ:.*?\n', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'WET READ VERSION.*?\n', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'^[\s_]*\n', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)  # Remove excessive blank lines
    
    return cleaned_text.strip()

def create_data_discovery() -> callable:
    """Factory function to create a data discovery function."""
    return discover_and_match_reports


