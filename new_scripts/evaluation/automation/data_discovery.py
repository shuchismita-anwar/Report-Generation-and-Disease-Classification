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
    llm_path: str = "../../real_analysis_results/",
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
    """Load dataset CSV files with all the metadata."""
    dataset_files = []
    
    if use_all_splits:
        # Load all splits for comprehensive coverage
        split_files = [
            "../../dataset_splits/train.csv",
            "../../dataset_splits/val.csv", 
            "../../dataset_splits/test.csv"
        ]
    else:
        # Just validation set
        split_files = ["../../dataset_splits/val.csv"]
    
    combined_df = pd.DataFrame()
    
    for csv_file in split_files:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                logger.info(f"Loaded {len(df)} entries from {csv_file}")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
        else:
            logger.warning(f"CSV file not found: {csv_file}")
    
    if combined_df.empty:
        logger.error("No dataset files could be loaded!")
        return pd.DataFrame()
    
    # Clean and standardize column names
    if 'dicom_id' in combined_df.columns:
        combined_df['study_id'] = combined_df['dicom_id']
    
    logger.info(f"Combined dataset: {len(combined_df)} total entries")
    return combined_df

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
            
            # Extract transcript path from CSV
            transcript_path = row.get('transcript_path', '')
            
            if transcript_path and _validate_transcript_path(transcript_path):
                # Create matched pair
                matched_pair = {
                    'study_id': study_id,
                    'llm_report': llm_file,
                    'ground_truth': transcript_path,
                    'patient_folder': row.get('patient_folder', ''),
                    'has_reflacx': row.get('ifReflacx', False),
                    'has_eyegaze': row.get('ifEyegaze', False),
                    'condition': row.get('condition', ''),
                    'report_path': row.get('report_path', ''),
                    'image_path': row.get('image_path', '')
                }
                matched_pairs.append(matched_pair)
                matched_gt_ids.add(study_id)
                
                logger.debug(f"✅ Matched: {study_id}")
            else:
                logger.warning(f"⚠️ Invalid transcript path for {study_id}: {transcript_path}")
                unmatched_llm.append(study_id)
        else:
            logger.debug(f"❌ No CSV entry found for LLM report: {study_id}")
            unmatched_llm.append(study_id)
    
    # Find unmatched ground truth entries (those in CSV but not in LLM reports)
    for _, row in dataset_df.iterrows():
        study_id = row.get('dicom_id', '')
        if study_id and study_id not in matched_gt_ids and study_id not in llm_reports:
            transcript_path = row.get('transcript_path', '')
            if transcript_path and _validate_transcript_path(transcript_path):
                unmatched_gt.append(study_id)
    
    return matched_pairs, unmatched_llm, unmatched_gt

def _validate_transcript_path(transcript_path: str) -> bool:
    """Validate that a transcript path exists and is accessible."""
    if not transcript_path:
        return False
    
    # Try both absolute and relative paths
    if os.path.exists(transcript_path):
        return True
    
    # Try relative to current directory
    relative_path = os.path.join("..", "..", transcript_path)
    if os.path.exists(relative_path):
        return True
    
    return False

def load_ground_truth_report(transcript_path: str) -> Dict:
    """Load and parse a ground truth transcript file."""
    try:
        # Try multiple path variations
        paths_to_try = [
            transcript_path,
            os.path.join("..", "..", transcript_path),
            transcript_path.replace("\\", "/"),
            os.path.join("..", "..", transcript_path.replace("\\", "/"))
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                
                # Extract the full text
                if 'full_text' in content:
                    return {
                        'report_text': content['full_text'].strip(),
                        'time_stamped_text': content.get('time_stamped_text', []),
                        'source_path': path
                    }
                else:
                    logger.warning(f"No 'full_text' found in {path}")
                    return None
        
        logger.error(f"Transcript file not found: {transcript_path}")
        return None
        
    except Exception as e:
        logger.error(f"Error loading transcript {transcript_path}: {e}")
        return None

def create_data_discovery() -> callable:
    """Factory function to create a data discovery function."""
    return discover_and_match_reports


