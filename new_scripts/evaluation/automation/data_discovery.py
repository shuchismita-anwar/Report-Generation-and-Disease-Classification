#!/usr/bin/env python3
"""
Data Discovery Module
====================
Automated discovery and cataloging of LLM outputs and ground truth reports.

This module provides functions to:
- Discover LLM-generated reports from real_analysis_results/
- Discover ground truth transcripts from MIMIC-Eye dataset
- Match reports by study IDs
- Validate discovered data

Author: Medical Report Evaluation System
Version: 1.0
"""

import json
import os
import glob
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def extract_study_id_from_filename(filepath: str) -> Optional[str]:
    """
    Extract study ID from LLM analysis filename.
    
    Args:
        filepath (str): Path to analysis file (e.g., analysis_002da0d9-ce49c30d-4dfcc1f8-746d2401-d8044d48.json)
        
    Returns:
        Optional[str]: Extracted study ID or None if not found
    """
    # Pattern for study ID: analysis_{study_id}.json
    pattern = r'analysis_([a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8})\.json'
    match = re.search(pattern, os.path.basename(filepath))
    
    if match:
        return match.group(1)
    
    # Fallback: try to extract any UUID-like pattern
    uuid_pattern = r'([a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8})'
    match = re.search(uuid_pattern, os.path.basename(filepath))
    
    return match.group(1) if match else None


def extract_patient_id(patient_dir: str) -> str:
    """
    Extract patient ID from patient directory path.
    
    Args:
        patient_dir (str): Path to patient directory
        
    Returns:
        str: Patient ID
    """
    return os.path.basename(patient_dir.rstrip('/'))


def extract_session_id(session_dir: str) -> str:
    """
    Extract session ID from session directory path.
    
    Args:
        session_dir (str): Path to session directory
        
    Returns:
        str: Session ID
    """
    return os.path.basename(session_dir.rstrip('/'))


def discover_llm_reports(base_path: str = "../../real_analysis_results/") -> Dict[str, Dict]:
    """
    Scan real_analysis_results/ for analysis_*.json files and catalog them.
    
    Args:
        base_path (str): Base path to LLM analysis results
        
    Returns:
        Dict[str, Dict]: Dictionary mapping study_id to file information
        
    Example:
        {
            "002da0d9-ce49c30d-4dfcc1f8-746d2401-d8044d48": {
                "file_path": "../../real_analysis_results/analysis_002da0d9-ce49c30d-4dfcc1f8-746d2401-d8044d48.json",
                "report_text": "Chest X-ray shows...",
                "timestamp": 1671234567.89,
                "file_size": 18432
            }
        }
    """
    logger = logging.getLogger(__name__)
    llm_index = {}
    
    # Resolve absolute path
    abs_base_path = os.path.abspath(base_path)
    logger.info(f"Discovering LLM reports in: {abs_base_path}")
    
    if not os.path.exists(abs_base_path):
        logger.warning(f"LLM reports directory does not exist: {abs_base_path}")
        return llm_index
    
    # Find all analysis_*.json files
    pattern = os.path.join(abs_base_path, "analysis_*.json")
    analysis_files = glob.glob(pattern)
    
    logger.info(f"Found {len(analysis_files)} potential LLM analysis files")
    
    for file_path in analysis_files:
        try:
            # Extract study_id from filename
            study_id = extract_study_id_from_filename(file_path)
            
            if not study_id:
                logger.warning(f"Could not extract study ID from filename: {file_path}")
                continue
            
            # Read and validate file content
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate study_id matches content
            content_study_id = data.get("study_id")
            if content_study_id and content_study_id != study_id:
                logger.warning(f"Study ID mismatch in {file_path}: filename={study_id}, content={content_study_id}")
                # Use content study_id as authoritative
                study_id = content_study_id
            
            # Extract report text from various possible locations
            report_text = None
            
            # Check medical_report.report_text (primary location)
            if "medical_report" in data and isinstance(data["medical_report"], dict):
                report_text = data["medical_report"].get("report_text", "")
            # Check llm_response.report (secondary location)
            elif "llm_response" in data and isinstance(data["llm_response"], dict):
                report_text = data["llm_response"].get("report", "")
            # Check analysis.report (fallback)
            elif "analysis" in data and isinstance(data["analysis"], dict):
                report_text = data["analysis"].get("report", "")
            # Check direct report field (fallback)
            elif "report" in data:
                report_text = data["report"]
            else:
                logger.warning(f"No report text found in {file_path}")
                continue
            
            if not report_text or not report_text.strip():
                logger.warning(f"Empty report text in {file_path}")
                continue
            
            # Store in index
            llm_index[study_id] = {
                "file_path": file_path,
                "report_text": report_text.strip(),
                "timestamp": os.path.getmtime(file_path),
                "file_size": os.path.getsize(file_path),
                "raw_data": data  # Store full data for debugging
            }
            
            logger.debug(f"Indexed LLM report: {study_id} ({len(report_text)} chars)")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Successfully indexed {len(llm_index)} LLM reports")
    return llm_index


def discover_ground_truth_reports(base_path: str = "../../mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye/") -> Dict[str, Dict]:
    """
    Scan mimic-eye patient folders for transcription.txt files and catalog them.
    
    Args:
        base_path (str): Base path to MIMIC-Eye dataset
        
    Returns:
        Dict[str, Dict]: Dictionary mapping study_id to transcript information
        
    Example:
        {
            "fa771fa1-d9571d07-bff8f655-327734a7-6e10b29d": {
                "file_path": "../../mimic-eye/.../transcription.txt",
                "transcript": "The heart size is normal...",
                "patient_id": "patient_10002428",
                "session_id": "P300R081391"
            }
        }
    """
    logger = logging.getLogger(__name__)
    gt_index = {}
    
    # Resolve absolute path
    abs_base_path = os.path.abspath(base_path)
    logger.info(f"Discovering ground truth reports in: {abs_base_path}")
    
    if not os.path.exists(abs_base_path):
        logger.warning(f"Ground truth directory does not exist: {abs_base_path}")
        return gt_index
    
    # Find all patient directories
    patient_pattern = os.path.join(abs_base_path, "patient_*")
    patient_dirs = glob.glob(patient_pattern)
    
    logger.info(f"Found {len(patient_dirs)} patient directories")
    
    for patient_dir in patient_dirs:
        if not os.path.isdir(patient_dir):
            continue
            
        patient_id = extract_patient_id(patient_dir)
        reflacx_path = os.path.join(patient_dir, "REFLACX", "main_data")
        
        if not os.path.exists(reflacx_path):
            logger.debug(f"No REFLACX/main_data in {patient_id}")
            continue
        
        # Find session directories
        session_dirs = glob.glob(os.path.join(reflacx_path, "*"))
        session_dirs = [d for d in session_dirs if os.path.isdir(d)]
        
        for session_dir in session_dirs:
            session_id = extract_session_id(session_dir)
            transcription_file = os.path.join(session_dir, "transcription.txt")
            
            if not os.path.exists(transcription_file):
                logger.debug(f"No transcription.txt in {patient_id}/{session_id}")
                continue
            
            try:
                # Extract study_id from norm_fixations CSV filename in the same directory
                study_id = None
                norm_fixations_files = glob.glob(os.path.join(session_dir, "norm_fixations_*.csv"))
                if norm_fixations_files:
                    # Extract study ID from filename like: norm_fixations_fa771fa1-d9571d07-bff8f655-327734a7-6e10b29d.csv
                    norm_file = norm_fixations_files[0]
                    filename = os.path.basename(norm_file)
                    # Pattern: norm_fixations_{study_id}.csv
                    match = re.search(r'norm_fixations_([a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8})\.csv', filename)
                    if match:
                        study_id = match.group(1)
                
                if not study_id:
                    logger.warning(f"No study_id found in {session_dir}")
                    continue
                
                # Read the actual radiologist report from transcription.txt
                with open(transcription_file, 'r', encoding='utf-8') as f:
                    transcript_text = f.read().strip()
                
                if not transcript_text:
                    logger.warning(f"Empty transcription in {transcription_file}")
                    continue
                
                # Store in index
                gt_index[study_id] = {
                    "file_path": transcription_file,
                    "transcript": transcript_text,
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "raw_data": {"transcription": transcript_text}  # Store for consistency
                }
                
                logger.debug(f"Indexed ground truth: {study_id} ({patient_id}/{session_id})")
                
            except Exception as e:
                logger.error(f"Error processing {transcription_file}: {e}")
    
    logger.info(f"Successfully indexed {len(gt_index)} ground truth reports")
    return gt_index


def create_matched_pairs(llm_index: Dict[str, Dict], gt_index: Dict[str, Dict]) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Match LLM reports with ground truth by study_id.
    
    Args:
        llm_index (Dict[str, Dict]): LLM reports index
        gt_index (Dict[str, Dict]): Ground truth reports index
        
    Returns:
        Tuple[List[Dict], List[str], List[str]]: (matched_pairs, unmatched_llm, unmatched_gt)
        
    Example matched pair:
        {
            "study_id": "002da0d9-ce49c30d-4dfcc1f8-746d2401-d8044d48",
            "llm_report": "Chest X-ray shows...",
            "ground_truth": "The heart size is normal...",
            "llm_file": "path/to/analysis.json",
            "gt_file": "path/to/transcript.json",
            "metadata": {...}
        }
    """
    logger = logging.getLogger(__name__)
    
    matched_pairs = []
    unmatched_llm = []
    unmatched_gt = []
    
    # Find matches
    for study_id in llm_index:
        if study_id in gt_index:
            llm_data = llm_index[study_id]
            gt_data = gt_index[study_id]
            
            matched_pair = {
                "study_id": study_id,
                "llm_report": llm_data["report_text"],
                "ground_truth": gt_data["transcript"],
                "llm_file": llm_data["file_path"],
                "gt_file": gt_data["file_path"],
                "metadata": {
                    "patient_id": gt_data["patient_id"],
                    "session_id": gt_data["session_id"],
                    "llm_timestamp": llm_data["timestamp"],
                    "llm_file_size": llm_data["file_size"],
                    "llm_report_length": len(llm_data["report_text"]),
                    "gt_transcript_length": len(gt_data["transcript"])
                }
            }
            
            matched_pairs.append(matched_pair)
            logger.debug(f"Matched: {study_id}")
        else:
            unmatched_llm.append(study_id)
            logger.debug(f"Unmatched LLM: {study_id}")
    
    # Find unmatched ground truth
    for study_id in gt_index:
        if study_id not in llm_index:
            unmatched_gt.append(study_id)
            logger.debug(f"Unmatched GT: {study_id}")
    
    logger.info(f"Matching results: {len(matched_pairs)} matched, {len(unmatched_llm)} unmatched LLM, {len(unmatched_gt)} unmatched GT")
    
    return matched_pairs, unmatched_llm, unmatched_gt


def discover_and_match_reports(llm_path: str = "../../real_analysis_results/", 
                              gt_path: str = "../../mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye/") -> Tuple[List[Dict], List[str], List[str]]:
    """
    Complete discovery and matching pipeline.
    
    Args:
        llm_path (str): Path to LLM analysis results
        gt_path (str): Path to MIMIC-Eye ground truth
        
    Returns:
        Tuple[List[Dict], List[str], List[str]]: (matched_pairs, unmatched_llm, unmatched_gt)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting complete discovery and matching pipeline")
    
    # Phase 1: Discover LLM reports
    logger.info("Phase 1: Discovering LLM reports...")
    llm_index = discover_llm_reports(llm_path)
    
    # Phase 2: Discover ground truth reports
    logger.info("Phase 2: Discovering ground truth reports...")
    gt_index = discover_ground_truth_reports(gt_path)
    
    # Phase 3: Create matched pairs
    logger.info("Phase 3: Creating matched pairs...")
    matched_pairs, unmatched_llm, unmatched_gt = create_matched_pairs(llm_index, gt_index)
    
    # Summary
    total_llm = len(llm_index)
    total_gt = len(gt_index)
    total_matched = len(matched_pairs)
    
    logger.info(f"Discovery complete: {total_llm} LLM reports, {total_gt} GT reports, {total_matched} matched pairs")
    
    if total_llm > 0:
        match_rate = total_matched / total_llm * 100
        logger.info(f"LLM match rate: {match_rate:.1f}%")
    
    if total_gt > 0:
        coverage_rate = total_matched / total_gt * 100
        logger.info(f"GT coverage rate: {coverage_rate:.1f}%")
    
    return matched_pairs, unmatched_llm, unmatched_gt


def validate_discovered_data(matched_pairs: List[Dict]) -> Dict[str, any]:
    """
    Validate the quality and consistency of discovered data.
    
    Args:
        matched_pairs (List[Dict]): List of matched report pairs
        
    Returns:
        Dict[str, any]: Validation results and statistics
    """
    logger = logging.getLogger(__name__)
    
    validation_results = {
        "total_pairs": len(matched_pairs),
        "valid_pairs": 0,
        "issues": [],
        "statistics": {
            "avg_llm_length": 0,
            "avg_gt_length": 0,
            "min_llm_length": float('inf'),
            "max_llm_length": 0,
            "min_gt_length": float('inf'),
            "max_gt_length": 0
        }
    }
    
    if not matched_pairs:
        validation_results["issues"].append("No matched pairs to validate")
        return validation_results
    
    llm_lengths = []
    gt_lengths = []
    
    for i, pair in enumerate(matched_pairs):
        try:
            # Check required fields
            required_fields = ["study_id", "llm_report", "ground_truth"]
            missing_fields = [field for field in required_fields if not pair.get(field)]
            
            if missing_fields:
                validation_results["issues"].append(f"Pair {i}: Missing fields {missing_fields}")
                continue
            
            # Check text lengths
            llm_len = len(pair["llm_report"])
            gt_len = len(pair["ground_truth"])
            
            if llm_len < 10:
                validation_results["issues"].append(f"Pair {i}: LLM report too short ({llm_len} chars)")
            
            if gt_len < 10:
                validation_results["issues"].append(f"Pair {i}: Ground truth too short ({gt_len} chars)")
            
            llm_lengths.append(llm_len)
            gt_lengths.append(gt_len)
            
            validation_results["valid_pairs"] += 1
            
        except Exception as e:
            validation_results["issues"].append(f"Pair {i}: Validation error - {e}")
    
    # Calculate statistics
    if llm_lengths:
        validation_results["statistics"]["avg_llm_length"] = sum(llm_lengths) / len(llm_lengths)
        validation_results["statistics"]["min_llm_length"] = min(llm_lengths)
        validation_results["statistics"]["max_llm_length"] = max(llm_lengths)
    
    if gt_lengths:
        validation_results["statistics"]["avg_gt_length"] = sum(gt_lengths) / len(gt_lengths)
        validation_results["statistics"]["min_gt_length"] = min(gt_lengths)
        validation_results["statistics"]["max_gt_length"] = max(gt_lengths)
    
    logger.info(f"Validation complete: {validation_results['valid_pairs']}/{validation_results['total_pairs']} valid pairs")
    
    if validation_results["issues"]:
        logger.warning(f"Found {len(validation_results['issues'])} validation issues")
    
    return validation_results


if __name__ == "__main__":
    """Test the discovery functions when run directly."""
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🔍 Testing data discovery functions...")
    
    try:
        matched_pairs, unmatched_llm, unmatched_gt = discover_and_match_reports()
        
        print(f"\n📊 Discovery Results:")
        print(f"✅ Matched pairs: {len(matched_pairs)}")
        print(f"⚠️  Unmatched LLM: {len(unmatched_llm)}")
        print(f"⚠️  Unmatched GT: {len(unmatched_gt)}")
        
        if matched_pairs:
            print(f"\n📋 Sample matched pairs:")
            for i, pair in enumerate(matched_pairs[:3]):
                print(f"  {i+1}. Study ID: {pair['study_id']}")
                print(f"     LLM: {pair['llm_report'][:100]}...")
                print(f"     GT:  {pair['ground_truth'][:100]}...")
        
        # Validate data
        validation = validate_discovered_data(matched_pairs)
        print(f"\n✅ Validation: {validation['valid_pairs']}/{validation['total_pairs']} valid pairs")
        
        if validation["issues"]:
            print(f"⚠️  Issues found: {len(validation['issues'])}")
            for issue in validation["issues"][:5]:  # Show first 5 issues
                print(f"   - {issue}")
        
    except Exception as e:
        print(f"❌ Discovery test failed: {e}")
        import traceback
        traceback.print_exc() 