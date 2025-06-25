#!/usr/bin/env python3
"""
Ground Truth Data Loader
========================
Loads and parses radiology reports from MIMIC dataset sources.

This module handles:
- Radiology report files (CXR-DICOM/*.txt)
- Text cleaning and extraction from report files
- Error handling for missing/malformed data

Author: Medical Report Evaluation System
Version: 2.0 - Updated for Radiology Reports
"""

import json
import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re


def _clean_radiology_report(content: str) -> str:
    """Clean radiology report text by extracting final report section."""
    if not content.strip():
        return ""
    
    # Split by lines and find final report section
    lines = content.split('\n')
    
    # Look for "FINAL REPORT" section
    final_report_start = -1
    for i, line in enumerate(lines):
        if re.search(r'FINAL\s+REPORT', line, re.IGNORECASE):
            final_report_start = i
            break
    
    # If no "FINAL REPORT" found, look for "IMPRESSION" or main content
    if final_report_start == -1:
        for i, line in enumerate(lines):
            if re.search(r'(IMPRESSION|FINDINGS):', line, re.IGNORECASE):
                final_report_start = max(0, i - 2)  # Include some context
                break
    
    # If still not found, skip WET READ and take the rest
    if final_report_start == -1:
        for i, line in enumerate(lines):
            if not re.search(r'WET\s+READ', line, re.IGNORECASE) and line.strip():
                final_report_start = i
                break
    
    # Extract content from final report section
    if final_report_start >= 0:
        report_lines = lines[final_report_start:]
    else:
        report_lines = lines
    
    # Clean up the content
    cleaned_lines = []
    for line in report_lines:
        line = line.strip()
        # Skip empty lines and obvious artifacts
        if line and not re.match(r'^[_\-=]+$', line):
            # Remove timestamps and other artifacts
            line = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?\s*[AP]M', '', line)
            line = re.sub(r'WET\s+READ:', '', line, flags=re.IGNORECASE)
            line = line.strip()
            if line:
                cleaned_lines.append(line)
    
    # Join lines and clean up extra whitespace
    cleaned_text = ' '.join(cleaned_lines)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()


class GroundTruthLoader:
    """
    Loads ground truth radiology reports from MIMIC dataset.
    
    Handles loading official radiology reports from CXR-DICOM folders:
    - patient_*/CXR-DICOM/*.txt
    """
    
    def __init__(self, mimic_base_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the Ground Truth Loader.
        
        Args:
            mimic_base_path (str): Base path to MIMIC dataset
            logger (Optional[logging.Logger]): Logger instance
        """
        self.mimic_base_path = Path(mimic_base_path)
        self.logger = logger or logging.getLogger(__name__)
        self.report_cache = {}
        
        # Validate base path
        if not self.mimic_base_path.exists():
            raise FileNotFoundError(f"MIMIC base path does not exist: {mimic_base_path}")
        
        self.logger.info(f"Initialized Ground Truth Loader with base path: {mimic_base_path}")
    
    def find_report_files(self, source_type: str = "radiology") -> Dict[str, List[str]]:
        """
        Find all radiology report files in the MIMIC dataset.
        
        Args:
            source_type (str): Type of reports to find ("radiology")
            
        Returns:
            Dict[str, List[str]]: Dictionary with source types as keys and file paths as values
        """
        report_files = {}
        
        # Find radiology reports
        if source_type == "radiology":
            report_pattern = str(self.mimic_base_path / "patient_*" / "CXR-DICOM" / "*.txt")
            report_file_list = glob.glob(report_pattern)
            report_files["radiology"] = sorted(report_file_list)
            self.logger.info(f"Found {len(report_file_list)} radiology report files")
        
        return report_files
    
    def extract_image_id_from_path(self, report_path: str) -> Optional[str]:
        """
        Extract image ID from radiology report file path.
        
        Args:
            report_path (str): Path to report file
            
        Returns:
            Optional[str]: Image ID if found, None otherwise
        """
        # Extract filename without extension (e.g., s56466802.txt -> s56466802)
        filename = Path(report_path).stem
        
        # Pattern to match study IDs (format: sxxxxxxxx)
        study_id_pattern = r's(\d{8})'
        
        match = re.search(study_id_pattern, filename)
        if match:
            return filename  # Return the full filename as ID
        
        # Also try to match longer image IDs if present in path
        image_id_pattern = r'([a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8})'
        match = re.search(image_id_pattern, report_path)
        if match:
            return match.group(1)
        
        self.logger.warning(f"Could not extract image ID from path: {report_path}")
        return filename  # Fallback to filename
    
    def load_single_report(self, report_path: str) -> Optional[Dict]:
        """
        Load a single radiology report file.
        
        Args:
            report_path (str): Path to the report .txt file
            
        Returns:
            Optional[Dict]: Parsed report data or None if failed
        """
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            # Extract image ID from path
            image_id = self.extract_image_id_from_path(report_path)
            
            # Clean the report text
            cleaned_text = _clean_radiology_report(raw_content)
            
            # Create structured output
            report_data = {
                "image_id": image_id,
                "source_type": "radiology",
                "file_path": report_path,
                "raw_content": raw_content,
                "full_text": cleaned_text,
                "has_timestamps": False,  # Reports don't have timestamps like transcripts
                "timestamp_count": 0,
                "text_length": len(cleaned_text),
                "raw_text_length": len(raw_content),
                "loaded_at": Path(report_path).stat().st_mtime
            }
            
            self.logger.debug(f"Loaded report: {image_id} - {report_data['text_length']} chars (cleaned)")
            return report_data
            
        except FileNotFoundError:
            self.logger.error(f"Report file not found: {report_path}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error loading {report_path}: {e}")
            return None
    
    def load_reports_batch(self, report_paths: List[str], 
                          max_errors: int = 10) -> Tuple[Dict[str, Dict], List[str]]:
        """
        Load multiple radiology report files in batch.
        
        Args:
            report_paths (List[str]): List of report file paths
            max_errors (int): Maximum number of errors before stopping
            
        Returns:
            Tuple[Dict[str, Dict], List[str]]: (loaded_reports, failed_paths)
        """
        loaded_reports = {}
        failed_paths = []
        error_count = 0
        
        self.logger.info(f"Loading {len(report_paths)} radiology report files...")
        
        for i, path in enumerate(report_paths):
            if error_count >= max_errors:
                self.logger.error(f"Stopping batch load: too many errors ({error_count})")
                break
            
            report_data = self.load_single_report(path)
            
            if report_data:
                image_id = report_data["image_id"]
                if image_id:
                    loaded_reports[image_id] = report_data
                else:
                    failed_paths.append(path)
                    error_count += 1
            else:
                failed_paths.append(path)
                error_count += 1
            
            # Progress logging
            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i + 1}/{len(report_paths)} files...")
        
        success_rate = len(loaded_reports) / len(report_paths) * 100 if report_paths else 0
        self.logger.info(f"Batch load complete: {len(loaded_reports)} successful, "
                        f"{len(failed_paths)} failed ({success_rate:.1f}% success rate)")
        
        return loaded_reports, failed_paths
    
    def get_report_statistics(self, reports: Dict[str, Dict]) -> Dict:
        """
        Generate statistics about loaded reports.
        
        Args:
            reports (Dict[str, Dict]): Dictionary of loaded reports
            
        Returns:
            Dict: Statistics about the reports
        """
        if not reports:
            return {"total": 0, "sources": {}, "text_lengths": {}}
        
        stats = {
            "total": len(reports),
            "sources": {"radiology": len(reports)},
            "text_lengths": {"min": float('inf'), "max": 0, "mean": 0, "total_chars": 0},
            "empty_texts": 0,
            "cleaning_efficiency": {"total_raw_chars": 0, "total_cleaned_chars": 0}
        }
        
        text_lengths = []
        raw_lengths = []
        
        for report in reports.values():
            # Text length analysis
            text_len = report["text_length"]
            raw_len = report["raw_text_length"]
            
            text_lengths.append(text_len)
            raw_lengths.append(raw_len)
            
            stats["text_lengths"]["total_chars"] += text_len
            stats["cleaning_efficiency"]["total_raw_chars"] += raw_len
            stats["cleaning_efficiency"]["total_cleaned_chars"] += text_len
            
            if text_len == 0:
                stats["empty_texts"] += 1
        
        if text_lengths:
            stats["text_lengths"]["min"] = min(text_lengths)
            stats["text_lengths"]["max"] = max(text_lengths)
            stats["text_lengths"]["mean"] = sum(text_lengths) / len(text_lengths)
        
        # Calculate cleaning efficiency
        if stats["cleaning_efficiency"]["total_raw_chars"] > 0:
            efficiency = stats["cleaning_efficiency"]["total_cleaned_chars"] / stats["cleaning_efficiency"]["total_raw_chars"]
            stats["cleaning_efficiency"]["ratio"] = efficiency
        
        return stats
    
    def load_all_reports(self, source_type: str = "radiology", 
                        cache_results: bool = True) -> Dict[str, Dict]:
        """
        Load all available radiology reports from the dataset.
        
        Args:
            source_type (str): Type of reports to load ("radiology")
            cache_results (bool): Whether to cache results for future use
            
        Returns:
            Dict[str, Dict]: Dictionary mapping image IDs to report data
        """
        cache_key = f"{source_type}_{self.mimic_base_path}"
        
        # Check cache first
        if cache_results and cache_key in self.report_cache:
            self.logger.info("Using cached report data")
            return self.report_cache[cache_key]
        
        # Find all report files
        report_files = self.find_report_files(source_type)
        
        # Get all files
        all_files = []
        for source, files in report_files.items():
            all_files.extend(files)
        
        if not all_files:
            self.logger.warning("No radiology report files found")
            return {}
        
        # Load reports
        reports, failed = self.load_reports_batch(all_files)
        
        # Generate and log statistics
        stats = self.get_report_statistics(reports)
        self.logger.info(f"Report loading statistics: {stats}")
        
        # Cache results
        if cache_results:
            self.report_cache[cache_key] = reports
        
        return reports


def test_ground_truth_loader():
    """Test function for the Ground Truth Loader."""
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    # Test with MIMIC dataset path
    mimic_path = "../../mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye/"
    
    try:
        # Initialize loader
        loader = GroundTruthLoader(mimic_path, logger)
        
        # Find report files
        files = loader.find_report_files("radiology")
        print(f"✅ Found report files: {sum(len(f) for f in files.values())} total")
        
        # Test loading a single file if available
        all_files = []
        for source_files in files.values():
            all_files.extend(source_files)
        
        if all_files:
            # Test single file load
            single_report = loader.load_single_report(all_files[0])
            if single_report:
                print(f"✅ Single report loaded: {single_report['image_id']}")
                print(f"   Text length: {single_report['text_length']} characters")
                print(f"   Raw length: {single_report['raw_text_length']} characters")
                print(f"   Source: {single_report['source_type']}")
            else:
                print("❌ Failed to load single report")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    """Run basic tests when script is executed directly."""
    print("🧪 Testing Ground Truth Loader...")
    success = test_ground_truth_loader()
    
    if success:
        print("\n🎉 Ground Truth Loader tests passed!")
    else:
        print("\n❌ Some tests failed!") 