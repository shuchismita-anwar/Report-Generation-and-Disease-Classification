#!/usr/bin/env python3
"""
Ground Truth Data Loader
========================
Loads and parses radiologist reports from MIMIC dataset sources.

This module handles:
- EyeGaze transcript files (audio_segmentation_transcripts)
- REFLACX transcript files (main_data)
- Different JSON structures and formats
- Error handling for missing/malformed data

Author: Medical Report Evaluation System
Version: 1.0
"""

import json
import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re


class GroundTruthLoader:
    """
    Loads ground truth radiologist reports from MIMIC dataset.
    
    Supports multiple transcript sources:
    - EyeGaze: patient_*/EyeGaze/audio_segmentation_transcripts/*/transcript.json
    - REFLACX: patient_*/REFLACX/main_data/*/transcript.json
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
        self.transcript_cache = {}
        
        # Validate base path
        if not self.mimic_base_path.exists():
            raise FileNotFoundError(f"MIMIC base path does not exist: {mimic_base_path}")
        
        self.logger.info(f"Initialized Ground Truth Loader with base path: {mimic_base_path}")
    
    def find_transcript_files(self, source_type: str = "both") -> Dict[str, List[str]]:
        """
        Find all transcript files in the MIMIC dataset.
        
        Args:
            source_type (str): Type of transcripts to find ("eyegaze", "reflacx", "both")
            
        Returns:
            Dict[str, List[str]]: Dictionary with source types as keys and file paths as values
            
        Raises:
            ValueError: If invalid source_type specified
        """
        valid_sources = {"eyegaze", "reflacx", "both"}
        if source_type not in valid_sources:
            raise ValueError(f"Invalid source_type '{source_type}'. Must be one of: {valid_sources}")
        
        transcript_files = {}
        
        # Find EyeGaze transcripts
        if source_type in ("eyegaze", "both"):
            eyegaze_pattern = str(self.mimic_base_path / "patient_*" / "EyeGaze" / "audio_segmentation_transcripts" / "*" / "transcript.json")
            eyegaze_files = glob.glob(eyegaze_pattern)
            transcript_files["eyegaze"] = sorted(eyegaze_files)
            self.logger.info(f"Found {len(eyegaze_files)} EyeGaze transcript files")
        
        # Find REFLACX transcripts
        if source_type in ("reflacx", "both"):
            reflacx_pattern = str(self.mimic_base_path / "patient_*" / "REFLACX" / "main_data" / "*" / "transcript.json")
            reflacx_files = glob.glob(reflacx_pattern)
            transcript_files["reflacx"] = sorted(reflacx_files)
            self.logger.info(f"Found {len(reflacx_files)} REFLACX transcript files")
        
        return transcript_files
    
    def extract_image_id_from_path(self, transcript_path: str) -> Optional[str]:
        """
        Extract image ID from transcript file path.
        
        Args:
            transcript_path (str): Path to transcript file
            
        Returns:
            Optional[str]: Image ID if found, None otherwise
        """
        # Pattern to match image IDs (format: xxxxxxxx-xxxxxxxx-xxxxxxxx-xxxxxxxx-xxxxxxxx)
        image_id_pattern = r'([a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8}-[a-f0-9]{8})'
        
        match = re.search(image_id_pattern, transcript_path)
        if match:
            return match.group(1)
        
        self.logger.warning(f"Could not extract image ID from path: {transcript_path}")
        return None
    
    def load_single_transcript(self, transcript_path: str) -> Optional[Dict]:
        """
        Load a single transcript file.
        
        Args:
            transcript_path (str): Path to the transcript JSON file
            
        Returns:
            Optional[Dict]: Parsed transcript data or None if failed
        """
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract image ID from path
            image_id = self.extract_image_id_from_path(transcript_path)
            
            # Determine source type
            source_type = "eyegaze" if "EyeGaze" in transcript_path else "reflacx"
            
            # Create structured output
            transcript_data = {
                "image_id": image_id,
                "source_type": source_type,
                "file_path": transcript_path,
                "raw_data": data,
                "full_text": data.get("full_text", ""),
                "has_timestamps": "time_stamped_text" in data,
                "timestamp_count": len(data.get("time_stamped_text", [])),
                "text_length": len(data.get("full_text", "")),
                "loaded_at": Path(transcript_path).stat().st_mtime
            }
            
            self.logger.debug(f"Loaded transcript: {image_id} ({source_type}) - {transcript_data['text_length']} chars")
            return transcript_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {transcript_path}: {e}")
            return None
        except FileNotFoundError:
            self.logger.error(f"Transcript file not found: {transcript_path}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error loading {transcript_path}: {e}")
            return None
    
    def load_transcripts_batch(self, transcript_paths: List[str], 
                              max_errors: int = 10) -> Tuple[Dict[str, Dict], List[str]]:
        """
        Load multiple transcript files in batch.
        
        Args:
            transcript_paths (List[str]): List of transcript file paths
            max_errors (int): Maximum number of errors before stopping
            
        Returns:
            Tuple[Dict[str, Dict], List[str]]: (loaded_transcripts, failed_paths)
        """
        loaded_transcripts = {}
        failed_paths = []
        error_count = 0
        
        self.logger.info(f"Loading {len(transcript_paths)} transcript files...")
        
        for i, path in enumerate(transcript_paths):
            if error_count >= max_errors:
                self.logger.error(f"Stopping batch load: too many errors ({error_count})")
                break
            
            transcript_data = self.load_single_transcript(path)
            
            if transcript_data:
                image_id = transcript_data["image_id"]
                if image_id:
                    loaded_transcripts[image_id] = transcript_data
                else:
                    failed_paths.append(path)
                    error_count += 1
            else:
                failed_paths.append(path)
                error_count += 1
            
            # Progress logging
            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i + 1}/{len(transcript_paths)} files...")
        
        success_rate = len(loaded_transcripts) / len(transcript_paths) * 100
        self.logger.info(f"Batch load complete: {len(loaded_transcripts)} successful, "
                        f"{len(failed_paths)} failed ({success_rate:.1f}% success rate)")
        
        return loaded_transcripts, failed_paths
    
    def get_transcript_statistics(self, transcripts: Dict[str, Dict]) -> Dict:
        """
        Generate statistics about loaded transcripts.
        
        Args:
            transcripts (Dict[str, Dict]): Dictionary of loaded transcripts
            
        Returns:
            Dict: Statistics about the transcripts
        """
        if not transcripts:
            return {"total": 0, "sources": {}, "text_lengths": {}}
        
        stats = {
            "total": len(transcripts),
            "sources": {"eyegaze": 0, "reflacx": 0},
            "text_lengths": {"min": float('inf'), "max": 0, "mean": 0, "total_chars": 0},
            "has_timestamps": 0,
            "empty_texts": 0
        }
        
        text_lengths = []
        
        for transcript in transcripts.values():
            # Count by source
            source = transcript["source_type"]
            stats["sources"][source] += 1
            
            # Text length analysis
            text_len = transcript["text_length"]
            text_lengths.append(text_len)
            stats["text_lengths"]["total_chars"] += text_len
            
            if text_len == 0:
                stats["empty_texts"] += 1
            
            # Timestamp analysis
            if transcript["has_timestamps"]:
                stats["has_timestamps"] += 1
        
        if text_lengths:
            stats["text_lengths"]["min"] = min(text_lengths)
            stats["text_lengths"]["max"] = max(text_lengths)
            stats["text_lengths"]["mean"] = sum(text_lengths) / len(text_lengths)
        
        return stats
    
    def load_all_transcripts(self, source_type: str = "both", 
                           cache_results: bool = True) -> Dict[str, Dict]:
        """
        Load all available transcripts from the dataset.
        
        Args:
            source_type (str): Type of transcripts to load ("eyegaze", "reflacx", "both")
            cache_results (bool): Whether to cache results for future use
            
        Returns:
            Dict[str, Dict]: Dictionary mapping image IDs to transcript data
        """
        cache_key = f"{source_type}_{self.mimic_base_path}"
        
        # Check cache first
        if cache_results and cache_key in self.transcript_cache:
            self.logger.info("Using cached transcript data")
            return self.transcript_cache[cache_key]
        
        # Find all transcript files
        transcript_files = self.find_transcript_files(source_type)
        
        # Combine all files
        all_files = []
        for source, files in transcript_files.items():
            all_files.extend(files)
        
        if not all_files:
            self.logger.warning("No transcript files found")
            return {}
        
        # Load transcripts
        transcripts, failed = self.load_transcripts_batch(all_files)
        
        # Generate and log statistics
        stats = self.get_transcript_statistics(transcripts)
        self.logger.info(f"Transcript loading statistics: {stats}")
        
        # Cache results
        if cache_results:
            self.transcript_cache[cache_key] = transcripts
        
        return transcripts


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
        
        # Find transcript files
        files = loader.find_transcript_files("both")
        print(f"✅ Found transcript files: {sum(len(f) for f in files.values())} total")
        
        # Test loading a single file if available
        all_files = []
        for source_files in files.values():
            all_files.extend(source_files)
        
        if all_files:
            # Test single file load
            single_transcript = loader.load_single_transcript(all_files[0])
            if single_transcript:
                print(f"✅ Single transcript loaded: {single_transcript['image_id']}")
                print(f"   Text length: {single_transcript['text_length']} characters")
                print(f"   Source: {single_transcript['source_type']}")
            else:
                print("❌ Failed to load single transcript")
        
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