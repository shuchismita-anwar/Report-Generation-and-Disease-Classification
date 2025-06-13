#!/usr/bin/env python3
"""
Ground Truth Validator
======================
Validates ground truth radiologist reports for quality and completeness.

This module handles:
- Validation of transcript files and data quality
- Detection of missing, corrupted, or invalid reports
- Quality assessment of extracted text
- Generation of validation reports and statistics

Author: Medical Report Evaluation System
Version: 1.0
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import re
from datetime import datetime


class GroundTruthValidator:
    """
    Validates ground truth data quality and completeness.
    
    Provides comprehensive validation including file integrity,
    text quality assessment, and completeness checking.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Ground Truth Validator.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Validation criteria
        self.min_text_length = 10
        self.max_text_length = 10000
        self.min_word_count = 3
        self.max_word_count = 2000
        
        # Required fields for different data sources
        self.required_fields = {
            "eyegaze": ["full_text"],
            "reflacx": ["full_text"],
            "general": ["full_text"]
        }
        
        # Medical content indicators
        self.medical_indicators = [
            'chest', 'lung', 'heart', 'radiograph', 'x-ray', 'patient',
            'normal', 'abnormal', 'examination', 'findings', 'impression',
            'bilateral', 'cardiopulmonary', 'thoracic', 'pulmonary'
        ]
        
        # Validation results storage
        self.validation_results = {}
        self.validation_summary = {}
    
    def validate_single_file(self, file_path: str) -> Dict:
        """
        Validate a single transcript file.
        
        Args:
            file_path (str): Path to the transcript file
            
        Returns:
            Dict: Validation results for the file
        """
        validation_result = {
            "file_path": file_path,
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {},
            "content_info": {},
            "quality_scores": {}
        }
        
        try:
            # Check file existence and accessibility
            if not os.path.exists(file_path):
                validation_result["is_valid"] = False
                validation_result["errors"].append("file_not_found")
                return validation_result
            
            # Get file information
            file_stat = os.stat(file_path)
            validation_result["file_info"] = {
                "size_bytes": file_stat.st_size,
                "modified_time": file_stat.st_mtime,
                "readable": os.access(file_path, os.R_OK)
            }
            
            # Check if file is readable
            if not validation_result["file_info"]["readable"]:
                validation_result["is_valid"] = False
                validation_result["errors"].append("file_not_readable")
                return validation_result
            
            # Check file size
            if file_stat.st_size == 0:
                validation_result["is_valid"] = False
                validation_result["errors"].append("empty_file")
                return validation_result
            
            if file_stat.st_size > 1024 * 1024:  # 1MB limit
                validation_result["warnings"].append("large_file_size")
            
            # Load and validate JSON content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
            except json.JSONDecodeError as e:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"invalid_json: {str(e)}")
                return validation_result
            except UnicodeDecodeError as e:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"encoding_error: {str(e)}")
                return validation_result
            
            # Validate content structure
            content_validation = self._validate_content_structure(content, file_path)
            validation_result["content_info"] = content_validation
            
            if not content_validation["has_required_fields"]:
                validation_result["is_valid"] = False
                validation_result["errors"].extend(content_validation["missing_fields"])
            
            # Validate text quality if available
            if content_validation.get("full_text"):
                text_validation = self._validate_text_quality(content_validation["full_text"])
                validation_result["quality_scores"] = text_validation
                
                if not text_validation["is_valid_length"]:
                    validation_result["warnings"].append("invalid_text_length")
                
                if not text_validation["has_medical_content"]:
                    validation_result["warnings"].append("no_medical_content")
                
                if text_validation["has_artifacts"]:
                    validation_result["warnings"].append("text_artifacts_detected")
        
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"validation_error: {str(e)}")
            self.logger.error(f"Error validating {file_path}: {e}")
        
        return validation_result
    
    def _validate_content_structure(self, content: Dict, file_path: str) -> Dict:
        """
        Validate the structure of transcript content.
        
        Args:
            content (Dict): Loaded JSON content
            file_path (str): Path to the file (for source type detection)
            
        Returns:
            Dict: Content validation results
        """
        # Determine source type
        source_type = "eyegaze" if "EyeGaze" in file_path else "reflacx"
        
        validation_info = {
            "source_type": source_type,
            "has_required_fields": True,
            "missing_fields": [],
            "available_fields": list(content.keys()),
            "full_text": "",
            "text_length": 0,
            "has_timestamps": False,
            "timestamp_count": 0
        }
        
        # Check required fields
        required = self.required_fields.get(source_type, self.required_fields["general"])
        for field in required:
            if field not in content:
                validation_info["has_required_fields"] = False
                validation_info["missing_fields"].append(f"missing_{field}")
        
        # Extract text information
        if "full_text" in content:
            full_text = content["full_text"]
            if isinstance(full_text, str):
                validation_info["full_text"] = full_text
                validation_info["text_length"] = len(full_text)
            else:
                validation_info["missing_fields"].append("invalid_full_text_type")
        
        # Check for timestamps
        if "time_stamped_text" in content:
            timestamp_data = content["time_stamped_text"]
            if isinstance(timestamp_data, list):
                validation_info["has_timestamps"] = True
                validation_info["timestamp_count"] = len(timestamp_data)
            else:
                validation_info["missing_fields"].append("invalid_timestamp_format")
        
        return validation_info
    
    def _validate_text_quality(self, text: str) -> Dict:
        """
        Validate the quality of extracted text.
        
        Args:
            text (str): Text to validate
            
        Returns:
            Dict: Text quality validation results
        """
        quality_info = {
            "text_length": len(text),
            "word_count": len(text.split()) if text else 0,
            "is_valid_length": True,
            "has_medical_content": False,
            "has_artifacts": False,
            "completeness_score": 0.0,
            "quality_issues": []
        }
        
        # Length validation
        if len(text) < self.min_text_length:
            quality_info["is_valid_length"] = False
            quality_info["quality_issues"].append("text_too_short")
        elif len(text) > self.max_text_length:
            quality_info["is_valid_length"] = False
            quality_info["quality_issues"].append("text_too_long")
        
        # Word count validation
        word_count = quality_info["word_count"]
        if word_count < self.min_word_count:
            quality_info["quality_issues"].append("insufficient_words")
        elif word_count > self.max_word_count:
            quality_info["quality_issues"].append("excessive_words")
        
        if text:
            # Medical content detection
            text_lower = text.lower()
            medical_word_count = sum(1 for indicator in self.medical_indicators 
                                   if indicator in text_lower)
            quality_info["has_medical_content"] = medical_word_count >= 2
            quality_info["medical_word_count"] = medical_word_count
            
            # Artifact detection
            artifacts = []
            
            # Check for timestamp artifacts
            if re.search(r'\d{1,2}:\d{2}', text):
                artifacts.append("timestamps")
            
            # Check for JSON artifacts
            if any(char in text for char in ['{', '}', '[', ']']):
                artifacts.append("json_artifacts")
            
            # Check for speaker annotations
            if re.search(r'\b(doctor|speaker|voice)\s*\d*\s*:', text, re.IGNORECASE):
                artifacts.append("speaker_annotations")
            
            # Check for excessive repetition
            words = text_lower.split()
            if len(words) > 10:
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                max_freq = max(word_freq.values())
                if max_freq > len(words) * 0.3:
                    artifacts.append("excessive_repetition")
            
            quality_info["has_artifacts"] = len(artifacts) > 0
            quality_info["artifacts_detected"] = artifacts
            
            # Calculate completeness score
            completeness_factors = []
            
            # Length factor (0-1)
            length_factor = min(len(text) / 100, 1.0)  # Normalize to 100 chars
            completeness_factors.append(length_factor)
            
            # Medical content factor (0-1)
            medical_factor = min(medical_word_count / 5, 1.0)  # Normalize to 5 medical words
            completeness_factors.append(medical_factor)
            
            # Structure factor (has sentence endings)
            has_endings = bool(re.search(r'[.!?]', text))
            structure_factor = 1.0 if has_endings else 0.5
            completeness_factors.append(structure_factor)
            
            # Artifact penalty
            artifact_penalty = 1.0 - (len(artifacts) * 0.1)
            completeness_factors.append(max(artifact_penalty, 0.0))
            
            quality_info["completeness_score"] = sum(completeness_factors) / len(completeness_factors)
        
        return quality_info
    
    def validate_dataset(self, transcript_data: Dict[str, Dict]) -> Dict:
        """
        Validate an entire dataset of transcripts.
        
        Args:
            transcript_data (Dict[str, Dict]): Dictionary of transcript data
            
        Returns:
            Dict: Dataset validation results
        """
        self.logger.info(f"Validating dataset with {len(transcript_data)} transcripts...")
        
        dataset_validation = {
            "total_files": len(transcript_data),
            "valid_files": 0,
            "invalid_files": 0,
            "files_with_warnings": 0,
            "validation_timestamp": datetime.now().isoformat(),
            "detailed_results": {},
            "summary_statistics": {},
            "quality_distribution": {},
            "common_issues": {}
        }
        
        # Validate each file
        for image_id, transcript_info in transcript_data.items():
            file_path = transcript_info.get("file_path", "")
            
            if file_path:
                validation_result = self.validate_single_file(file_path)
            else:
                # Validate the loaded data directly
                validation_result = self._validate_loaded_data(image_id, transcript_info)
            
            dataset_validation["detailed_results"][image_id] = validation_result
            
            # Update counters
            if validation_result["is_valid"]:
                dataset_validation["valid_files"] += 1
            else:
                dataset_validation["invalid_files"] += 1
            
            if validation_result["warnings"]:
                dataset_validation["files_with_warnings"] += 1
        
        # Generate summary statistics
        dataset_validation["summary_statistics"] = self._generate_summary_statistics(
            dataset_validation["detailed_results"]
        )
        
        # Generate quality distribution
        dataset_validation["quality_distribution"] = self._generate_quality_distribution(
            dataset_validation["detailed_results"]
        )
        
        # Identify common issues
        dataset_validation["common_issues"] = self._identify_common_issues(
            dataset_validation["detailed_results"]
        )
        
        # Store results
        self.validation_results = dataset_validation
        
        # Log summary
        self._log_validation_summary(dataset_validation)
        
        return dataset_validation
    
    def _validate_loaded_data(self, image_id: str, transcript_info: Dict) -> Dict:
        """
        Validate already loaded transcript data.
        
        Args:
            image_id (str): Image ID
            transcript_info (Dict): Loaded transcript information
            
        Returns:
            Dict: Validation results
        """
        validation_result = {
            "image_id": image_id,
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "content_info": {},
            "quality_scores": {}
        }
        
        try:
            # Check basic structure
            if "full_text" not in transcript_info and "raw_data" not in transcript_info:
                validation_result["is_valid"] = False
                validation_result["errors"].append("missing_text_data")
                return validation_result
            
            # Extract text
            text = ""
            if "full_text" in transcript_info:
                text = transcript_info["full_text"]
            elif "raw_data" in transcript_info and "full_text" in transcript_info["raw_data"]:
                text = transcript_info["raw_data"]["full_text"]
            
            # Validate text quality
            if text:
                text_validation = self._validate_text_quality(text)
                validation_result["quality_scores"] = text_validation
                
                if not text_validation["is_valid_length"]:
                    validation_result["warnings"].append("invalid_text_length")
                
                if not text_validation["has_medical_content"]:
                    validation_result["warnings"].append("no_medical_content")
            else:
                validation_result["is_valid"] = False
                validation_result["errors"].append("empty_text")
            
            # Check metadata
            validation_result["content_info"] = {
                "source_type": transcript_info.get("source_type", "unknown"),
                "text_length": transcript_info.get("text_length", len(text)),
                "has_timestamps": transcript_info.get("has_timestamps", False)
            }
        
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"validation_error: {str(e)}")
        
        return validation_result
    
    def _generate_summary_statistics(self, detailed_results: Dict) -> Dict:
        """Generate summary statistics from detailed validation results."""
        if not detailed_results:
            return {}
        
        # Collect metrics
        text_lengths = []
        word_counts = []
        completeness_scores = []
        medical_content_count = 0
        
        for result in detailed_results.values():
            quality_scores = result.get("quality_scores", {})
            
            if "text_length" in quality_scores:
                text_lengths.append(quality_scores["text_length"])
            
            if "word_count" in quality_scores:
                word_counts.append(quality_scores["word_count"])
            
            if "completeness_score" in quality_scores:
                completeness_scores.append(quality_scores["completeness_score"])
            
            if quality_scores.get("has_medical_content", False):
                medical_content_count += 1
        
        stats = {
            "text_length": self._calculate_stats(text_lengths),
            "word_count": self._calculate_stats(word_counts),
            "completeness_score": self._calculate_stats(completeness_scores),
            "medical_content_percentage": (medical_content_count / len(detailed_results)) * 100 if detailed_results else 0
        }
        
        return stats
    
    def _calculate_stats(self, values: List[float]) -> Dict:
        """Calculate basic statistics for a list of values."""
        if not values:
            return {"count": 0}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2]
        }
    
    def _generate_quality_distribution(self, detailed_results: Dict) -> Dict:
        """Generate quality distribution from validation results."""
        distribution = {
            "excellent": 0,    # completeness_score >= 0.9
            "good": 0,         # completeness_score >= 0.7
            "fair": 0,         # completeness_score >= 0.5
            "poor": 0          # completeness_score < 0.5
        }
        
        for result in detailed_results.values():
            score = result.get("quality_scores", {}).get("completeness_score", 0)
            
            if score >= 0.9:
                distribution["excellent"] += 1
            elif score >= 0.7:
                distribution["good"] += 1
            elif score >= 0.5:
                distribution["fair"] += 1
            else:
                distribution["poor"] += 1
        
        return distribution
    
    def _identify_common_issues(self, detailed_results: Dict) -> Dict:
        """Identify common issues across the dataset."""
        issue_counts = {}
        
        for result in detailed_results.values():
            # Count errors
            for error in result.get("errors", []):
                issue_counts[f"error_{error}"] = issue_counts.get(f"error_{error}", 0) + 1
            
            # Count warnings
            for warning in result.get("warnings", []):
                issue_counts[f"warning_{warning}"] = issue_counts.get(f"warning_{warning}", 0) + 1
            
            # Count quality issues
            quality_issues = result.get("quality_scores", {}).get("quality_issues", [])
            for issue in quality_issues:
                issue_counts[f"quality_{issue}"] = issue_counts.get(f"quality_{issue}", 0) + 1
        
        # Sort by frequency
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_issues[:10])  # Top 10 most common issues
    
    def _log_validation_summary(self, validation_results: Dict) -> None:
        """Log validation summary."""
        total = validation_results["total_files"]
        valid = validation_results["valid_files"]
        invalid = validation_results["invalid_files"]
        warnings = validation_results["files_with_warnings"]
        
        self.logger.info(f"Dataset validation complete:")
        self.logger.info(f"  Total files: {total}")
        self.logger.info(f"  Valid files: {valid} ({valid/total*100:.1f}%)")
        self.logger.info(f"  Invalid files: {invalid} ({invalid/total*100:.1f}%)")
        self.logger.info(f"  Files with warnings: {warnings} ({warnings/total*100:.1f}%)")
        
        # Log quality distribution
        quality_dist = validation_results["quality_distribution"]
        self.logger.info(f"  Quality distribution:")
        for quality, count in quality_dist.items():
            percentage = (count / total) * 100 if total > 0 else 0
            self.logger.info(f"    {quality.title()}: {count} ({percentage:.1f}%)")
        
        # Log common issues
        common_issues = validation_results["common_issues"]
        if common_issues:
            self.logger.info(f"  Most common issues:")
            for issue, count in list(common_issues.items())[:5]:
                self.logger.info(f"    {issue}: {count} occurrences")
    
    def export_validation_report(self, output_path: str) -> None:
        """
        Export validation results to a JSON file.
        
        Args:
            output_path (str): Path to save the validation report
        """
        if not self.validation_results:
            self.logger.warning("No validation results to export")
            return
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Validation report exported to: {output_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to export validation report: {e}")


def test_ground_truth_validator():
    """Test function for the Ground Truth Validator."""
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    # Create test data
    test_data = {
        "valid_sample": {
            "source_type": "eyegaze",
            "file_path": "test_file.json",
            "full_text": "Normal chest radiograph. The lungs are clear bilaterally. Heart size is within normal limits.",
            "text_length": 95,
            "has_timestamps": True
        },
        "invalid_sample": {
            "source_type": "reflacx",
            "file_path": "missing_file.json",
            "full_text": "",
            "text_length": 0,
            "has_timestamps": False
        },
        "warning_sample": {
            "source_type": "eyegaze",
            "file_path": "artifact_file.json",
            "full_text": "Doctor: 12:34:56 The chest shows... Speaker 2: findings are normal.",
            "text_length": 70,
            "has_timestamps": True
        }
    }
    
    try:
        validator = GroundTruthValidator(logger)
        
        print("🧪 Testing Ground Truth Validator...")
        
        # Test dataset validation
        validation_results = validator.validate_dataset(test_data)
        
        print(f"✅ Dataset validation completed:")
        print(f"   Total files: {validation_results['total_files']}")
        print(f"   Valid files: {validation_results['valid_files']}")
        print(f"   Invalid files: {validation_results['invalid_files']}")
        print(f"   Files with warnings: {validation_results['files_with_warnings']}")
        
        # Test quality distribution
        quality_dist = validation_results['quality_distribution']
        print(f"   Quality distribution: {quality_dist}")
        
        # Test common issues
        common_issues = validation_results['common_issues']
        print(f"   Common issues: {list(common_issues.keys())[:3]}")
        
        print("\n✅ All validation tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    success = test_ground_truth_validator()
    
    if success:
        print("\n🎉 Ground Truth Validator tests passed!")
    else:
        print("\n❌ Some tests failed!") 