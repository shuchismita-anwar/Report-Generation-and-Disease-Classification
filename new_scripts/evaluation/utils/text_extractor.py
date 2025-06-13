#!/usr/bin/env python3
"""
Report Text Extractor
=====================
Extracts and processes text from various transcript formats.

This module handles:
- Different JSON structures from EyeGaze and REFLACX
- Text extraction from full_text and time_stamped_text fields
- Handling of missing or malformed data
- Text preprocessing and normalization

Author: Medical Report Evaluation System
Version: 1.0
"""

import re
import logging
from typing import Dict, List, Optional, Union, Tuple
import json


class ReportTextExtractor:
    """
    Extracts and processes text from radiologist transcript data.
    
    Handles multiple transcript formats and provides robust text extraction
    with error handling for missing or malformed data.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Report Text Extractor.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Common medical abbreviations and their expansions
        self.medical_abbreviations = {
            'AP': 'anteroposterior',
            'PA': 'posteroanterior',
            'LAT': 'lateral',
            'CXR': 'chest X-ray',
            'NPO': 'nothing by mouth',
            'SOB': 'shortness of breath',
            'DOE': 'dyspnea on exertion',
            'CHF': 'congestive heart failure',
            'COPD': 'chronic obstructive pulmonary disease',
            'PE': 'pulmonary embolism',
            'PNA': 'pneumonia',
            'PTX': 'pneumothorax',
            'R/O': 'rule out',
            'WNL': 'within normal limits'
        }
    
    def extract_full_text(self, transcript_data: Dict) -> str:
        """
        Extract the full report text from transcript data.
        
        Args:
            transcript_data (Dict): Raw transcript data from JSON
            
        Returns:
            str: Extracted full text or empty string if not found
        """
        # Primary field: full_text
        if "full_text" in transcript_data and transcript_data["full_text"]:
            text = transcript_data["full_text"].strip()
            if text:
                self.logger.debug(f"Extracted full_text: {len(text)} characters")
                return text
        
        # Fallback: Try to reconstruct from time_stamped_text
        if "time_stamped_text" in transcript_data:
            reconstructed = self.reconstruct_from_timestamps(transcript_data["time_stamped_text"])
            if reconstructed:
                self.logger.debug(f"Reconstructed from timestamps: {len(reconstructed)} characters")
                return reconstructed
        
        # Fallback: Check for other text fields
        text_fields = ["text", "report", "content", "transcript", "description"]
        for field in text_fields:
            if field in transcript_data and transcript_data[field]:
                text = str(transcript_data[field]).strip()
                if text:
                    self.logger.debug(f"Extracted from {field}: {len(text)} characters")
                    return text
        
        self.logger.warning("No text found in transcript data")
        return ""
    
    def reconstruct_from_timestamps(self, time_stamped_data: List[Dict]) -> str:
        """
        Reconstruct full text from time-stamped segments.
        
        Args:
            time_stamped_data (List[Dict]): List of time-stamped text segments
            
        Returns:
            str: Reconstructed text
        """
        if not time_stamped_data:
            return ""
        
        text_segments = []
        
        for segment in time_stamped_data:
            if isinstance(segment, dict):
                # Try different possible text field names
                text_fields = ["text", "content", "transcript", "words", "phrase"]
                for field in text_fields:
                    if field in segment and segment[field]:
                        text_segments.append(str(segment[field]).strip())
                        break
            elif isinstance(segment, str):
                text_segments.append(segment.strip())
        
        # Combine segments with appropriate spacing
        reconstructed = " ".join(text_segments)
        return reconstructed.strip()
    
    def extract_text_with_metadata(self, transcript_data: Dict) -> Dict[str, Union[str, int, bool]]:
        """
        Extract text along with metadata about the extraction.
        
        Args:
            transcript_data (Dict): Raw transcript data
            
        Returns:
            Dict: Text and extraction metadata
        """
        # Extract the main text
        full_text = self.extract_full_text(transcript_data)
        
        # Gather metadata
        metadata = {
            "text": full_text,
            "text_length": len(full_text),
            "is_empty": len(full_text) == 0,
            "has_full_text_field": "full_text" in transcript_data,
            "has_timestamps": "time_stamped_text" in transcript_data,
            "timestamp_count": len(transcript_data.get("time_stamped_text", [])),
            "extraction_method": self._determine_extraction_method(transcript_data, full_text),
            "word_count": len(full_text.split()) if full_text else 0,
            "sentence_count": len([s for s in full_text.split('.') if s.strip()]) if full_text else 0
        }
        
        # Check for potential issues
        metadata["potential_issues"] = self._detect_text_issues(full_text, transcript_data)
        
        return metadata
    
    def _determine_extraction_method(self, transcript_data: Dict, extracted_text: str) -> str:
        """
        Determine which method was used to extract the text.
        
        Args:
            transcript_data (Dict): Raw transcript data
            extracted_text (str): The extracted text
            
        Returns:
            str: Method used for extraction
        """
        if not extracted_text:
            return "none"
        
        if "full_text" in transcript_data and transcript_data.get("full_text", "").strip():
            return "full_text"
        
        if "time_stamped_text" in transcript_data:
            return "timestamp_reconstruction"
        
        # Check other fields
        text_fields = ["text", "report", "content", "transcript", "description"]
        for field in text_fields:
            if field in transcript_data and transcript_data.get(field, ""):
                return f"fallback_{field}"
        
        return "unknown"
    
    def _detect_text_issues(self, text: str, transcript_data: Dict) -> List[str]:
        """
        Detect potential issues with extracted text.
        
        Args:
            text (str): Extracted text
            transcript_data (Dict): Raw transcript data
            
        Returns:
            List[str]: List of potential issues found
        """
        issues = []
        
        if not text:
            issues.append("empty_text")
            return issues
        
        # Check for very short text (likely incomplete)
        if len(text) < 10:
            issues.append("very_short_text")
        
        # Check for very long text (might be malformed)
        if len(text) > 5000:
            issues.append("very_long_text")
        
        # Check for non-English characters (excluding medical symbols)
        if re.search(r'[^\x00-\x7F]+', text):
            issues.append("non_ascii_characters")
        
        # Check for excessive repetition
        words = text.lower().split()
        if len(words) > 10:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.3:  # More than 30% repetition
                issues.append("excessive_repetition")
        
        # Check for incomplete sentences
        if text and not text.rstrip().endswith(('.', '!', '?')):
            issues.append("incomplete_sentence")
        
        # Check for timestamp artifacts
        if re.search(r'\d{1,2}:\d{2}', text):
            issues.append("timestamp_artifacts")
        
        # Check for JSON artifacts
        if any(char in text for char in ['{', '}', '[', ']', '":']):
            issues.append("json_artifacts")
        
        return issues
    
    def extract_batch(self, transcript_batch: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Extract text from a batch of transcripts.
        
        Args:
            transcript_batch (Dict[str, Dict]): Dictionary of transcript data
            
        Returns:
            Dict[str, Dict]: Dictionary of extracted texts with metadata
        """
        extracted_batch = {}
        
        self.logger.info(f"Extracting text from {len(transcript_batch)} transcripts...")
        
        for image_id, transcript_data in transcript_batch.items():
            try:
                # Extract raw data if it's nested
                raw_data = transcript_data.get("raw_data", transcript_data)
                
                # Extract text with metadata
                extraction_result = self.extract_text_with_metadata(raw_data)
                
                # Add original metadata if available
                extraction_result.update({
                    "image_id": image_id,
                    "source_type": transcript_data.get("source_type", "unknown"),
                    "file_path": transcript_data.get("file_path", ""),
                    "original_text_length": transcript_data.get("text_length", 0)
                })
                
                extracted_batch[image_id] = extraction_result
                
            except Exception as e:
                self.logger.error(f"Error extracting text for {image_id}: {e}")
                # Create error entry
                extracted_batch[image_id] = {
                    "text": "",
                    "text_length": 0,
                    "is_empty": True,
                    "extraction_method": "error",
                    "potential_issues": ["extraction_error"],
                    "error": str(e)
                }
        
        # Generate batch statistics
        self._log_extraction_statistics(extracted_batch)
        
        return extracted_batch
    
    def _log_extraction_statistics(self, extracted_batch: Dict[str, Dict]) -> None:
        """
        Log statistics about the text extraction results.
        
        Args:
            extracted_batch (Dict[str, Dict]): Batch of extraction results
        """
        if not extracted_batch:
            return
        
        total = len(extracted_batch)
        successful = sum(1 for result in extracted_batch.values() if not result["is_empty"])
        empty = total - successful
        
        # Calculate text length statistics
        text_lengths = [result["text_length"] for result in extracted_batch.values() 
                       if not result["is_empty"]]
        
        if text_lengths:
            avg_length = sum(text_lengths) / len(text_lengths)
            min_length = min(text_lengths)
            max_length = max(text_lengths)
        else:
            avg_length = min_length = max_length = 0
        
        # Count extraction methods
        methods = {}
        for result in extracted_batch.values():
            method = result.get("extraction_method", "unknown")
            methods[method] = methods.get(method, 0) + 1
        
        # Count issues
        all_issues = []
        for result in extracted_batch.values():
            all_issues.extend(result.get("potential_issues", []))
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Log statistics
        self.logger.info(f"Text extraction statistics:")
        self.logger.info(f"  Total: {total}, Successful: {successful}, Empty: {empty}")
        self.logger.info(f"  Text lengths - Avg: {avg_length:.1f}, Min: {min_length}, Max: {max_length}")
        self.logger.info(f"  Extraction methods: {methods}")
        
        if issue_counts:
            self.logger.info(f"  Issues found: {issue_counts}")
    
    def validate_extracted_text(self, text: str, min_length: int = 5, 
                               max_length: int = 10000) -> Tuple[bool, List[str]]:
        """
        Validate extracted text for quality and completeness.
        
        Args:
            text (str): Text to validate
            min_length (int): Minimum acceptable text length
            max_length (int): Maximum acceptable text length
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check length
        if len(text) < min_length:
            issues.append(f"text_too_short (< {min_length} chars)")
        
        if len(text) > max_length:
            issues.append(f"text_too_long (> {max_length} chars)")
        
        # Check for medical content (basic validation)
        medical_terms = [
            "chest", "lung", "heart", "normal", "abnormal", "radiograph",
            "x-ray", "impression", "findings", "patient", "examination"
        ]
        
        text_lower = text.lower()
        has_medical_content = any(term in text_lower for term in medical_terms)
        
        if not has_medical_content and len(text) > 20:
            issues.append("no_medical_content_detected")
        
        # Check text structure
        if text and not re.search(r'[.!?]', text):
            issues.append("no_sentence_endings")
        
        is_valid = len(issues) == 0
        return is_valid, issues


def test_text_extractor():
    """Test function for the Report Text Extractor."""
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    # Test data samples
    test_samples = [
        # Sample 1: Normal structure with full_text
        {
            "full_text": "Normal chest radiograph. The lungs are clear bilaterally. The heart size is normal. No acute abnormalities identified.",
            "time_stamped_text": []
        },
        
        # Sample 2: Empty full_text, with time_stamped_text
        {
            "full_text": "",
            "time_stamped_text": [
                {"text": "Normal chest"},
                {"text": "radiograph showing"},
                {"text": "clear lungs"}
            ]
        },
        
        # Sample 3: Missing full_text, alternative field
        {
            "report": "Chest X-ray demonstrates normal cardiac silhouette and clear lung fields.",
            "time_stamped_text": []
        },
        
        # Sample 4: Malformed data
        {
            "full_text": None,
            "time_stamped_text": "not_a_list"
        }
    ]
    
    try:
        extractor = ReportTextExtractor(logger)
        
        print("🧪 Testing Report Text Extractor...")
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\n--- Test Sample {i} ---")
            
            # Test basic extraction
            text = extractor.extract_full_text(sample)
            print(f"Extracted text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Test extraction with metadata
            metadata = extractor.extract_text_with_metadata(sample)
            print(f"Text length: {metadata['text_length']}")
            print(f"Extraction method: {metadata['extraction_method']}")
            print(f"Issues: {metadata['potential_issues']}")
            
            # Test validation
            is_valid, issues = extractor.validate_extracted_text(text)
            print(f"Valid: {is_valid}, Validation issues: {issues}")
        
        print("\n✅ All text extraction tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    success = test_text_extractor()
    
    if success:
        print("\n🎉 Text Extractor tests passed!")
    else:
        print("\n❌ Some tests failed!") 