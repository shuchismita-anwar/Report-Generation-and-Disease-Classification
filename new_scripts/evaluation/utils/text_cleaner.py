#!/usr/bin/env python3
"""
Report Text Cleaner
===================
Cleans and normalizes radiologist report text for evaluation.

This module handles:
- Removal of timestamps and formatting artifacts
- Whitespace normalization and standardization
- Special character handling
- Medical text preprocessing
- Consistent formatting for evaluation

Author: Medical Report Evaluation System
Version: 1.0
"""

import re
import string
import logging
from typing import Dict, List, Optional, Tuple
import unicodedata


class ReportTextCleaner:
    """
    Cleans and normalizes radiologist report text for evaluation.
    
    Provides comprehensive text cleaning including timestamp removal,
    whitespace normalization, and medical text standardization.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Report Text Cleaner.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Common patterns to remove or normalize
        self.timestamp_patterns = [
            r'\d{1,2}:\d{2}:\d{2}',  # HH:MM:SS
            r'\d{1,2}:\d{2}',        # HH:MM
            r'\[\d{1,2}:\d{2}:\d{2}\]',  # [HH:MM:SS]
            r'\(\d{1,2}:\d{2}:\d{2}\)',  # (HH:MM:SS)
            r'<\d{1,2}:\d{2}:\d{2}>',    # <HH:MM:SS>
        ]
        
        # Patterns for speaker annotations
        self.speaker_patterns = [
            r'\b(doctor|dr|radiologist|physician|md)[\s]*:',
            r'\b(resident|fellow|attending)[\s]*:',
            r'\b(speaker|voice|person)\s*\d*[\s]*:',
            r'\b(male|female)\s*(voice|speaker)[\s]*:',
        ]
        
        # Medical abbreviation normalization
        self.medical_normalizations = {
            # Common variations
            'x-ray': 'radiograph',
            'xray': 'radiograph',
            'cxr': 'chest radiograph',
            'ap': 'anteroposterior',
            'pa': 'posteroanterior',
            'lat': 'lateral',
            'wnl': 'within normal limits',
            'nml': 'normal',
            'nl': 'normal',
            'abn': 'abnormal',
            'neg': 'negative',
            'pos': 'positive',
            'bil': 'bilateral',
            'bilat': 'bilateral',
            'r/o': 'rule out',
            'r o': 'rule out',
            'w/': 'with',
            'w/o': 'without',
            'h/o': 'history of',
            'pt': 'patient',
            'pts': 'patients',
            'yr': 'year',
            'yrs': 'years',
            'mo': 'month',
            'mos': 'months',
            'wk': 'week',
            'wks': 'weeks',
            'yo': 'year old',
            'y/o': 'year old',
            'y.o.': 'year old',
        }
        
        # Punctuation patterns
        self.punctuation_patterns = {
            r'\.{2,}': '.',  # Multiple periods
            r',{2,}': ',',   # Multiple commas
            r'\s+([,.;:!?])': r'\1',  # Space before punctuation
            r'([,.;:!?])\s*([,.;:!?])': r'\1 \2',  # Multiple punctuation
        }
    
    def remove_timestamps(self, text: str) -> str:
        """
        Remove timestamp patterns from text.
        
        Args:
            text (str): Input text with potential timestamps
            
        Returns:
            str: Text with timestamps removed
        """
        cleaned_text = text
        
        for pattern in self.timestamp_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        return cleaned_text
    
    def remove_speaker_annotations(self, text: str) -> str:
        """
        Remove speaker annotations and labels from text.
        
        Args:
            text (str): Input text with potential speaker annotations
            
        Returns:
            str: Text with speaker annotations removed
        """
        cleaned_text = text
        
        for pattern in self.speaker_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        return cleaned_text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text (str): Input text with irregular whitespace
            
        Returns:
            str: Text with normalized whitespace
        """
        # Replace multiple whitespace characters with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        text = re.sub(r'\n+', ' ', text)
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize punctuation in text.
        
        Args:
            text (str): Input text with irregular punctuation
            
        Returns:
            str: Text with normalized punctuation
        """
        cleaned_text = text
        
        for pattern, replacement in self.punctuation_patterns.items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # Ensure sentences end with period
        if cleaned_text and not cleaned_text.rstrip().endswith(('.', '!', '?')):
            cleaned_text = cleaned_text.rstrip() + '.'
        
        return cleaned_text
    
    def normalize_medical_terms(self, text: str) -> str:
        """
        Normalize medical abbreviations and terms.
        
        Args:
            text (str): Input text with medical abbreviations
            
        Returns:
            str: Text with normalized medical terms
        """
        cleaned_text = text.lower()
        
        # Apply medical normalizations
        for abbrev, expansion in self.medical_normalizations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            cleaned_text = re.sub(pattern, expansion, cleaned_text, flags=re.IGNORECASE)
        
        return cleaned_text
    
    def remove_special_characters(self, text: str, keep_medical: bool = True) -> str:
        """
        Remove or normalize special characters.
        
        Args:
            text (str): Input text with special characters
            keep_medical (bool): Whether to keep medical symbols
            
        Returns:
            str: Text with special characters processed
        """
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Remove or replace special characters
        if keep_medical:
            # Keep medical symbols: %, +, -, /, °, μ, etc.
            medical_chars = r'[%+\-/°μ×÷±≤≥<>]'
            # Replace other special characters with space
            text = re.sub(r'[^\w\s.,;:!?()[\]{}"\'' + medical_chars + r']', ' ', text)
        else:
            # Remove all special characters except basic punctuation
            text = re.sub(r'[^\w\s.,;:!?()[\]{}"\']', ' ', text)
        
        return text
    
    def clean_text_comprehensive(self, text: str, 
                               remove_timestamps: bool = True,
                               remove_speakers: bool = True,
                               normalize_medical: bool = True,
                               normalize_punctuation: bool = True,
                               preserve_case: bool = False) -> str:
        """
        Perform comprehensive text cleaning.
        
        Args:
            text (str): Input text to clean
            remove_timestamps (bool): Whether to remove timestamps
            remove_speakers (bool): Whether to remove speaker annotations
            normalize_medical (bool): Whether to normalize medical terms
            normalize_punctuation (bool): Whether to normalize punctuation
            preserve_case (bool): Whether to preserve original case
            
        Returns:
            str: Comprehensively cleaned text
        """
        if not text:
            return ""
        
        cleaned_text = text
        
        # Remove timestamps
        if remove_timestamps:
            cleaned_text = self.remove_timestamps(cleaned_text)
        
        # Remove speaker annotations
        if remove_speakers:
            cleaned_text = self.remove_speaker_annotations(cleaned_text)
        
        # Remove special characters
        cleaned_text = self.remove_special_characters(cleaned_text, keep_medical=True)
        
        # Normalize whitespace
        cleaned_text = self.normalize_whitespace(cleaned_text)
        
        # Normalize medical terms (this also lowercases)
        if normalize_medical:
            cleaned_text = self.normalize_medical_terms(cleaned_text)
        elif not preserve_case:
            cleaned_text = cleaned_text.lower()
        
        # Normalize punctuation
        if normalize_punctuation:
            cleaned_text = self.normalize_punctuation(cleaned_text)
        
        # Final whitespace cleanup
        cleaned_text = self.normalize_whitespace(cleaned_text)
        
        return cleaned_text
    
    def clean_batch(self, text_batch: Dict[str, str], **kwargs) -> Dict[str, Dict]:
        """
        Clean a batch of texts.
        
        Args:
            text_batch (Dict[str, str]): Dictionary of texts to clean
            **kwargs: Arguments passed to clean_text_comprehensive
            
        Returns:
            Dict[str, Dict]: Dictionary with cleaned texts and metadata
        """
        cleaned_batch = {}
        
        self.logger.info(f"Cleaning {len(text_batch)} texts...")
        
        for text_id, original_text in text_batch.items():
            try:
                # Clean the text
                cleaned_text = self.clean_text_comprehensive(original_text, **kwargs)
                
                # Calculate cleaning statistics
                stats = self._calculate_cleaning_stats(original_text, cleaned_text)
                
                cleaned_batch[text_id] = {
                    "original_text": original_text,
                    "cleaned_text": cleaned_text,
                    "original_length": len(original_text),
                    "cleaned_length": len(cleaned_text),
                    "length_reduction": stats["length_reduction"],
                    "word_count_change": stats["word_count_change"],
                    "cleaning_applied": stats["cleaning_applied"]
                }
                
            except Exception as e:
                self.logger.error(f"Error cleaning text {text_id}: {e}")
                cleaned_batch[text_id] = {
                    "original_text": original_text,
                    "cleaned_text": original_text,  # Return original on error
                    "original_length": len(original_text),
                    "cleaned_length": len(original_text),
                    "length_reduction": 0,
                    "word_count_change": 0,
                    "cleaning_applied": [],
                    "error": str(e)
                }
        
        # Log batch statistics
        self._log_batch_cleaning_stats(cleaned_batch)
        
        return cleaned_batch
    
    def _calculate_cleaning_stats(self, original: str, cleaned: str) -> Dict:
        """
        Calculate statistics about the cleaning process.
        
        Args:
            original (str): Original text
            cleaned (str): Cleaned text
            
        Returns:
            Dict: Cleaning statistics
        """
        original_words = len(original.split()) if original else 0
        cleaned_words = len(cleaned.split()) if cleaned else 0
        
        length_reduction = len(original) - len(cleaned)
        word_count_change = original_words - cleaned_words
        
        # Detect what cleaning was applied
        cleaning_applied = []
        
        # Check for timestamp removal
        if any(re.search(pattern, original) for pattern in self.timestamp_patterns):
            cleaning_applied.append("timestamps_removed")
        
        # Check for speaker removal
        if any(re.search(pattern, original, re.IGNORECASE) for pattern in self.speaker_patterns):
            cleaning_applied.append("speakers_removed")
        
        # Check for case changes
        if original != original.lower() and cleaned == cleaned.lower():
            cleaning_applied.append("case_normalized")
        
        # Check for whitespace normalization
        if re.search(r'\s{2,}', original) and not re.search(r'\s{2,}', cleaned):
            cleaning_applied.append("whitespace_normalized")
        
        # Check for punctuation changes
        if len(re.findall(r'[.!?]', original)) != len(re.findall(r'[.!?]', cleaned)):
            cleaning_applied.append("punctuation_normalized")
        
        return {
            "length_reduction": length_reduction,
            "word_count_change": word_count_change,
            "cleaning_applied": cleaning_applied
        }
    
    def _log_batch_cleaning_stats(self, cleaned_batch: Dict[str, Dict]) -> None:
        """
        Log statistics about batch cleaning results.
        
        Args:
            cleaned_batch (Dict[str, Dict]): Batch of cleaned texts
        """
        if not cleaned_batch:
            return
        
        total_texts = len(cleaned_batch)
        successful_cleanings = sum(1 for result in cleaned_batch.values() if "error" not in result)
        
        # Calculate aggregate statistics
        total_length_reduction = sum(result["length_reduction"] for result in cleaned_batch.values())
        total_word_reduction = sum(result["word_count_change"] for result in cleaned_batch.values())
        
        avg_length_reduction = total_length_reduction / total_texts
        avg_word_reduction = total_word_reduction / total_texts
        
        # Count cleaning operations
        all_operations = []
        for result in cleaned_batch.values():
            all_operations.extend(result.get("cleaning_applied", []))
        
        operation_counts = {}
        for op in all_operations:
            operation_counts[op] = operation_counts.get(op, 0) + 1
        
        # Log results
        self.logger.info(f"Text cleaning statistics:")
        self.logger.info(f"  Total texts: {total_texts}, Successful: {successful_cleanings}")
        self.logger.info(f"  Average length reduction: {avg_length_reduction:.1f} characters")
        self.logger.info(f"  Average word reduction: {avg_word_reduction:.1f} words")
        
        if operation_counts:
            self.logger.info(f"  Cleaning operations applied: {operation_counts}")
    
    def validate_cleaned_text(self, original: str, cleaned: str) -> Tuple[bool, List[str]]:
        """
        Validate that text cleaning was successful and appropriate.
        
        Args:
            original (str): Original text
            cleaned (str): Cleaned text
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for excessive reduction
        if len(cleaned) < len(original) * 0.3:  # More than 70% reduction
            issues.append("excessive_text_reduction")
        
        # Check for complete text loss
        if original and not cleaned:
            issues.append("text_completely_removed")
        
        # Check for medical content preservation
        original_medical_words = self._count_medical_words(original)
        cleaned_medical_words = self._count_medical_words(cleaned)
        
        if original_medical_words > 0 and cleaned_medical_words < original_medical_words * 0.5:
            issues.append("medical_content_lost")
        
        # Check for proper sentence structure
        if cleaned and not re.search(r'[.!?]$', cleaned.strip()):
            issues.append("missing_sentence_ending")
        
        # Check for excessive whitespace
        if re.search(r'\s{2,}', cleaned):
            issues.append("excessive_whitespace_remaining")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _count_medical_words(self, text: str) -> int:
        """
        Count medical words in text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            int: Number of medical words found
        """
        medical_keywords = [
            'chest', 'lung', 'heart', 'radiograph', 'x-ray', 'patient',
            'normal', 'abnormal', 'examination', 'findings', 'impression',
            'bilateral', 'cardiopulmonary', 'thoracic', 'pulmonary',
            'cardiac', 'mediastinal', 'pleural', 'pneumonia', 'pneumothorax'
        ]
        
        text_lower = text.lower()
        count = 0
        
        for keyword in medical_keywords:
            count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
        
        return count


def test_text_cleaner():
    """Test function for the Report Text Cleaner."""
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    # Test samples with various cleaning challenges
    test_samples = [
        # Sample 1: Timestamps and speaker annotations
        "Doctor: 09:15:30 The chest X-ray shows normal lung fields. Radiologist: 09:16:45 No acute abnormalities seen.",
        
        # Sample 2: Whitespace and punctuation issues
        "Normal   chest  radiograph  .  .  .  The  lungs  are  clear    bilaterally,,,  heart  size  WNL.",
        
        # Sample 3: Medical abbreviations
        "CXR shows bil clear lungs. Pt is 45 y/o w/ h/o COPD. R/O PNA. Heart size nl.",
        
        # Sample 4: Special characters and formatting
        "Chest X-ray: NORMAL!!! No acute findings... Patient's condition: STABLE @@@ Temperature: 98.6°F",
        
        # Sample 5: Mixed case and formatting
        "CHEST RADIOGRAPH   demonstrates   NORMAL   cardiac   silhouette   AND   clear   LUNG   fields   bilaterally   .",
        
        # Sample 6: Minimal cleaning needed
        "Normal chest radiograph with clear lung fields bilaterally.",
        
        # Sample 7: Complex medical report
        "AP and lateral chest radiographs demonstrate normal cardiac silhouette. The lungs are clear bilaterally. No pleural effusions or pneumothorax. Bony structures are intact.",
        
        # Sample 8: Problematic content
        "Speaker 1: 12:34:56 Um, the, uh, chest... Speaker 2: 12:35:10 shows normal... Speaker 1: 12:35:20 findings.",
        
        # Sample 9: Empty or minimal content
        "",
        
        # Sample 10: Unicode and special characters
        "Radiológica normal. Pulmões limpos. Coração—tamanho normal."
    ]
    
    try:
        cleaner = ReportTextCleaner(logger)
        
        print("🧪 Testing Report Text Cleaner...")
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\n--- Test Sample {i} ---")
            print(f"Original: '{sample[:60]}{'...' if len(sample) > 60 else ''}'")
            
            # Test comprehensive cleaning
            cleaned = cleaner.clean_text_comprehensive(sample)
            print(f"Cleaned:  '{cleaned[:60]}{'...' if len(cleaned) > 60 else ''}'")
            
            # Show length change
            length_change = len(sample) - len(cleaned)
            print(f"Length change: {length_change} characters")
            
            # Test validation
            is_valid, issues = cleaner.validate_cleaned_text(sample, cleaned)
            print(f"Valid: {is_valid}")
            if issues:
                print(f"Issues: {issues}")
        
        # Test batch cleaning
        print(f"\n--- Batch Cleaning Test ---")
        batch_dict = {f"sample_{i}": sample for i, sample in enumerate(test_samples, 1)}
        batch_results = cleaner.clean_batch(batch_dict)
        
        print(f"Batch processed: {len(batch_results)} items")
        
        print("\n✅ All text cleaning tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    success = test_text_cleaner()
    
    if success:
        print("\n🎉 Text Cleaner tests passed!")
    else:
        print("\n❌ Some tests failed!") 