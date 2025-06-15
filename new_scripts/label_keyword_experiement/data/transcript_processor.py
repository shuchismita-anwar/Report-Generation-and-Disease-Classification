"""
Transcript Processor for Hybrid RadBERT + MedCLIP Label & Keyword Prediction
============================================================================

This module handles loading and processing of transcript JSON files from the MIMIC dataset.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptProcessor:
    """
    Handles loading and processing of transcript files from the MIMIC dataset.
    
    This class provides functionality to:
    - Find all transcript files in the dataset
    - Load and parse individual transcript JSON files
    - Extract text content from transcripts
    - Handle malformed or missing files gracefully
    """
    
    def __init__(self, base_data_dir: str = "../../"):
        """
        Initialize the TranscriptProcessor.
        
        Args:
            base_data_dir: Base directory containing the data files
        """
        self.base_data_dir = Path(base_data_dir)
        self.transcript_cache = {}  # Cache for loaded transcripts
        self.error_log = []  # Track files that failed to load
        
    def find_transcript_files(self, csv_path: str) -> List[str]:
        """
        Find all transcript files referenced in the dataset CSV.
        
        Args:
            csv_path: Path to the dataset CSV file
            
        Returns:
            List of transcript file paths
        """
        try:
            df = pd.read_csv(csv_path)
            
            if 'transcript_path' not in df.columns:
                logger.error(f"No 'transcript_path' column found in {csv_path}")
                return []
            
            # Get unique transcript paths and filter out NaN values
            transcript_paths = df['transcript_path'].dropna().unique().tolist()
            
            logger.info(f"Found {len(transcript_paths)} unique transcript files in {csv_path}")
            return transcript_paths
            
        except Exception as e:
            logger.error(f"Error reading CSV file {csv_path}: {e}")
            return []
    
    def load_transcript(self, transcript_path: str) -> Optional[str]:
        """
        Load and parse a single transcript file.
        
        Args:
            transcript_path: Path to the transcript JSON file (relative to base_data_dir)
            
        Returns:
            Transcript text content as string, or None if loading failed
        """
        # Check cache first
        if transcript_path in self.transcript_cache:
            return self.transcript_cache[transcript_path]
        
        try:
            # Construct full path
            full_path = self.base_data_dir / transcript_path
            
            if not full_path.exists():
                logger.warning(f"Transcript file not found: {full_path}")
                self.error_log.append(f"File not found: {transcript_path}")
                return None
            
            # Load JSON file
            with open(full_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Extract text content
            text_content = self._extract_text_from_transcript(transcript_data)
            
            # Cache the result
            self.transcript_cache[transcript_path] = text_content
            
            return text_content
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {transcript_path}: {e}")
            self.error_log.append(f"JSON decode error: {transcript_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading transcript {transcript_path}: {e}")
            self.error_log.append(f"Load error: {transcript_path}")
            return None
    
    def _extract_text_from_transcript(self, transcript_data: Dict) -> str:
        """
        Extract text content from transcript JSON data.
        
        Args:
            transcript_data: Parsed JSON data from transcript file
            
        Returns:
            Extracted text content as string
        """
        text_content = ""
        
        try:
            # Handle different possible transcript formats
            if isinstance(transcript_data, dict):
                # Check for MIMIC transcript format first
                if 'full_text' in transcript_data:
                    text_content = str(transcript_data['full_text'])
                # Check for other common text fields
                elif 'text' in transcript_data:
                    text_content = str(transcript_data['text'])
                elif 'content' in transcript_data:
                    text_content = str(transcript_data['content'])
                elif 'report' in transcript_data:
                    text_content = str(transcript_data['report'])
                elif 'findings' in transcript_data and 'impression' in transcript_data:
                    # Combine findings and impression
                    findings = str(transcript_data.get('findings', ''))
                    impression = str(transcript_data.get('impression', ''))
                    text_content = f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"
                else:
                    # Try to extract any string values
                    text_parts = []
                    for key, value in transcript_data.items():
                        if isinstance(value, str) and len(value.strip()) > 10:
                            text_parts.append(f"{key.upper()}: {value}")
                    text_content = "\n\n".join(text_parts)
                    
            elif isinstance(transcript_data, str):
                text_content = transcript_data
            elif isinstance(transcript_data, list):
                # Handle list of text segments
                text_parts = [str(item) for item in transcript_data if isinstance(item, str)]
                text_content = " ".join(text_parts)
            else:
                text_content = str(transcript_data)
            
            # Clean up the text
            text_content = text_content.strip()
            
            # Remove excessive whitespace
            import re
            text_content = re.sub(r'\s+', ' ', text_content)
            text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
            
            return text_content
            
        except Exception as e:
            logger.error(f"Error extracting text from transcript: {e}")
            return str(transcript_data)  # Fallback to string conversion
    
    def load_multiple_transcripts(self, transcript_paths: List[str]) -> Dict[str, Optional[str]]:
        """
        Load multiple transcript files.
        
        Args:
            transcript_paths: List of transcript file paths
            
        Returns:
            Dictionary mapping transcript paths to their text content
        """
        results = {}
        
        for transcript_path in transcript_paths:
            results[transcript_path] = self.load_transcript(transcript_path)
        
        # Log statistics
        successful_loads = sum(1 for content in results.values() if content is not None)
        logger.info(f"Successfully loaded {successful_loads}/{len(transcript_paths)} transcripts")
        
        return results
    
    def get_transcript_statistics(self, transcript_paths: List[str]) -> Dict[str, Union[int, float]]:
        """
        Get statistics about transcript files.
        
        Args:
            transcript_paths: List of transcript file paths
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_files': len(transcript_paths),
            'successful_loads': 0,
            'failed_loads': 0,
            'total_characters': 0,
            'total_words': 0,
            'avg_characters': 0.0,
            'avg_words': 0.0,
            'min_length': float('inf'),
            'max_length': 0
        }
        
        for transcript_path in transcript_paths:
            content = self.load_transcript(transcript_path)
            
            if content is not None:
                stats['successful_loads'] += 1
                char_count = len(content)
                word_count = len(content.split())
                
                stats['total_characters'] += char_count
                stats['total_words'] += word_count
                stats['min_length'] = min(stats['min_length'], char_count)
                stats['max_length'] = max(stats['max_length'], char_count)
            else:
                stats['failed_loads'] += 1
        
        # Calculate averages
        if stats['successful_loads'] > 0:
            stats['avg_characters'] = stats['total_characters'] / stats['successful_loads']
            stats['avg_words'] = stats['total_words'] / stats['successful_loads']
        
        if stats['min_length'] == float('inf'):
            stats['min_length'] = 0
        
        return stats
    
    def validate_transcript_files(self, csv_path: str) -> Tuple[List[str], List[str]]:
        """
        Validate all transcript files referenced in a CSV.
        
        Args:
            csv_path: Path to the dataset CSV file
            
        Returns:
            Tuple of (valid_files, invalid_files)
        """
        transcript_paths = self.find_transcript_files(csv_path)
        valid_files = []
        invalid_files = []
        
        for transcript_path in transcript_paths:
            content = self.load_transcript(transcript_path)
            if content is not None and len(content.strip()) > 0:
                valid_files.append(transcript_path)
            else:
                invalid_files.append(transcript_path)
        
        logger.info(f"Validation complete: {len(valid_files)} valid, {len(invalid_files)} invalid files")
        return valid_files, invalid_files
    
    def clear_cache(self):
        """Clear the transcript cache to free memory."""
        self.transcript_cache.clear()
        logger.info("Transcript cache cleared")
    
    def get_error_log(self) -> List[str]:
        """Get list of files that failed to load."""
        return self.error_log.copy()


# Convenience functions for easy usage
def load_transcript_from_path(transcript_path: str, base_data_dir: str = "../../") -> Optional[str]:
    """
    Convenience function to load a single transcript.
    
    Args:
        transcript_path: Path to transcript file
        base_data_dir: Base data directory
        
    Returns:
        Transcript text content or None
    """
    processor = TranscriptProcessor(base_data_dir)
    return processor.load_transcript(transcript_path)


def get_all_transcripts_from_csv(csv_path: str, base_data_dir: str = "../../") -> Dict[str, Optional[str]]:
    """
    Convenience function to load all transcripts from a CSV file.
    
    Args:
        csv_path: Path to dataset CSV
        base_data_dir: Base data directory
        
    Returns:
        Dictionary mapping transcript paths to content
    """
    processor = TranscriptProcessor(base_data_dir)
    transcript_paths = processor.find_transcript_files(csv_path)
    return processor.load_multiple_transcripts(transcript_paths)


if __name__ == "__main__":
    # Example usage and testing
    processor = TranscriptProcessor()
    
    # Test with train CSV
    train_csv = "../../dataset_splits/train.csv"
    if os.path.exists(train_csv):
        print("Testing transcript loading...")
        transcript_paths = processor.find_transcript_files(train_csv)[:5]  # Test first 5
        results = processor.load_multiple_transcripts(transcript_paths)
        
        for path, content in results.items():
            if content:
                print(f"\n{path}: {len(content)} characters")
                print(f"Preview: {content[:200]}...")
            else:
                print(f"\n{path}: Failed to load")
    else:
        print(f"Train CSV not found at {train_csv}") 