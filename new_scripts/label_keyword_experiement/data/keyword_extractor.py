"""
Keyword Extractor for Hybrid RadBERT + MedCLIP Label & Keyword Prediction
=========================================================================

This module handles extraction of medical keywords from transcript text using RadBERT
for sophisticated medical NLP and entity recognition.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union
from collections import Counter, defaultdict

# Import RadBERT module
try:
    from models.radbert_module import RadBERTModule, create_radbert_processor
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models.radbert_module import RadBERTModule, create_radbert_processor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordExtractor:
    """
    Extracts medical keywords from transcript text using RadBERT for sophisticated medical NLP.
    
    This class provides functionality to:
    - Extract medical entities using RadBERT semantic understanding
    - Filter and rank keywords by clinical relevance and confidence scores
    - Create fallback keyword lists when RadBERT is unavailable
    - Save extracted keywords to JSON format
    """
    
    def __init__(self, use_radbert: bool = True, radbert_model: str = "UCSD-VA-health/RadBERT-RoBERTa-4m"):
        """
        Initialize the KeywordExtractor with RadBERT or fallback to basic patterns.
        
        Args:
            use_radbert: Whether to use RadBERT for extraction
            radbert_model: RadBERT model name for HuggingFace
        """
        self.use_radbert = use_radbert
        
        # Initialize RadBERT processor
        if use_radbert:
            try:
                logger.info("Initializing RadBERT processor for keyword extraction")
                self.radbert_processor = create_radbert_processor(model_name=radbert_model)
                logger.info("RadBERT processor initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize RadBERT: {e}")
                logger.warning("Falling back to basic pattern extraction")
                self.use_radbert = False
                self.radbert_processor = None
        else:
            self.radbert_processor = None
            logger.info("Using basic pattern extraction (RadBERT disabled)")
        
        # Fallback basic patterns (kept for compatibility and fallback)
        self.medical_patterns = self._create_medical_patterns()
        self.anatomical_terms = self._create_anatomical_vocabulary()
        self.pathology_terms = self._create_pathology_vocabulary()
        self.device_terms = self._create_device_vocabulary()
        self.descriptor_terms = self._create_descriptor_vocabulary()
        self.stopwords = self._create_medical_stopwords()
        
        # Statistics tracking
        self.extraction_stats = {
            'total_transcripts_processed': 0,
            'total_keywords_extracted': 0,
            'unique_keywords': 0,
            'avg_keywords_per_transcript': 0.0,
            'radbert_used': self.use_radbert
        }
    
    def _create_medical_patterns(self) -> List[Dict[str, Union[str, re.Pattern]]]:
        """Create regex patterns for medical term extraction."""
        patterns = [
            # Anatomical structures
            {
                'category': 'anatomy',
                'pattern': re.compile(r'\b(?:heart|cardiac|lung|pulmonary|chest|thorax|mediastin|hilar|pleural|diaphragm|rib|clavicle|spine|vertebra|sternum|aorta|vena cava)\w*\b', re.IGNORECASE),
                'description': 'Anatomical structures'
            },
            
            # Pathological conditions
            {
                'category': 'pathology',
                'pattern': re.compile(r'\b(?:atelectasis|consolidation|effusion|pneumothorax|pneumonia|edema|opacity|infiltrate|nodule|mass|lesion|cardiomegaly|enlargement|abnormality)\w*\b', re.IGNORECASE),
                'description': 'Pathological conditions'
            },
            
            # Medical devices
            {
                'category': 'devices',
                'pattern': re.compile(r'\b(?:tube|catheter|line|device|pacemaker|defibrillator|stent|prosthesis|implant|hardware|wire|lead|port|picc|ng tube|endotracheal|tracheostomy)\w*\b', re.IGNORECASE),
                'description': 'Medical devices and equipment'
            },
            
            # Descriptive terms
            {
                'category': 'descriptors',
                'pattern': re.compile(r'\b(?:normal|abnormal|enlarged|small|large|bilateral|unilateral|left|right|upper|lower|anterior|posterior|lateral|medial|central|peripheral|acute|chronic|mild|moderate|severe|stable|improved|worsened)\w*\b', re.IGNORECASE),
                'description': 'Descriptive medical terms'
            },
            
            # Negation indicators (important for medical context)
            {
                'category': 'negation',
                'pattern': re.compile(r'\b(?:no|not|without|absent|clear|negative|unremarkable|within normal limits|normal)\b', re.IGNORECASE),
                'description': 'Negation and normal findings'
            }
        ]
        return patterns
    
    def _create_anatomical_vocabulary(self) -> Set[str]:
        """Create vocabulary of anatomical terms."""
        return {
            # Cardiac
            'heart', 'cardiac', 'cardiomediastinal', 'pericardium', 'aorta', 'pulmonary artery',
            'left ventricle', 'right ventricle', 'atrium', 'valve',
            
            # Pulmonary
            'lung', 'lungs', 'pulmonary', 'bronchi', 'alveolar', 'pleura', 'pleural',
            'left lung', 'right lung', 'upper lobe', 'lower lobe', 'middle lobe',
            'lingula', 'apex', 'base', 'hilum', 'hilar',
            
            # Thoracic structures
            'chest', 'thorax', 'mediastinum', 'diaphragm', 'costophrenic angle',
            'rib', 'ribs', 'clavicle', 'sternum', 'spine', 'vertebra',
            
            # Regions
            'bilateral', 'unilateral', 'left', 'right', 'upper', 'lower',
            'anterior', 'posterior', 'lateral', 'medial', 'central', 'peripheral'
        }
    
    def _create_pathology_vocabulary(self) -> Set[str]:
        """Create vocabulary of pathological terms."""
        return {
            # Common findings
            'atelectasis', 'consolidation', 'opacity', 'infiltrate', 'effusion',
            'pneumothorax', 'pneumonia', 'edema', 'congestion',
            
            # Masses and nodules
            'nodule', 'mass', 'lesion', 'density', 'shadow',
            
            # Cardiac pathology
            'cardiomegaly', 'enlargement', 'dilation', 'hypertrophy',
            
            # General descriptors
            'abnormality', 'abnormal', 'pathology', 'disease',
            'inflammation', 'infection', 'fluid', 'air'
        }
    
    def _create_device_vocabulary(self) -> Set[str]:
        """Create vocabulary of medical devices."""
        return {
            # Tubes and lines
            'tube', 'catheter', 'line', 'ng tube', 'endotracheal tube',
            'tracheostomy', 'chest tube', 'picc line', 'central line',
            
            # Cardiac devices
            'pacemaker', 'defibrillator', 'icd', 'stent', 'valve',
            
            # Support devices
            'ventilator', 'cpap', 'bipap', 'oxygen', 'cannula',
            
            # Surgical hardware
            'hardware', 'wire', 'screw', 'plate', 'rod', 'prosthesis',
            'implant', 'graft', 'clip', 'staple'
        }
    
    def _create_descriptor_vocabulary(self) -> Set[str]:
        """Create vocabulary of descriptive terms."""
        return {
            # Size descriptors
            'small', 'large', 'enlarged', 'dilated', 'expanded',
            'compressed', 'collapsed', 'distended',
            
            # Severity descriptors
            'mild', 'moderate', 'severe', 'marked', 'significant',
            'minimal', 'trace', 'extensive',
            
            # Temporal descriptors
            'acute', 'chronic', 'subacute', 'recent', 'old', 'new',
            'stable', 'improved', 'worsened', 'progressive',
            
            # Quality descriptors
            'clear', 'opaque', 'dense', 'lucent', 'heterogeneous',
            'homogeneous', 'irregular', 'smooth', 'rough'
        }
    
    def _create_medical_stopwords(self) -> Set[str]:
        """Create set of medical stopwords to filter out."""
        return {
            # Common non-informative words
            'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for',
            'with', 'without', 'is', 'are', 'was', 'were', 'been',
            'have', 'has', 'had', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'cannot', 'there', 'here', 'this',
            'that', 'these', 'those', 'a', 'an', 'as', 'by', 'from',
            
            # Medical report boilerplate
            'examination', 'study', 'image', 'images', 'view', 'views',
            'comparison', 'compared', 'prior', 'previous', 'technique',
            'quality', 'adequate', 'limited', 'suboptimal'
        }
    
    def extract_keywords_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical keywords from transcript text using RadBERT or fallback patterns.
        
        Args:
            text: Transcript text to process
            
        Returns:
            Dictionary with categorized keywords
        """
        if not text or not text.strip():
            return {'anatomy': [], 'pathology': [], 'devices': [], 'descriptors': [], 'location': []}
        
        # Use RadBERT if available
        if self.use_radbert and self.radbert_processor is not None:
            return self._extract_with_radbert(text)
        else:
            return self._extract_with_patterns(text)
    
    def _extract_with_radbert(self, text: str) -> Dict[str, List[str]]:
        """
        Extract keywords using RadBERT semantic understanding.
        
        Args:
            text: Transcript text to process
            
        Returns:
            Dictionary with categorized keywords
        """
        try:
            # Use RadBERT for medical keyword extraction
            radbert_keywords = self.radbert_processor.extract_medical_keywords(text, top_k=10)
            
            # Convert RadBERT format to expected format
            categorized_keywords = {}
            for category, keyword_list in radbert_keywords.items():
                # Extract just the keyword strings (remove confidence scores for compatibility)
                keywords = [item['keyword'] for item in keyword_list if item['confidence'] > 0.3]
                categorized_keywords[category] = keywords
            
            # Ensure all expected categories are present
            expected_categories = ['anatomy', 'pathology', 'devices', 'descriptors', 'location']
            for category in expected_categories:
                if category not in categorized_keywords:
                    categorized_keywords[category] = []
            
            logger.debug(f"RadBERT extracted {sum(len(v) for v in categorized_keywords.values())} keywords")
            return categorized_keywords
            
        except Exception as e:
            logger.warning(f"RadBERT extraction failed: {e}")
            logger.warning("Falling back to pattern-based extraction")
            return self._extract_with_patterns(text)
    
    def _extract_with_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        Extract keywords using basic pattern matching (fallback method).
        
        Args:
            text: Transcript text to process
            
        Returns:
            Dictionary with categorized keywords
        """
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Extract keywords by category
        categorized_keywords = {}
        
        for pattern_info in self.medical_patterns:
            category = pattern_info['category']
            pattern = pattern_info['pattern']
            
            # Find all matches
            matches = pattern.findall(cleaned_text)
            
            # Clean and filter matches
            filtered_matches = []
            for match in matches:
                cleaned_match = match.lower().strip()
                if (len(cleaned_match) >= 3 and 
                    cleaned_match not in self.stopwords and
                    not cleaned_match.isdigit()):
                    filtered_matches.append(cleaned_match)
            
            categorized_keywords[category] = list(set(filtered_matches))  # Remove duplicates
        
        # Additional vocabulary-based extraction
        vocab_keywords = self._extract_vocabulary_keywords(cleaned_text)
        
        # Merge vocabulary keywords with pattern-based keywords
        for category, keywords in vocab_keywords.items():
            if category in categorized_keywords:
                categorized_keywords[category].extend(keywords)
                categorized_keywords[category] = list(set(categorized_keywords[category]))
            else:
                categorized_keywords[category] = keywords
        
        return categorized_keywords
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for keyword extraction."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation but keep medical abbreviations
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Handle common medical abbreviations
        text = re.sub(r'\bng\b', 'ng tube', text)
        text = re.sub(r'\bet\b', 'endotracheal', text)
        text = re.sub(r'\bicd\b', 'defibrillator', text)
        
        return text.strip()
    
    def _extract_vocabulary_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract keywords using predefined vocabularies."""
        vocab_keywords = {
            'anatomy': [],
            'pathology': [],
            'devices': [],
            'descriptors': []
        }
        
        # Check each vocabulary
        vocabularies = {
            'anatomy': self.anatomical_terms,
            'pathology': self.pathology_terms,
            'devices': self.device_terms,
            'descriptors': self.descriptor_terms
        }
        
        for category, vocab in vocabularies.items():
            for term in vocab:
                if term.lower() in text:
                    vocab_keywords[category].append(term.lower())
        
        return vocab_keywords
    
    def create_dummy_keywords(self, num_keywords: int = 20) -> Dict[str, List[str]]:
        """
        Create dummy keyword lists for testing purposes.
        
        Args:
            num_keywords: Number of keywords to generate per category
            
        Returns:
            Dictionary with dummy keywords by category
        """
        dummy_keywords = {
            'anatomy': [
                'heart', 'lung', 'chest', 'mediastinum', 'pleura',
                'diaphragm', 'rib', 'clavicle', 'hilum', 'aorta',
                'pulmonary', 'cardiac', 'thorax', 'sternum', 'spine'
            ][:num_keywords//4],
            
            'pathology': [
                'atelectasis', 'consolidation', 'effusion', 'pneumothorax',
                'cardiomegaly', 'opacity', 'infiltrate', 'edema', 'pneumonia',
                'abnormality', 'enlargement', 'nodule', 'mass', 'lesion'
            ][:num_keywords//4],
            
            'devices': [
                'tube', 'catheter', 'pacemaker', 'line', 'device',
                'ng tube', 'endotracheal', 'picc', 'stent', 'hardware',
                'wire', 'implant', 'prosthesis', 'defibrillator'
            ][:num_keywords//4],
            
            'descriptors': [
                'normal', 'abnormal', 'enlarged', 'small', 'bilateral',
                'left', 'right', 'upper', 'lower', 'mild', 'moderate',
                'severe', 'acute', 'chronic', 'stable', 'clear'
            ][:num_keywords//4]
        }
        
        return dummy_keywords
    
    def extract_keywords_from_transcripts(self, transcripts: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract keywords from multiple transcripts using RadBERT or fallback pattern matching.
        
        Args:
            transcripts: Dictionary mapping transcript paths to text content
            
        Returns:
            Dictionary mapping transcript paths to extracted keywords
        """
        # Use RadBERT batch processing if available
        if self.use_radbert and self.radbert_processor is not None:
            return self._extract_from_transcripts_radbert(transcripts)
        else:
            return self._extract_from_transcripts_patterns(transcripts)
    
    def _extract_from_transcripts_radbert(self, transcripts: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract keywords from multiple transcripts using RadBERT.
        
        Args:
            transcripts: Dictionary mapping transcript IDs to text content
            
        Returns:
            Dictionary mapping transcript IDs to categorized keywords
        """
        logger.info(f"Processing {len(transcripts)} transcripts with RadBERT")
        
        try:
            # Use RadBERT's batch processing method
            radbert_results = self.radbert_processor.extract_keywords_from_transcripts(transcripts)
            
            # Convert RadBERT format to expected format
            all_keywords = {}
            processed_count = 0
            total_keywords = 0
            
            for transcript_id, radbert_keywords in radbert_results.items():
                categorized_keywords = {}
                
                for category, keyword_list in radbert_keywords.items():
                    # Extract just the keyword strings (remove confidence scores)
                    keywords = [item['keyword'] for item in keyword_list if item['confidence'] > 0.3]
                    categorized_keywords[category] = keywords
                    total_keywords += len(keywords)
                
                # Ensure all expected categories are present
                expected_categories = ['anatomy', 'pathology', 'devices', 'descriptors', 'location']
                for category in expected_categories:
                    if category not in categorized_keywords:
                        categorized_keywords[category] = []
                
                all_keywords[transcript_id] = categorized_keywords
                processed_count += 1
            
            # Update statistics
            self.extraction_stats['total_transcripts_processed'] = processed_count
            self.extraction_stats['total_keywords_extracted'] = total_keywords
            
            unique_keywords = set()
            for keywords in all_keywords.values():
                for keyword_list in keywords.values():
                    unique_keywords.update(keyword_list)
            
            self.extraction_stats['unique_keywords'] = len(unique_keywords)
            self.extraction_stats['avg_keywords_per_transcript'] = (
                total_keywords / processed_count if processed_count > 0 else 0.0
            )
            
            logger.info(f"RadBERT extraction completed: {processed_count} transcripts, "
                       f"{total_keywords} total keywords, {len(unique_keywords)} unique keywords")
            
            return all_keywords
            
        except Exception as e:
            logger.warning(f"RadBERT batch processing failed: {e}")
            logger.warning("Falling back to pattern-based extraction")
            return self._extract_from_transcripts_patterns(transcripts)
    
    def _extract_from_transcripts_patterns(self, transcripts: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract keywords from multiple transcripts using basic pattern matching (fallback).
        
        Args:
            transcripts: Dictionary mapping transcript paths to text content
            
        Returns:
            Dictionary mapping transcript paths to extracted keywords
        """
        all_keywords = {}
        total_keywords = 0
        
        logger.info(f"Processing {len(transcripts)} transcripts with pattern matching")
        
        for transcript_path, text in transcripts.items():
            if text is not None:
                keywords = self._extract_with_patterns(text)
                all_keywords[transcript_path] = keywords
                
                # Count total keywords
                for category_keywords in keywords.values():
                    total_keywords += len(category_keywords)
            else:
                all_keywords[transcript_path] = {'anatomy': [], 'pathology': [], 'devices': [], 'descriptors': [], 'location': []}
        
        # Update statistics
        self.extraction_stats['total_transcripts_processed'] = len(transcripts)
        self.extraction_stats['total_keywords_extracted'] = total_keywords
        self.extraction_stats['avg_keywords_per_transcript'] = total_keywords / len(transcripts) if transcripts else 0
        
        # Count unique keywords
        all_unique_keywords = set()
        for keywords_dict in all_keywords.values():
            for category_keywords in keywords_dict.values():
                all_unique_keywords.update(category_keywords)
        self.extraction_stats['unique_keywords'] = len(all_unique_keywords)
        
        logger.info(f"Pattern extraction completed: {len(transcripts)} transcripts")
        logger.info(f"Total keywords: {total_keywords}, Unique: {len(all_unique_keywords)}")
        logger.info(f"Average keywords per transcript: {self.extraction_stats['avg_keywords_per_transcript']:.2f}")
        
        return all_keywords
    
    def save_keywords_to_json(self, keywords: Dict[str, Dict[str, List[str]]], output_path: str):
        """
        Save extracted keywords to JSON file.
        
        Args:
            keywords: Extracted keywords dictionary
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        extractor_type = 'radbert_extraction' if self.use_radbert else 'basic_pattern_matching'
        output_data = {
            'extraction_stats': self.extraction_stats,
            'keywords_by_transcript': keywords,
            'metadata': {
                'extractor_type': extractor_type,
                'total_transcripts': len(keywords),
                'categories': ['anatomy', 'pathology', 'devices', 'descriptors', 'location'],
                'radbert_used': self.use_radbert
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Keywords saved to {output_path}")
    
    def get_keyword_statistics(self, keywords: Dict[str, Dict[str, List[str]]]) -> Dict[str, Union[int, float, Dict]]:
        """
        Get statistics about extracted keywords.
        
        Args:
            keywords: Extracted keywords dictionary
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_transcripts': len(keywords),
            'total_keywords': 0,
            'keywords_by_category': defaultdict(int),
            'most_common_keywords': {},
            'transcripts_with_keywords': 0
        }
        
        all_keywords_by_category = defaultdict(list)
        
        for transcript_keywords in keywords.values():
            has_keywords = False
            for category, category_keywords in transcript_keywords.items():
                if category_keywords:
                    has_keywords = True
                    stats['total_keywords'] += len(category_keywords)
                    stats['keywords_by_category'][category] += len(category_keywords)
                    all_keywords_by_category[category].extend(category_keywords)
            
            if has_keywords:
                stats['transcripts_with_keywords'] += 1
        
        # Find most common keywords per category
        for category, category_keywords in all_keywords_by_category.items():
            counter = Counter(category_keywords)
            stats['most_common_keywords'][category] = counter.most_common(10)
        
        stats['avg_keywords_per_transcript'] = stats['total_keywords'] / len(keywords) if keywords else 0
        stats['coverage'] = stats['transcripts_with_keywords'] / len(keywords) if keywords else 0
        
        return dict(stats)


# Convenience functions
def extract_keywords_from_transcript_file(transcript_path: str, text: str) -> Dict[str, List[str]]:
    """
    Convenience function to extract keywords from a single transcript.
    
    Args:
        transcript_path: Path to transcript file (for logging)
        text: Transcript text content
        
    Returns:
        Extracted keywords by category
    """
    extractor = KeywordExtractor()
    return extractor.extract_keywords_from_text(text)


def create_basic_keyword_dataset(transcripts: Dict[str, str], output_dir: str = "output/keywords") -> str:
    """
    Convenience function to create a basic keyword dataset from transcripts.
    
    Args:
        transcripts: Dictionary mapping transcript paths to text content
        output_dir: Directory to save keyword files
        
    Returns:
        Path to saved keywords file
    """
    extractor = KeywordExtractor()
    keywords = extractor.extract_keywords_from_transcripts(transcripts)
    
    output_path = Path(output_dir) / "basic_keywords.json"
    extractor.save_keywords_to_json(keywords, str(output_path))
    
    return str(output_path)


if __name__ == "__main__":
    # Example usage and testing
    from .transcript_processor import TranscriptProcessor
    
    print("Testing keyword extraction...")
    
    # Load some transcripts
    processor = TranscriptProcessor()
    transcript_paths = processor.find_transcript_files("../../dataset_splits/train.csv")[:5]
    transcripts = processor.load_multiple_transcripts(transcript_paths)
    
    # Extract keywords
    extractor = KeywordExtractor()
    keywords = extractor.extract_keywords_from_transcripts(transcripts)
    
    # Show results
    for transcript_path, transcript_keywords in keywords.items():
        print(f"\n{transcript_path}:")
        for category, category_keywords in transcript_keywords.items():
            if category_keywords:
                print(f"  {category}: {category_keywords[:5]}...")  # Show first 5
    
    # Save to file
    output_path = "output/keywords/basic_keywords.json"
    extractor.save_keywords_to_json(keywords, output_path)
    print(f"\nKeywords saved to {output_path}")
    
    # Show statistics
    stats = extractor.get_keyword_statistics(keywords)
    print(f"\nStatistics:")
    print(f"  Total keywords: {stats['total_keywords']}")
    print(f"  Average per transcript: {stats['avg_keywords_per_transcript']:.2f}")
    print(f"  Coverage: {stats['coverage']:.2%}") 