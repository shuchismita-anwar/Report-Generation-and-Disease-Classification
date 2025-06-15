"""
Vocabulary Builder for Hybrid RadBERT + MedCLIP Label & Keyword Prediction
=========================================================================

This module handles building medical vocabularies from extracted keywords.
Creates word2idx and idx2word mappings for model training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union
from collections import Counter, defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VocabularyBuilder:
    """
    Builds medical vocabulary from extracted keywords.
    
    This class provides functionality to:
    - Aggregate keywords from multiple transcripts
    - Count keyword frequencies across all data
    - Filter by minimum frequency thresholds
    - Create word2idx and idx2word mappings
    - Save vocabulary artifacts to JSON format
    """
    
    def __init__(self, min_frequency: int = 2, max_vocab_size: int = 2000):
        """
        Initialize the VocabularyBuilder.
        
        Args:
            min_frequency: Minimum frequency for a keyword to be included
            max_vocab_size: Maximum vocabulary size
        """
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,    # Padding token
            '<UNK>': 1,    # Unknown token
            '<START>': 2,  # Start token (for sequence generation)
            '<END>': 3     # End token (for sequence generation)
        }
        
        # Vocabulary mappings
        self.word2idx = {}
        self.idx2word = {}
        self.word_frequencies = Counter()
        self.category_frequencies = defaultdict(Counter)
        
        # Statistics
        self.vocab_stats = {
            'total_keywords_processed': 0,
            'unique_keywords_before_filtering': 0,
            'unique_keywords_after_filtering': 0,
            'final_vocab_size': 0,
            'keywords_by_category': {},
            'frequency_distribution': {},
            'coverage_stats': {}
        }
    
    def build_vocabulary_from_keywords(self, keywords_data: Dict[str, Dict[str, List[str]]]) -> Dict[str, Union[Dict, int, float]]:
        """
        Build vocabulary from extracted keywords data.
        
        Args:
            keywords_data: Dictionary mapping transcript paths to categorized keywords
            
        Returns:
            Dictionary containing vocabulary mappings and statistics
        """
        logger.info("Building vocabulary from extracted keywords...")
        
        # Reset counters
        self.word_frequencies.clear()
        self.category_frequencies.clear()
        
        # Count all keywords
        total_keywords = 0
        for transcript_path, transcript_keywords in keywords_data.items():
            for category, category_keywords in transcript_keywords.items():
                for keyword in category_keywords:
                    # Clean and normalize keyword
                    cleaned_keyword = self._clean_keyword(keyword)
                    if cleaned_keyword:
                        self.word_frequencies[cleaned_keyword] += 1
                        self.category_frequencies[category][cleaned_keyword] += 1
                        total_keywords += 1
        
        self.vocab_stats['total_keywords_processed'] = total_keywords
        self.vocab_stats['unique_keywords_before_filtering'] = len(self.word_frequencies)
        
        logger.info(f"Processed {total_keywords} total keywords")
        logger.info(f"Found {len(self.word_frequencies)} unique keywords before filtering")
        
        # Filter by frequency and build vocabulary
        self._build_filtered_vocabulary()
        
        # Create mappings
        self._create_vocabulary_mappings()
        
        # Calculate statistics
        self._calculate_vocabulary_statistics()
        
        logger.info(f"Final vocabulary size: {self.vocab_stats['final_vocab_size']}")
        
        return self._get_vocabulary_data()
    
    def build_vocabulary_from_keywords_list(self, keywords_list: List[str]) -> Dict[str, Union[Dict, int, float]]:
        """
        Build vocabulary from a simple list of keywords.
        
        Args:
            keywords_list: Simple list of keyword strings
            
        Returns:
            Dictionary containing vocabulary mappings and statistics
        """
        logger.info("Building vocabulary from extracted keywords...")
        
        # Convert simple list to expected format
        keywords_data = {
            'sample_transcript': {
                'medical_terms': keywords_list
            }
        }
        
        return self.build_vocabulary_from_keywords(keywords_data)
    
    def build_vocabulary_from_json(self, keywords_json_path: str) -> Dict[str, Union[Dict, int, float]]:
        """
        Build vocabulary from saved keywords JSON file.
        
        Args:
            keywords_json_path: Path to keywords JSON file
            
        Returns:
            Dictionary containing vocabulary mappings and statistics
        """
        logger.info(f"Loading keywords from {keywords_json_path}")
        
        with open(keywords_json_path, 'r', encoding='utf-8') as f:
            keywords_file = json.load(f)
        
        keywords_data = keywords_file['keywords_by_transcript']
        return self.build_vocabulary_from_keywords(keywords_data)
    
    def _clean_keyword(self, keyword: str) -> Optional[str]:
        """
        Clean and normalize a keyword.
        
        Args:
            keyword: Raw keyword string
            
        Returns:
            Cleaned keyword or None if invalid
        """
        if not keyword or not isinstance(keyword, str):
            return None
        
        # Convert to lowercase and strip
        cleaned = keyword.lower().strip()
        
        # Filter out very short or very long keywords
        if len(cleaned) < 2 or len(cleaned) > 50:
            return None
        
        # Filter out purely numeric keywords
        if cleaned.isdigit():
            return None
        
        # Filter out keywords with too many special characters
        special_char_count = sum(1 for c in cleaned if not c.isalnum() and c not in [' ', '-', '_'])
        if special_char_count > len(cleaned) // 2:
            return None
        
        return cleaned
    
    def _build_filtered_vocabulary(self):
        """Build filtered vocabulary based on frequency thresholds."""
        # Filter by minimum frequency
        filtered_words = {
            word: freq for word, freq in self.word_frequencies.items()
            if freq >= self.min_frequency
        }
        
        # Sort by frequency (descending) and take top max_vocab_size
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        
        # Take top words up to max_vocab_size (accounting for special tokens)
        available_slots = self.max_vocab_size - len(self.special_tokens)
        top_words = sorted_words[:available_slots]
        
        # Update word frequencies to only include selected words
        self.word_frequencies = Counter(dict(top_words))
        
        self.vocab_stats['unique_keywords_after_filtering'] = len(self.word_frequencies)
        
        logger.info(f"After filtering: {len(self.word_frequencies)} keywords remain")
    
    def _create_vocabulary_mappings(self):
        """Create word2idx and idx2word mappings."""
        # Start with special tokens
        self.word2idx = self.special_tokens.copy()
        self.idx2word = {idx: word for word, idx in self.special_tokens.items()}
        
        # Add regular vocabulary words
        current_idx = len(self.special_tokens)
        
        # Sort words by frequency (most frequent first)
        sorted_words = sorted(self.word_frequencies.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words:
            self.word2idx[word] = current_idx
            self.idx2word[current_idx] = word
            current_idx += 1
        
        self.vocab_stats['final_vocab_size'] = len(self.word2idx)
    
    def _calculate_vocabulary_statistics(self):
        """Calculate detailed vocabulary statistics."""
        # Keywords by category (after filtering)
        category_stats = {}
        for category, category_counter in self.category_frequencies.items():
            # Only count keywords that made it into final vocabulary
            filtered_category_keywords = {
                word: freq for word, freq in category_counter.items()
                if word in self.word2idx
            }
            category_stats[category] = {
                'count': len(filtered_category_keywords),
                'total_frequency': sum(filtered_category_keywords.values()),
                'top_keywords': sorted(filtered_category_keywords.items(), key=lambda x: x[1], reverse=True)[:10]
            }
        
        self.vocab_stats['keywords_by_category'] = category_stats
        
        # Frequency distribution
        freq_dist = Counter(self.word_frequencies.values())
        self.vocab_stats['frequency_distribution'] = {
            'min_frequency': min(self.word_frequencies.values()) if self.word_frequencies else 0,
            'max_frequency': max(self.word_frequencies.values()) if self.word_frequencies else 0,
            'avg_frequency': sum(self.word_frequencies.values()) / len(self.word_frequencies) if self.word_frequencies else 0,
            'frequency_bins': dict(freq_dist)
        }
        
        # Coverage statistics
        total_keyword_instances = sum(self.word_frequencies.values())
        self.vocab_stats['coverage_stats'] = {
            'total_keyword_instances': total_keyword_instances,
            'vocab_coverage': len(self.word_frequencies) / self.vocab_stats['unique_keywords_before_filtering'] if self.vocab_stats['unique_keywords_before_filtering'] > 0 else 0,
            'frequency_coverage': total_keyword_instances / self.vocab_stats['total_keywords_processed'] if self.vocab_stats['total_keywords_processed'] > 0 else 0
        }
    
    def _get_vocabulary_data(self) -> Dict[str, Union[Dict, int, float]]:
        """Get complete vocabulary data for saving."""
        return {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_frequencies': dict(self.word_frequencies),
            'category_frequencies': {cat: dict(counter) for cat, counter in self.category_frequencies.items()},
            'special_tokens': self.special_tokens,
            'config': {
                'min_frequency': self.min_frequency,
                'max_vocab_size': self.max_vocab_size
            },
            'statistics': self.vocab_stats
        }
    
    def save_vocabulary(self, vocabulary_data: Dict, output_path: str):
        """
        Save vocabulary to JSON file.
        
        Args:
            vocabulary_data: Vocabulary data dictionary
            output_path: Path to save vocabulary JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocabulary_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Vocabulary saved to {output_path}")
    
    def load_vocabulary(self, vocabulary_path: str) -> Dict[str, Union[Dict, int, float]]:
        """
        Load vocabulary from JSON file.
        
        Args:
            vocabulary_path: Path to vocabulary JSON file
            
        Returns:
            Vocabulary data dictionary
        """
        with open(vocabulary_path, 'r', encoding='utf-8') as f:
            vocabulary_data = json.load(f)
        
        # Restore internal state
        self.word2idx = vocabulary_data['word2idx']
        self.idx2word = {int(k): v for k, v in vocabulary_data['idx2word'].items()}  # JSON keys are strings
        self.word_frequencies = Counter(vocabulary_data['word_frequencies'])
        self.category_frequencies = {
            cat: Counter(counter) for cat, counter in vocabulary_data['category_frequencies'].items()
        }
        self.special_tokens = vocabulary_data['special_tokens']
        self.min_frequency = vocabulary_data['config']['min_frequency']
        self.max_vocab_size = vocabulary_data['config']['max_vocab_size']
        self.vocab_stats = vocabulary_data['statistics']
        
        logger.info(f"Vocabulary loaded from {vocabulary_path}")
        logger.info(f"Vocabulary size: {len(self.word2idx)}")
        
        return vocabulary_data
    
    def get_word_index(self, word: str) -> int:
        """
        Get index for a word.
        
        Args:
            word: Word to look up
            
        Returns:
            Index of word, or UNK token index if not found
        """
        cleaned_word = self._clean_keyword(word)
        return self.word2idx.get(cleaned_word, self.special_tokens['<UNK>'])
    
    def get_index_word(self, index: int) -> str:
        """
        Get word for an index.
        
        Args:
            index: Index to look up
            
        Returns:
            Word at index, or UNK token if not found
        """
        return self.idx2word.get(index, '<UNK>')
    
    def encode_keywords(self, keywords: List[str], max_length: Optional[int] = None) -> List[int]:
        """
        Encode a list of keywords to indices.
        
        Args:
            keywords: List of keyword strings
            max_length: Maximum sequence length (with padding)
            
        Returns:
            List of keyword indices
        """
        indices = [self.get_word_index(keyword) for keyword in keywords]
        
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                # Pad with PAD token
                indices.extend([self.special_tokens['<PAD>']] * (max_length - len(indices)))
        
        return indices
    
    def decode_keywords(self, indices: List[int], remove_special: bool = True) -> List[str]:
        """
        Decode a list of indices to keywords.
        
        Args:
            indices: List of keyword indices
            remove_special: Whether to remove special tokens
            
        Returns:
            List of keyword strings
        """
        words = [self.get_index_word(idx) for idx in indices]
        
        if remove_special:
            special_token_set = set(self.special_tokens.keys())
            words = [word for word in words if word not in special_token_set]
        
        return words
    
    def get_vocabulary_summary(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Get a summary of the vocabulary.
        
        Returns:
            Dictionary with vocabulary summary statistics
        """
        return {
            'vocab_size': len(self.word2idx),
            'special_tokens': len(self.special_tokens),
            'regular_tokens': len(self.word2idx) - len(self.special_tokens),
            'min_frequency': self.vocab_stats['frequency_distribution']['min_frequency'],
            'max_frequency': self.vocab_stats['frequency_distribution']['max_frequency'],
            'avg_frequency': self.vocab_stats['frequency_distribution']['avg_frequency'],
            'categories': list(self.vocab_stats['keywords_by_category'].keys()),
            'category_counts': {cat: stats['count'] for cat, stats in self.vocab_stats['keywords_by_category'].items()},
            'coverage': self.vocab_stats['coverage_stats']['vocab_coverage'],
            'most_frequent_words': sorted(self.word_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
        }


# Convenience functions
def build_vocabulary_from_keywords_file(keywords_json_path: str, 
                                       output_path: str = "output/vocabulary/simple_vocab.json",
                                       min_frequency: int = 2,
                                       max_vocab_size: int = 2000) -> str:
    """
    Convenience function to build vocabulary from keywords JSON file.
    
    Args:
        keywords_json_path: Path to keywords JSON file
        output_path: Path to save vocabulary JSON file
        min_frequency: Minimum frequency for inclusion
        max_vocab_size: Maximum vocabulary size
        
    Returns:
        Path to saved vocabulary file
    """
    builder = VocabularyBuilder(min_frequency=min_frequency, max_vocab_size=max_vocab_size)
    vocabulary_data = builder.build_vocabulary_from_json(keywords_json_path)
    builder.save_vocabulary(vocabulary_data, output_path)
    return output_path


def create_simple_vocabulary_from_keywords(keywords_data: Dict[str, Dict[str, List[str]]],
                                         output_path: str = "output/vocabulary/simple_vocab.json") -> str:
    """
    Convenience function to create vocabulary directly from keywords data.
    
    Args:
        keywords_data: Dictionary mapping transcript paths to categorized keywords
        output_path: Path to save vocabulary JSON file
        
    Returns:
        Path to saved vocabulary file
    """
    builder = VocabularyBuilder()
    vocabulary_data = builder.build_vocabulary_from_keywords(keywords_data)
    builder.save_vocabulary(vocabulary_data, output_path)
    return output_path


if __name__ == "__main__":
    # Example usage and testing
    print("Testing vocabulary building...")
    
    # Build vocabulary from existing keywords file
    keywords_file = "output/keywords/basic_keywords.json"
    if Path(keywords_file).exists():
        builder = VocabularyBuilder(min_frequency=1, max_vocab_size=500)  # Lower thresholds for testing
        vocabulary_data = builder.build_vocabulary_from_json(keywords_file)
        
        # Save vocabulary
        vocab_output_path = "output/vocabulary/simple_vocab.json"
        builder.save_vocabulary(vocabulary_data, vocab_output_path)
        
        # Show summary
        summary = builder.get_vocabulary_summary()
        print(f"\nVocabulary Summary:")
        print(f"  Total vocabulary size: {summary['vocab_size']}")
        print(f"  Regular tokens: {summary['regular_tokens']}")
        print(f"  Categories: {summary['categories']}")
        print(f"  Category counts: {summary['category_counts']}")
        print(f"  Most frequent words: {[word for word, freq in summary['most_frequent_words'][:10]]}")
        
        # Test encoding/decoding
        test_keywords = ['heart', 'lung', 'pneumothorax', 'unknown_word']
        encoded = builder.encode_keywords(test_keywords)
        decoded = builder.decode_keywords(encoded)
        print(f"\nEncoding test:")
        print(f"  Original: {test_keywords}")
        print(f"  Encoded: {encoded}")
        print(f"  Decoded: {decoded}")
        
        print(f"\n✅ Vocabulary building test complete!")
    else:
        print(f"Keywords file not found: {keywords_file}")
        print("Please run keyword extraction first.") 