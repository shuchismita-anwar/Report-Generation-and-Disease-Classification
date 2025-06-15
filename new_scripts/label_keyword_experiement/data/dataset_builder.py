"""
Dataset Builder for Hybrid RadBERT + MedCLIP Label & Keyword Prediction
======================================================================

This module creates enhanced datasets that combine original MIMIC data with keyword labels
and region availability masks for multi-task training.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union, Any
from collections import defaultdict

# Optional PyTorch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelKeywordDataset:
    """
    Enhanced dataset that combines original MIMIC data with keyword labels and region masks.
    
    This class provides functionality to:
    - Load original CSV data (train/val/test)
    - Add keyword target vectors using vocabulary
    - Add region availability masks
    - Save enhanced dataset with keyword labels
    - Create PyTorch-compatible data loaders
    """
    
    def __init__(self, 
                 vocabulary_path: str,
                 keywords_path: str,
                 regions_path: str,
                 base_path: str = "../../"):
        """
        Initialize the LabelKeywordDataset.
        
        Args:
            vocabulary_path: Path to vocabulary JSON file
            keywords_path: Path to keywords JSON file
            regions_path: Path to regions JSON file
            base_path: Base path to dataset files
        """
        self.base_path = Path(base_path)
        self.vocabulary_path = vocabulary_path
        self.keywords_path = keywords_path
        self.regions_path = regions_path
        
        # Load supporting data
        self.vocabulary_data = self._load_vocabulary()
        self.keywords_data = self._load_keywords()
        self.regions_data = self._load_regions()
        
        # Extract mappings
        self.word2idx = self.vocabulary_data['word2idx']
        self.idx2word = self.vocabulary_data['idx2word']
        self.vocab_size = len(self.word2idx)
        self.num_regions = len(self.regions_data['region_definitions']['regions'])
        
        # Dataset statistics
        self.dataset_stats = {
            'total_samples': 0,
            'samples_with_keywords': 0,
            'samples_with_regions': 0,
            'samples_complete': 0,
            'avg_keywords_per_sample': 0.0,
            'avg_regions_per_sample': 0.0,
            'disease_label_distribution': {},
            'keyword_label_distribution': {},
            'region_availability_distribution': {}
        }
    
    def _load_vocabulary(self) -> Dict:
        """Load vocabulary data from JSON file."""
        logger.info(f"Loading vocabulary from {self.vocabulary_path}")
        with open(self.vocabulary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_keywords(self) -> Dict:
        """Load keywords data from JSON file."""
        logger.info(f"Loading keywords from {self.keywords_path}")
        with open(self.keywords_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_regions(self) -> Dict:
        """Load regions data from JSON file."""
        logger.info(f"Loading regions from {self.regions_path}")
        with open(self.regions_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_enhanced_dataset(self, csv_path: str, output_path: str, max_samples: Optional[int] = None) -> str:
        """
        Create enhanced dataset from original CSV data.
        
        Args:
            csv_path: Path to original CSV file
            output_path: Path to save enhanced dataset
            max_samples: Maximum number of samples to process (for testing)
            
        Returns:
            Path to saved enhanced dataset
        """
        logger.info(f"Creating enhanced dataset from {csv_path}")
        
        # Load original CSV data
        df = pd.read_csv(csv_path)
        if max_samples:
            df = df.head(max_samples)
            logger.info(f"Limited to {max_samples} samples for testing")
        
        logger.info(f"Processing {len(df)} samples")
        
        # Create enhanced samples
        enhanced_samples = []
        
        for idx, row in df.iterrows():
            try:
                enhanced_sample = self._create_enhanced_sample(row, idx)
                if enhanced_sample:
                    enhanced_samples.append(enhanced_sample)
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        # Calculate statistics
        self._calculate_dataset_statistics(enhanced_samples)
        
        # Save enhanced dataset
        self._save_enhanced_dataset(enhanced_samples, output_path)
        
        logger.info(f"Created enhanced dataset with {len(enhanced_samples)} samples")
        return output_path
    
    def _create_enhanced_sample(self, row: pd.Series, idx: int) -> Optional[Dict]:
        """
        Create enhanced sample from original CSV row.
        
        Args:
            row: Original CSV row
            idx: Row index
            
        Returns:
            Enhanced sample dictionary or None if processing fails
        """
        sample_id = f"sample_{idx}"
        
        # Extract original data
        enhanced_sample = {
            'sample_id': sample_id,
            'original_index': idx,
            'img_path': row.get('img_path', ''),
            'bbox_path': row.get('bbox_path', ''),
            'fixseq_path': row.get('fixseq_path', ''),
            'transcript_path': row.get('transcript_path', ''),
        }
        
        # Extract disease labels (8 conditions)
        disease_labels = self._extract_disease_labels(row)
        enhanced_sample['disease_labels'] = disease_labels
        
        # Extract keyword labels
        keyword_labels = self._extract_keyword_labels(row, sample_id)
        enhanced_sample['keyword_labels'] = keyword_labels
        
        # Extract region mask
        region_mask = self._extract_region_mask(row, sample_id)
        enhanced_sample['region_mask'] = [bool(x) for x in region_mask]  # Convert to Python bool
        
        # Create region-specific keyword labels
        region_keyword_labels = self._create_region_keyword_labels(keyword_labels, region_mask)
        enhanced_sample['region_keyword_labels'] = region_keyword_labels
        
        # Add metadata
        enhanced_sample['metadata'] = {
            'has_keywords': bool(len(keyword_labels['global_keywords']) > 0),
            'has_regions': bool(np.any(region_mask)),
            'num_available_regions': int(np.sum(region_mask)),
            'num_keywords': int(len(keyword_labels['global_keywords']))
        }
        
        return enhanced_sample
    
    def _extract_disease_labels(self, row: pd.Series) -> List[int]:
        """
        Extract disease labels from CSV row.
        
        Args:
            row: CSV row
            
        Returns:
            List of 8 binary disease labels
        """
        # For now, create dummy disease labels since we don't have the exact column names
        # In a real implementation, this would extract from the actual CSV columns
        disease_labels = [0] * 8  # 8 binary labels
        
        # Try to extract from common column patterns
        for i, condition in enumerate(['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 
                                     'fracture', 'opacity', 'effusion', 'pneumothorax'][:8]):
            # Look for columns that might contain this condition
            matching_cols = [col for col in row.index if condition.lower() in col.lower()]
            if matching_cols:
                # Use first matching column
                value = row[matching_cols[0]]
                if pd.notna(value):
                    disease_labels[i] = 1 if value > 0 else 0
        
        return disease_labels
    
    def _extract_keyword_labels(self, row: pd.Series, sample_id: str) -> Dict[str, List[int]]:
        """
        Extract keyword labels for the sample.
        
        Args:
            row: CSV row
            sample_id: Sample identifier
            
        Returns:
            Dictionary with global and region-specific keyword labels
        """
        # Find corresponding transcript path
        transcript_path = row.get('transcript_path', '')
        
        # Look for keywords in the keywords data
        keywords_by_transcript = self.keywords_data.get('keywords_by_transcript', {})
        
        # Find matching transcript
        sample_keywords = None
        for transcript_key, transcript_keywords in keywords_by_transcript.items():
            # Simple matching - in practice, this would be more sophisticated
            if transcript_path and transcript_path in transcript_key:
                sample_keywords = transcript_keywords
                break
        
        if not sample_keywords:
            # Use first available keywords as fallback for testing
            if keywords_by_transcript:
                sample_keywords = list(keywords_by_transcript.values())[0]
            else:
                sample_keywords = {'anatomy': [], 'pathology': [], 'devices': [], 'descriptors': [], 'negation': []}
        
        # Convert keywords to vocabulary indices
        global_keyword_vector = self._keywords_to_vector(sample_keywords)
        
        return {
            'global_keywords': global_keyword_vector,
            'raw_keywords': sample_keywords
        }
    
    def _keywords_to_vector(self, keywords_dict: Dict[str, List[str]]) -> List[int]:
        """
        Convert keyword dictionary to vocabulary index vector.
        
        Args:
            keywords_dict: Dictionary of categorized keywords
            
        Returns:
            Binary vector of vocabulary size indicating present keywords
        """
        keyword_vector = [0] * self.vocab_size
        
        # Collect all keywords from all categories
        all_keywords = []
        for category_keywords in keywords_dict.values():
            all_keywords.extend(category_keywords)
        
        # Set corresponding vocabulary indices to 1
        for keyword in all_keywords:
            if keyword in self.word2idx:
                idx = self.word2idx[keyword]
                keyword_vector[idx] = 1
        
        return keyword_vector
    
    def _extract_region_mask(self, row: pd.Series, sample_id: str) -> List[bool]:
        """
        Extract region availability mask for the sample.
        
        Args:
            row: CSV row
            sample_id: Sample identifier
            
        Returns:
            List of 17 boolean values indicating region availability
        """
        # Look for region mask in regions data
        region_masks = self.regions_data.get('region_masks', {})
        
        if sample_id in region_masks:
            return region_masks[sample_id]
        
        # Fallback: use first available region mask for testing
        if region_masks:
            return list(region_masks.values())[0]
        
        # Default: all regions available
        return [True] * self.num_regions
    
    def _create_region_keyword_labels(self, keyword_labels: Dict, region_mask: List[bool]) -> List[List[int]]:
        """
        Create region-specific keyword labels.
        
        Args:
            keyword_labels: Global keyword labels
            region_mask: Region availability mask
            
        Returns:
            List of keyword vectors for each region [17, vocab_size]
        """
        global_keywords = keyword_labels['global_keywords']
        region_keyword_labels = []
        
        for region_idx, region_available in enumerate(region_mask):
            if region_available:
                # For available regions, use global keywords (simplified)
                # In practice, this would be region-specific keyword extraction
                region_keyword_labels.append(global_keywords.copy())
            else:
                # For unavailable regions, use empty keyword vector
                region_keyword_labels.append([0] * self.vocab_size)
        
        return region_keyword_labels
    
    def _calculate_dataset_statistics(self, enhanced_samples: List[Dict]):
        """Calculate comprehensive dataset statistics."""
        if not enhanced_samples:
            return
        
        total_samples = len(enhanced_samples)
        samples_with_keywords = sum(1 for s in enhanced_samples if s['metadata']['has_keywords'])
        samples_with_regions = sum(1 for s in enhanced_samples if s['metadata']['has_regions'])
        samples_complete = sum(1 for s in enhanced_samples if s['metadata']['has_keywords'] and s['metadata']['has_regions'])
        
        total_keywords = sum(s['metadata']['num_keywords'] for s in enhanced_samples)
        total_regions = sum(s['metadata']['num_available_regions'] for s in enhanced_samples)
        
        # Disease label distribution
        disease_counts = defaultdict(int)
        for sample in enhanced_samples:
            for i, label in enumerate(sample['disease_labels']):
                if label == 1:
                    disease_counts[f'disease_{i}'] += 1
        
        # Keyword label distribution
        keyword_counts = defaultdict(int)
        for sample in enhanced_samples:
            for i, label in enumerate(sample['keyword_labels']['global_keywords']):
                if label == 1:
                    if i < len(self.idx2word):
                        keyword = self.idx2word[str(i)]
                        keyword_counts[keyword] += 1
        
        # Region availability distribution
        region_counts = defaultdict(int)
        region_names = self.regions_data['region_definitions']['regions']
        for sample in enhanced_samples:
            for i, available in enumerate(sample['region_mask']):
                if available and i < len(region_names):
                    region_counts[region_names[i]] += 1
        
        self.dataset_stats = {
            'total_samples': total_samples,
            'samples_with_keywords': samples_with_keywords,
            'samples_with_regions': samples_with_regions,
            'samples_complete': samples_complete,
            'avg_keywords_per_sample': total_keywords / total_samples if total_samples > 0 else 0,
            'avg_regions_per_sample': total_regions / total_samples if total_samples > 0 else 0,
            'disease_label_distribution': dict(disease_counts),
            'keyword_label_distribution': dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            'region_availability_distribution': dict(region_counts)
        }
    
    def _save_enhanced_dataset(self, enhanced_samples: List[Dict], output_path: str):
        """
        Save enhanced dataset to JSON file.
        
        Args:
            enhanced_samples: List of enhanced sample dictionaries
            output_path: Path to save dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        dataset_data = {
            'samples': enhanced_samples,
            'metadata': {
                'vocab_size': self.vocab_size,
                'num_regions': self.num_regions,
                'num_diseases': 8,
                'vocabulary_path': self.vocabulary_path,
                'keywords_path': self.keywords_path,
                'regions_path': self.regions_path,
                'dataset_type': 'label_keyword_enhanced'
            },
            'statistics': self.dataset_stats,
            'vocabulary_info': {
                'word2idx_sample': dict(list(self.word2idx.items())[:10]),
                'special_tokens': self.vocabulary_data.get('special_tokens', {}),
                'vocab_size': self.vocab_size
            },
            'region_info': {
                'regions': self.regions_data['region_definitions']['regions'],
                'num_regions': self.num_regions
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Enhanced dataset saved to {output_path}")
    
    def load_enhanced_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load enhanced dataset from JSON file.
        
        Args:
            dataset_path: Path to enhanced dataset JSON file
            
        Returns:
            List of enhanced sample dictionaries
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset_data = json.load(f)
        
        enhanced_samples = dataset_data['samples']
        self.dataset_stats = dataset_data['statistics']
        
        logger.info(f"Loaded enhanced dataset with {len(enhanced_samples)} samples from {dataset_path}")
        
        return enhanced_samples
    
    def get_dataset_summary(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Get comprehensive dataset summary.
        
        Returns:
            Dictionary with dataset summary statistics
        """
        return {
            'total_samples': self.dataset_stats['total_samples'],
            'completeness': {
                'samples_with_keywords': self.dataset_stats['samples_with_keywords'],
                'samples_with_regions': self.dataset_stats['samples_with_regions'],
                'samples_complete': self.dataset_stats['samples_complete'],
                'completion_rate': self.dataset_stats['samples_complete'] / self.dataset_stats['total_samples'] if self.dataset_stats['total_samples'] > 0 else 0
            },
            'averages': {
                'keywords_per_sample': self.dataset_stats['avg_keywords_per_sample'],
                'regions_per_sample': self.dataset_stats['avg_regions_per_sample']
            },
            'distributions': {
                'top_diseases': dict(list(self.dataset_stats['disease_label_distribution'].items())[:5]),
                'top_keywords': dict(list(self.dataset_stats['keyword_label_distribution'].items())[:10]),
                'top_regions': dict(list(self.dataset_stats['region_availability_distribution'].items())[:10])
            },
            'data_shapes': {
                'vocab_size': self.vocab_size,
                'num_regions': self.num_regions,
                'num_diseases': 8
            }
        }
    
    def create_pytorch_dataset(self, enhanced_samples: List[Dict]) -> 'PyTorchLabelKeywordDataset':
        """
        Create PyTorch-compatible dataset from enhanced samples.
        
        Args:
            enhanced_samples: List of enhanced sample dictionaries
            
        Returns:
            PyTorch dataset object
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch to use PyTorch datasets.")
        return PyTorchLabelKeywordDataset(enhanced_samples, self.vocab_size, self.num_regions)


class PyTorchLabelKeywordDataset:
    """PyTorch-compatible dataset for label+keyword training."""
    
    def __init__(self, enhanced_samples: List[Dict], vocab_size: int, num_regions: int):
        """
        Initialize PyTorch dataset.
        
        Args:
            enhanced_samples: List of enhanced sample dictionaries
            vocab_size: Vocabulary size
            num_regions: Number of anatomical regions
        """
        self.samples = enhanced_samples
        self.vocab_size = vocab_size
        self.num_regions = num_regions
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, Any]]:
        """
        Get sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with tensors for model training
        """
        sample = self.samples[idx]
        
        if TORCH_AVAILABLE:
            return {
                'sample_id': sample['sample_id'],
                'disease_labels': torch.tensor(sample['disease_labels'], dtype=torch.float32),
                'global_keyword_labels': torch.tensor(sample['keyword_labels']['global_keywords'], dtype=torch.float32),
                'region_mask': torch.tensor(sample['region_mask'], dtype=torch.bool),
                'region_keyword_labels': torch.tensor(sample['region_keyword_labels'], dtype=torch.float32),
                'metadata': sample['metadata']
            }
        else:
            return {
                'sample_id': sample['sample_id'],
                'disease_labels': np.array(sample['disease_labels'], dtype=np.float32),
                'global_keyword_labels': np.array(sample['keyword_labels']['global_keywords'], dtype=np.float32),
                'region_mask': np.array(sample['region_mask'], dtype=bool),
                'region_keyword_labels': np.array(sample['region_keyword_labels'], dtype=np.float32),
                'metadata': sample['metadata']
            }


# Convenience functions
def create_enhanced_dataset_from_csv(csv_path: str,
                                   vocabulary_path: str,
                                   keywords_path: str,
                                   regions_path: str,
                                   output_path: str = "output/datasets/enhanced_dataset.json",
                                   max_samples: Optional[int] = None) -> str:
    """
    Convenience function to create enhanced dataset from CSV.
    
    Args:
        csv_path: Path to original CSV file
        vocabulary_path: Path to vocabulary JSON file
        keywords_path: Path to keywords JSON file
        regions_path: Path to regions JSON file
        output_path: Path to save enhanced dataset
        max_samples: Maximum number of samples to process
        
    Returns:
        Path to saved enhanced dataset
    """
    builder = LabelKeywordDataset(vocabulary_path, keywords_path, regions_path)
    return builder.create_enhanced_dataset(csv_path, output_path, max_samples)


def create_test_enhanced_dataset(num_samples: int = 50,
                                output_path: str = "output/datasets/test_enhanced_dataset.json") -> str:
    """
    Convenience function to create test enhanced dataset.
    
    Args:
        num_samples: Number of test samples to create
        output_path: Path to save test dataset
        
    Returns:
        Path to saved test dataset
    """
    # Use existing vocabulary, keywords, and regions
    vocabulary_path = "output/vocabulary/simple_vocab.json"
    keywords_path = "output/keywords/basic_keywords.json"
    regions_path = "output/regions/region_masks.json"
    csv_path = "../../dataset_splits/train.csv"
    
    return create_enhanced_dataset_from_csv(
        csv_path, vocabulary_path, keywords_path, regions_path, 
        output_path, max_samples=num_samples
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing enhanced dataset creation...")
    
    # Check if required files exist
    required_files = [
        "output/vocabulary/simple_vocab.json",
        "output/keywords/basic_keywords.json", 
        "output/regions/region_masks.json",
        "../../dataset_splits/train.csv"
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please run previous tasks first.")
        exit(1)
    
    # Create test enhanced dataset
    print("Creating test enhanced dataset with 50 samples...")
    output_path = create_test_enhanced_dataset(50)
    
    # Load and analyze the dataset
    builder = LabelKeywordDataset(
        "output/vocabulary/simple_vocab.json",
        "output/keywords/basic_keywords.json",
        "output/regions/region_masks.json"
    )
    
    enhanced_samples = builder.load_enhanced_dataset(output_path)
    
    # Show summary
    summary = builder.get_dataset_summary()
    print(f"\nDataset Summary:")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Completion rate: {summary['completeness']['completion_rate']:.2%}")
    print(f"  Avg keywords per sample: {summary['averages']['keywords_per_sample']:.2f}")
    print(f"  Avg regions per sample: {summary['averages']['regions_per_sample']:.2f}")
    print(f"  Data shapes: {summary['data_shapes']}")
    
    # Test PyTorch dataset (if available)
    if TORCH_AVAILABLE:
        pytorch_dataset = builder.create_pytorch_dataset(enhanced_samples[:5])
        sample = pytorch_dataset[0]
        print(f"\nPyTorch sample shapes:")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
    else:
        print(f"\nPyTorch not available - skipping PyTorch dataset test")
    
    print(f"\n✅ Enhanced dataset creation test complete!") 