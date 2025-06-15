"""
Region Handler for Hybrid RadBERT + MedCLIP Label & Keyword Prediction
=====================================================================

This module handles anatomical region availability detection and management.
Creates region availability masks for the 17 anatomical regions.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union
from collections import defaultdict, Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegionHandler:
    """
    Handles anatomical region availability detection and management.
    
    This class provides functionality to:
    - Load bbox paths from dataset CSV files
    - Detect available regions per sample from actual bbox files
    - Create region availability masks (17 regions)
    - Generate consistent region representations
    - Handle missing regions and different data formats (REFLACX vs EyeGaze)
    """
    
    # Define the 17 anatomical regions (standardized names)
    ANATOMICAL_REGIONS = [
        'cardiac silhouette',           # 0
        'left clavicle',               # 1
        'right clavicle',              # 2
        'left costophrenic angle',     # 3
        'right costophrenic angle',    # 4
        'left hilar structures',       # 5
        'right hilar structures',      # 6
        'left lower lung zone',        # 7
        'left mid lung zone',          # 8
        'left upper lung zone',        # 9
        'right lower lung zone',       # 10
        'right mid lung zone',         # 11
        'right upper lung zone',       # 12
        'left lung',                   # 13
        'right lung',                  # 14
        'trachea',                     # 15
        'upper mediastinum'            # 16
    ]
    
    def __init__(self, base_path: str = "../../"):
        """
        Initialize the RegionHandler.
        
        Args:
            base_path: Base path to the dataset files
        """
        self.base_path = Path(base_path)
        self.num_regions = len(self.ANATOMICAL_REGIONS)
        
        # Region mappings
        self.region_to_idx = {region: idx for idx, region in enumerate(self.ANATOMICAL_REGIONS)}
        self.idx_to_region = {idx: region for idx, region in enumerate(self.ANATOMICAL_REGIONS)}
        
        # Statistics
        self.region_stats = {
            'total_samples_processed': 0,
            'samples_with_bbox': 0,
            'samples_without_bbox': 0,
            'region_availability_counts': defaultdict(int),
            'dataset_type_stats': defaultdict(dict),
            'region_coverage_by_dataset': {}
        }
        
        # Cache for processed region masks
        self._region_mask_cache = {}
    
    def load_bbox_paths_from_csv(self, csv_path: str) -> Dict[str, str]:
        """
        Load bbox paths from dataset CSV file.
        
        Args:
            csv_path: Path to CSV file containing bbox paths
            
        Returns:
            Dictionary mapping sample IDs to bbox file paths
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return {}
        
        logger.info(f"Loading bbox paths from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Check for bbox column
            bbox_columns = [col for col in df.columns if 'bbox' in col.lower()]
            if not bbox_columns:
                logger.warning(f"No bbox column found in {csv_path}")
                return {}
            
            bbox_col = bbox_columns[0]  # Use first bbox column found
            logger.info(f"Using bbox column: {bbox_col}")
            
            # Create mapping from index/ID to bbox path
            bbox_paths = {}
            for idx, row in df.iterrows():
                bbox_path = row[bbox_col]
                if pd.notna(bbox_path) and str(bbox_path).strip():
                    # Use row index as sample ID for now
                    sample_id = f"sample_{idx}"
                    bbox_paths[sample_id] = str(bbox_path).strip()
            
            logger.info(f"Loaded {len(bbox_paths)} bbox paths from {len(df)} total samples")
            return bbox_paths
            
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}")
            return {}
    
    def detect_available_regions_from_bbox(self, bbox_path: str) -> List[bool]:
        """
        Detect available regions from actual bbox file.
        
        Args:
            bbox_path: Path to bbox file
            
        Returns:
            List of 17 boolean values indicating region availability
        """
        full_bbox_path = self.base_path / bbox_path
        
        # Check if bbox file exists
        if not full_bbox_path.exists():
            logger.debug(f"Bbox file not found: {full_bbox_path}")
            return [False] * self.num_regions
        
        try:
            # Determine dataset type and load accordingly
            dataset_type = self._determine_dataset_type(bbox_path)
            
            if dataset_type == 'reflacx':
                return self._detect_regions_reflacx(full_bbox_path)
            elif dataset_type == 'eyegaze':
                return self._detect_regions_eyegaze(full_bbox_path)
            else:
                logger.warning(f"Unknown dataset type for {bbox_path}")
                return [False] * self.num_regions
                
        except Exception as e:
            logger.error(f"Error processing bbox file {bbox_path}: {e}")
            return [False] * self.num_regions
    
    def _determine_dataset_type(self, bbox_path: str) -> str:
        """
        Determine dataset type from bbox path.
        
        Args:
            bbox_path: Path to bbox file
            
        Returns:
            Dataset type string
        """
        bbox_path_lower = bbox_path.lower()
        
        if 'reflacx' in bbox_path_lower:
            return 'reflacx'
        elif 'eyegaze' in bbox_path_lower:
            return 'eyegaze'
        else:
            return 'unknown'
    
    def _detect_regions_reflacx(self, bbox_file_path: Path) -> List[bool]:
        """
        Detect available regions from REFLACX bbox file.
        
        Args:
            bbox_file_path: Path to REFLACX bbox CSV file
            
        Returns:
            List of 17 boolean values indicating region availability
        """
        try:
            bbox_df = pd.read_csv(bbox_file_path)
            
            # Check if this is actually EyeGaze format (has 'bbox_name' instead of 'class_name')
            if 'bbox_name' in bbox_df.columns and 'class_name' not in bbox_df.columns:
                logger.debug(f"File appears to be EyeGaze format, switching detection method: {bbox_file_path}")
                return self._detect_regions_eyegaze(bbox_file_path)
            
            # REFLACX format: columns include 'class_name'
            if 'class_name' not in bbox_df.columns:
                logger.warning(f"No 'class_name' column in REFLACX file: {bbox_file_path}")
                return [False] * self.num_regions
            
            # Get available region names
            available_regions = set(bbox_df['class_name'].unique())
            
            # Create boolean mask for 17 standard regions
            region_mask = []
            for region in self.ANATOMICAL_REGIONS:
                region_mask.append(region in available_regions)
            
            return region_mask
            
        except Exception as e:
            logger.error(f"Error processing REFLACX bbox file {bbox_file_path}: {e}")
            return [False] * self.num_regions
    
    def _detect_regions_eyegaze(self, bbox_file_path: Path) -> List[bool]:
        """
        Detect available regions from EyeGaze bbox file.
        
        Args:
            bbox_file_path: Path to EyeGaze bbox CSV file
            
        Returns:
            List of 17 boolean values indicating region availability
        """
        try:
            bbox_df = pd.read_csv(bbox_file_path)
            
            # EyeGaze format: columns include 'bbox_name'
            if 'bbox_name' not in bbox_df.columns:
                logger.warning(f"No 'bbox_name' column in EyeGaze file: {bbox_file_path}")
                return [False] * self.num_regions
            
            # Get available region names
            available_regions = set(bbox_df['bbox_name'].unique())
            
            # Create boolean mask for 17 standard regions
            region_mask = []
            for region in self.ANATOMICAL_REGIONS:
                region_mask.append(region in available_regions)
            
            return region_mask
            
        except Exception as e:
            logger.error(f"Error processing EyeGaze bbox file {bbox_file_path}: {e}")
            return [False] * self.num_regions

    def get_region_mask(self, bbox_path: str) -> np.ndarray:
        """
        Get region availability mask for a single sample.
        
        Args:
            bbox_path: Path to bbox file
            
        Returns:
            Boolean numpy array of shape [17] indicating region availability
        """
        # Check cache first
        if bbox_path in self._region_mask_cache:
            return self._region_mask_cache[bbox_path]
        
        # Detect regions from actual bbox file
        region_mask = self.detect_available_regions_from_bbox(bbox_path)
        region_array = np.array(region_mask, dtype=bool)
        
        # Cache the result
        self._region_mask_cache[bbox_path] = region_array
        
        # Update statistics
        self.region_stats['total_samples_processed'] += 1
        if region_array.any():
            self.region_stats['samples_with_bbox'] += 1
            for idx, available in enumerate(region_array):
                if available:
                    region_name = self.ANATOMICAL_REGIONS[idx]
                    self.region_stats['region_availability_counts'][region_name] += 1
        else:
            self.region_stats['samples_without_bbox'] += 1
        
        return region_array

    def process_multiple_samples(self, bbox_paths: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Process multiple samples and return region masks.
        
        Args:
            bbox_paths: Dictionary mapping sample IDs to bbox file paths
            
        Returns:
            Dictionary mapping sample IDs to region masks
        """
        logger.info(f"Processing {len(bbox_paths)} samples for region detection...")
        
        region_masks = {}
        dataset_stats = defaultdict(lambda: {'total': 0, 'reflacx': 0, 'eyegaze': 0, 'unknown': 0})
        
        for sample_id, bbox_path in bbox_paths.items():
            # Get region mask
            region_mask = self.get_region_mask(bbox_path)
            region_masks[sample_id] = region_mask
            
            # Track dataset type statistics
            dataset_type = self._determine_dataset_type(bbox_path)
            dataset_stats['overall']['total'] += 1
            dataset_stats['overall'][dataset_type] += 1
            
            # Progress logging
            if len(region_masks) % 100 == 0:
                logger.info(f"Processed {len(region_masks)}/{len(bbox_paths)} samples...")
        
        # Update statistics
        self._calculate_region_coverage_by_dataset(bbox_paths, region_masks)
        self.region_stats['dataset_type_stats'] = dict(dataset_stats)
        
        logger.info(f"Region processing complete: {len(region_masks)} samples processed")
        logger.info(f"Dataset distribution: REFLACX={dataset_stats['overall']['reflacx']}, "
                   f"EyeGaze={dataset_stats['overall']['eyegaze']}, "
                   f"Unknown={dataset_stats['overall']['unknown']}")
        
        return region_masks
    
    def _calculate_region_coverage_by_dataset(self, bbox_paths: Dict[str, str], region_masks: Dict[str, np.ndarray]):
        """Calculate region coverage statistics by dataset type."""
        reflacx_coverage = np.zeros(self.num_regions, dtype=int)
        eyegaze_coverage = np.zeros(self.num_regions, dtype=int)
        reflacx_total = 0
        eyegaze_total = 0
        
        for sample_id, bbox_path in bbox_paths.items():
            dataset_type = self._determine_dataset_type(bbox_path)
            region_mask = region_masks[sample_id]
            
            if dataset_type == 'reflacx':
                reflacx_coverage += region_mask.astype(int)
                reflacx_total += 1
            elif dataset_type == 'eyegaze':
                eyegaze_coverage += region_mask.astype(int)
                eyegaze_total += 1
        
        # Calculate percentages
        self.region_stats['region_coverage_by_dataset'] = {
            'reflacx': {
                'total_samples': reflacx_total,
                'region_coverage': (reflacx_coverage / max(reflacx_total, 1) * 100).tolist(),
                'region_names': self.ANATOMICAL_REGIONS
            },
            'eyegaze': {
                'total_samples': eyegaze_total,
                'region_coverage': (eyegaze_coverage / max(eyegaze_total, 1) * 100).tolist(),
                'region_names': self.ANATOMICAL_REGIONS
            }
        }

    def get_region_statistics(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Get comprehensive region statistics.
        
        Returns:
            Dictionary with region statistics
        """
        stats = self.region_stats.copy()
        
        # Add derived statistics
        if stats['total_samples_processed'] > 0:
            stats['bbox_availability_rate'] = stats['samples_with_bbox'] / stats['total_samples_processed']
            
            # Calculate average regions per sample
            total_available_regions = sum(stats['region_availability_counts'].values())
            stats['avg_regions_per_sample'] = total_available_regions / stats['total_samples_processed']
            
            # Calculate region availability rates
            region_availability_rates = {}
            for region_name, count in stats['region_availability_counts'].items():
                region_availability_rates[region_name] = count / stats['total_samples_processed']
            stats['region_availability_rates'] = region_availability_rates
        
        return stats
    
    def create_region_tensor_batch(self, region_masks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create a batch tensor from region masks.
        
        Args:
            region_masks: Dictionary mapping sample IDs to region masks
            
        Returns:
            NumPy array of shape [batch_size, 17] with boolean values
        """
        if not region_masks:
            return np.empty((0, self.num_regions), dtype=bool)
        
        # Stack all region masks
        mask_list = list(region_masks.values())
        batch_tensor = np.stack(mask_list, axis=0)
        
        return batch_tensor
    
    def save_region_data(self, region_masks: Dict[str, np.ndarray], output_path: str):
        """
        Save region masks and statistics to JSON file.
        
        Args:
            region_masks: Dictionary mapping sample IDs to region masks
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        serializable_masks = {
            sample_id: mask.tolist() for sample_id, mask in region_masks.items()
        }
        
        output_data = {
            'region_masks': serializable_masks,
            'region_definitions': {
                'regions': self.ANATOMICAL_REGIONS,
                'num_regions': self.num_regions,
                'region_to_idx': self.region_to_idx
            },
            'statistics': self.get_region_statistics(),
            'metadata': {
                'handler_type': 'dummy_region_detection',
                'total_samples': len(region_masks),
                'regions_per_sample': self.num_regions
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Region data saved to {output_path}")
    
    def load_region_data(self, input_path: str) -> Dict[str, np.ndarray]:
        """
        Load region masks from JSON file.
        
        Args:
            input_path: Path to JSON file
            
        Returns:
            Dictionary mapping sample IDs to region masks
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert back to numpy arrays
        region_masks = {
            sample_id: np.array(mask, dtype=bool)
            for sample_id, mask in data['region_masks'].items()
        }
        
        # Restore statistics
        self.region_stats = data['statistics']
        
        logger.info(f"Loaded region data for {len(region_masks)} samples from {input_path}")
        
        return region_masks
    
    def get_available_regions_for_sample(self, sample_id: str, region_masks: Dict[str, np.ndarray]) -> List[str]:
        """
        Get list of available region names for a sample.
        
        Args:
            sample_id: Sample identifier
            region_masks: Dictionary mapping sample IDs to region masks
            
        Returns:
            List of available region names
        """
        if sample_id not in region_masks:
            return []
        
        region_mask = region_masks[sample_id]
        available_regions = []
        
        for region_idx, is_available in enumerate(region_mask):
            if is_available:
                available_regions.append(self.ANATOMICAL_REGIONS[region_idx])
        
        return available_regions
    
    def create_region_imputation_strategy(self, region_masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Create region imputation strategy based on availability statistics.
        
        Args:
            region_masks: Dictionary mapping sample IDs to region masks
            
        Returns:
            Dictionary mapping region names to imputation probabilities
        """
        region_availability_counts = defaultdict(int)
        total_samples = len(region_masks)
        
        # Count region availability
        for region_mask in region_masks.values():
            for region_idx, is_available in enumerate(region_mask):
                if is_available:
                    region_name = self.ANATOMICAL_REGIONS[region_idx]
                    region_availability_counts[region_name] += 1
        
        # Calculate imputation probabilities
        imputation_strategy = {}
        for region_name in self.ANATOMICAL_REGIONS:
            availability_rate = region_availability_counts[region_name] / total_samples if total_samples > 0 else 0
            imputation_strategy[region_name] = availability_rate
        
        return imputation_strategy


# Convenience functions
def process_regions_from_csv(csv_path: str, 
                           output_path: str = "output/regions/region_masks.json",
                           base_path: str = "../../") -> str:
    """
    Convenience function to process regions from CSV file.
    
    Args:
        csv_path: Path to CSV file containing bbox paths
        output_path: Path to save region data JSON file
        base_path: Base path to dataset files
        
    Returns:
        Path to saved region data file
    """
    handler = RegionHandler(base_path=base_path)
    bbox_paths = handler.load_bbox_paths_from_csv(csv_path)
    region_masks = handler.process_multiple_samples(bbox_paths)
    handler.save_region_data(region_masks, output_path)
    return output_path


def create_dummy_region_masks(num_samples: int = 10,
                            output_path: str = "output/regions/dummy_region_masks.json") -> str:
    """
    Convenience function to create dummy region masks for testing.
    
    Args:
        num_samples: Number of dummy samples to create
        output_path: Path to save region data JSON file
        
    Returns:
        Path to saved region data file
    """
    handler = RegionHandler()
    
    # Create dummy bbox paths
    dummy_bbox_paths = {}
    for i in range(num_samples):
        if i % 3 == 0:
            dataset_type = 'eyegaze'
        elif i % 3 == 1:
            dataset_type = 'reflacx'
        else:
            dataset_type = 'unknown'
        
        dummy_bbox_paths[f"sample_{i}"] = f"dummy_path/{dataset_type}/sample_{i}/bbox.json"
    
    # Process region masks
    region_masks = handler.process_multiple_samples(dummy_bbox_paths)
    handler.save_region_data(region_masks, output_path)
    
    return output_path


if __name__ == "__main__":
    # Example usage and testing
    print("Testing region handling...")
    
    # Test with train CSV
    train_csv = "../../dataset_splits/train.csv"
    if Path(train_csv).exists():
        handler = RegionHandler()
        bbox_paths = handler.load_bbox_paths_from_csv(train_csv)
        
        if bbox_paths:
            # Process first 10 samples
            sample_bbox_paths = dict(list(bbox_paths.items())[:10])
            region_masks = handler.process_multiple_samples(sample_bbox_paths)
            
            # Save region data
            output_path = "output/regions/region_masks.json"
            handler.save_region_data(region_masks, output_path)
            
            # Show statistics
            stats = handler.get_region_statistics()
            print(f"\nRegion Statistics:")
            print(f"  Total samples: {stats['total_samples_processed']}")
            print(f"  Samples with bbox: {stats['samples_with_bbox']}")
            print(f"  Average regions per sample: {stats.get('avg_regions_per_sample', 0):.2f}")
            print(f"  Dataset types: {stats['dataset_type_stats']}")
            
            # Show some sample region masks
            print(f"\nSample region masks:")
            for i, (sample_id, region_mask) in enumerate(list(region_masks.items())[:3]):
                available_regions = handler.get_available_regions_for_sample(sample_id, region_masks)
                print(f"  {sample_id}: {len(available_regions)}/17 regions available")
                print(f"    Available: {available_regions[:5]}...")  # Show first 5
            
            print(f"\n✅ Region handling test complete!")
        else:
            print("No bbox paths found in CSV file")
    else:
        print(f"Train CSV not found: {train_csv}")
        print("Creating dummy region masks for testing...")
        
        # Create dummy data
        dummy_output = create_dummy_region_masks(10)
        print(f"Dummy region masks created: {dummy_output}") 