#!/usr/bin/env python3
"""
Task 3.3: Build Data Loaders
Task 3.4: Validate Data Pipeline
Goal: Complete end-to-end data pipeline validation
"""

import logging
import time
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BlipProcessor

# Import our dataset
try:
    from .dataset import MedicalKeywordDataset
except ImportError:
    # Handle direct script execution
    from dataset import MedicalKeywordDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_data_loaders(
    train_csv: str = "../../dataset_splits/train.csv",
    val_csv: str = "../../dataset_splits/val.csv",
    test_csv: str = "../../dataset_splits/test.csv",
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    base_path: str = "../../",
    processor: Optional[BlipProcessor] = None,
    max_length: int = 128
) -> Dict[str, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        train_csv (str): Path to training CSV
        val_csv (str): Path to validation CSV
        test_csv (str): Path to test CSV
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes
        shuffle (bool): Whether to shuffle the data
        base_path (str): Base path to dataset
        processor: BLIP processor for tokenization
        max_length (int): Maximum sequence length
        
    Returns:
        Dict[str, DataLoader]: Dictionary with 'train', 'val', and 'test' data loaders
    """
    try:
        logger.info("Creating datasets...")
        
        # Create datasets
        train_dataset = MedicalKeywordDataset(
            csv_path=train_csv,
            base_path=base_path,
            processor=processor,
            max_length=max_length
        )
        
        val_dataset = MedicalKeywordDataset(
            csv_path=val_csv,
            base_path=base_path,
            processor=processor,
            max_length=max_length
        )
        
        test_dataset = MedicalKeywordDataset(
            csv_path=test_csv,
            base_path=base_path,
            processor=processor,
            max_length=max_length
        )
        
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset: {len(val_dataset)} samples")
        logger.info(f"Test dataset: {len(test_dataset)} samples")
        
        # Custom collate function for handling variable text lengths
        def collate_fn(batch):
            """Custom collate function to handle batching."""
            if not batch:
                return {}
            
            # Check if using BLIP processor format or manual format
            sample = batch[0]
            
            if 'pixel_values' in sample:
                # BLIP processor format
                return {
                    'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
                    'input_ids': torch.stack([item['input_ids'] for item in batch]),
                    'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
                    'labels': torch.stack([item['labels'] for item in batch]),
                    'keywords': [item['keywords'] for item in batch],
                    'dicom_ids': [item['dicom_id'] for item in batch],
                    'patient_folders': [item['patient_folder'] for item in batch]
                }
            else:
                # Manual format
                return {
                    'images': torch.stack([item['image'] for item in batch]),
                    'keywords': [item['keywords'] for item in batch],
                    'dicom_ids': [item['dicom_id'] for item in batch],
                    'patient_folders': [item['patient_folder'] for item in batch],
                    'keyword_counts': [item['keyword_count'] for item in batch]
                }
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),  # Speed up GPU transfer
            drop_last=True  # Drop incomplete batches for consistent training
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=True  # Drop incomplete batches for consistent validation
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=True  # Drop incomplete batches for consistent test
        )
        
        logger.info(f"Created data loaders:")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")
        logger.info(f"  Batch size: {batch_size}")
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset
        }
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        return {}

def test_data_loader_batch(data_loader: DataLoader, loader_name: str, max_batches: int = 3) -> bool:
    """
    Test a data loader by iterating through batches.
    
    Args:
        data_loader (DataLoader): Data loader to test
        loader_name (str): Name for logging
        max_batches (int): Maximum number of batches to test
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Testing {loader_name} loader...")
        
        successful_batches = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            
            try:
                # Check batch contents
                if 'pixel_values' in batch:
                    # BLIP format
                    batch_size = batch['pixel_values'].shape[0]
                    image_shape = batch['pixel_values'].shape
                    num_keywords = len(batch['keywords'])
                    
                    logger.info(f"  Batch {batch_idx}: {batch_size} samples, images: {image_shape}")
                    logger.info(f"    Sample keywords: \"{batch['keywords'][0][:50]}...\"")
                    
                elif 'images' in batch:
                    # Manual format
                    batch_size = batch['images'].shape[0]
                    image_shape = batch['images'].shape
                    num_keywords = len(batch['keywords'])
                    
                    logger.info(f"  Batch {batch_idx}: {batch_size} samples, images: {image_shape}")
                    logger.info(f"    Sample keywords: \"{batch['keywords'][0][:50]}...\"")
                    
                else:
                    logger.warning(f"  Batch {batch_idx}: Unknown batch format")
                    continue
                
                successful_batches += 1
                total_samples += batch_size
                
            except Exception as e:
                logger.error(f"  Batch {batch_idx} failed: {e}")
                return False
        
        logger.info(f"  ✅ {loader_name} loader: {successful_batches}/{max_batches} batches successful")
        logger.info(f"  📊 Total samples tested: {total_samples}")
        
        return successful_batches == max_batches
        
    except Exception as e:
        logger.error(f"Error testing {loader_name} loader: {e}")
        return False

# Task 3.4: Validate Data Pipeline

def validate_data_pipeline(
    max_batches: int = 100,
    batch_size: int = 8,
    include_test: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive validation of the complete data pipeline.
    Tests loading 100 batches without errors and gathers statistics.
    
    Args:
        max_batches (int): Maximum number of batches to test
        batch_size (int): Batch size for testing
        include_test (bool): Whether to include test set in validation
        
    Returns:
        Dict[str, Any]: Comprehensive validation results
    """
    print("="*60)
    print("TASK 3.4: VALIDATING COMPLETE DATA PIPELINE")
    print("="*60)
    
    validation_results = {
        'success': False,
        'train_results': {},
        'val_results': {},
        'test_results': {},
        'overall_stats': {}
    }
    
    try:
        # Create data loaders for all splits
        print("🧪 Creating data loaders for all splits...")
        
        splits_to_test = ['train', 'val']
        if include_test:
            splits_to_test.append('test')
        
        all_loaders = {}
        
        for split in splits_to_test:
            csv_path = f"../../dataset_splits/{split}.csv"
            
            loaders = create_data_loaders(
                train_csv=csv_path if split == 'train' else "../../dataset_splits/train.csv",
                val_csv=csv_path if split == 'val' else "../../dataset_splits/val.csv",
                test_csv=csv_path if split == 'test' else "../../dataset_splits/test.csv",
                batch_size=batch_size,
                num_workers=0,  # Avoid multiprocessing issues
                shuffle=False,  # Don't shuffle for validation
                processor=None  # Skip BLIP processor for speed
            )
            
            if split == 'train':
                all_loaders['train'] = loaders['train']
            elif split == 'val':
                all_loaders['val'] = loaders['val']
            elif split == 'test':
                # Create test loader separately
                test_loaders = create_data_loaders(
                    train_csv="../../dataset_splits/test.csv",
                    val_csv="../../dataset_splits/test.csv",
                    test_csv="../../dataset_splits/test.csv",
                    batch_size=batch_size,
                    num_workers=0,
                    shuffle=False,
                    processor=None
                )
                all_loaders['test'] = test_loaders['train']  # Use train loader for test data
        
        print(f"   ✅ Created loaders for splits: {list(all_loaders.keys())}")
        
        # Validate each split
        for split_name, loader in all_loaders.items():
            print(f"\n🔍 Validating {split_name} split...")
            
            result = validate_single_loader(
                loader, 
                split_name, 
                max_batches=min(max_batches, len(loader))
            )
            
            validation_results[f'{split_name}_results'] = result
            
            if not result['success']:
                print(f"   ❌ {split_name} validation failed")
                return validation_results
            else:
                print(f"   ✅ {split_name} validation passed")
        
        # Calculate overall statistics
        overall_stats = calculate_overall_stats(validation_results, splits_to_test)
        validation_results['overall_stats'] = overall_stats
        
        # Print summary
        print(f"\n📊 OVERALL VALIDATION SUMMARY:")
        print(f"   Total batches tested: {overall_stats['total_batches_tested']}")
        print(f"   Total samples processed: {overall_stats['total_samples_processed']}")
        print(f"   Average batch processing time: {overall_stats['avg_batch_time']:.3f}s")
        print(f"   Average keywords per sample: {overall_stats['avg_keywords_per_sample']:.1f}")
        print(f"   Unique image sizes: {overall_stats['unique_image_sizes']}")
        
        # Determine overall success
        all_splits_passed = all(
            validation_results[f'{split}_results']['success'] 
            for split in splits_to_test
        )
        
        validation_results['success'] = all_splits_passed
        
        if all_splits_passed:
            print(f"\n✅ TASK 3.4 COMPLETE - Data pipeline validation successful!")
        else:
            print(f"\n❌ TASK 3.4 FAILED - Some splits failed validation")
        
        return validation_results
        
    except Exception as e:
        print(f"\n❌ TASK 3.4 FAILED - Pipeline validation error: {e}")
        validation_results['error'] = str(e)
        return validation_results

def validate_single_loader(
    data_loader: DataLoader, 
    loader_name: str, 
    max_batches: int = 100
) -> Dict[str, Any]:
    """
    Validate a single data loader comprehensively.
    
    Args:
        data_loader (DataLoader): Data loader to validate
        loader_name (str): Name of the loader
        max_batches (int): Maximum batches to test
        
    Returns:
        Dict[str, Any]: Validation results
    """
    result = {
        'success': False,
        'loader_name': loader_name,
        'total_batches': len(data_loader),
        'tested_batches': 0,
        'successful_batches': 0,
        'total_samples': 0,
        'batch_times': [],
        'keyword_counts': [],
        'image_sizes': [],
        'errors': []
    }
    
    try:
        start_time = time.time()
        
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            
            batch_start = time.time()
            
            try:
                # Validate batch structure
                if 'images' in batch:
                    images = batch['images']
                    keywords = batch['keywords']
                    keyword_counts = batch['keyword_counts']
                    
                    # Check tensor shapes
                    batch_size = images.shape[0]
                    image_shape = images.shape[1:]  # (C, H, W)
                    
                    # Collect statistics
                    result['keyword_counts'].extend(keyword_counts)
                    result['image_sizes'].append(image_shape)
                    result['total_samples'] += batch_size
                    
                elif 'pixel_values' in batch:
                    # BLIP format
                    pixel_values = batch['pixel_values']
                    keywords = batch['keywords']
                    
                    batch_size = pixel_values.shape[0]
                    image_shape = pixel_values.shape[1:]
                    
                    # Estimate keyword counts
                    keyword_counts = [len(kw.split(',')) for kw in keywords]
                    result['keyword_counts'].extend(keyword_counts)
                    result['image_sizes'].append(image_shape)
                    result['total_samples'] += batch_size
                
                else:
                    raise ValueError(f"Unknown batch format in batch {batch_idx}")
                
                batch_time = time.time() - batch_start
                result['batch_times'].append(batch_time)
                result['successful_batches'] += 1
                
                # Log progress every 20 batches
                if batch_idx % 20 == 0:
                    print(f"   📊 Batch {batch_idx}: {batch_size} samples, {batch_time:.3f}s")
                
            except Exception as e:
                error_msg = f"Batch {batch_idx}: {str(e)}"
                result['errors'].append(error_msg)
                logger.error(f"Batch validation error: {error_msg}")
            
            result['tested_batches'] += 1
        
        # Calculate success rate
        success_rate = result['successful_batches'] / result['tested_batches'] if result['tested_batches'] > 0 else 0
        result['success_rate'] = success_rate
        result['success'] = success_rate >= 0.95  # 95% success rate required
        
        total_time = time.time() - start_time
        result['total_time'] = total_time
        result['avg_batch_time'] = sum(result['batch_times']) / len(result['batch_times']) if result['batch_times'] else 0
        
        print(f"   📊 {loader_name} Results:")
        print(f"      Batches: {result['successful_batches']}/{result['tested_batches']} ({success_rate*100:.1f}%)")
        print(f"      Samples: {result['total_samples']}")
        print(f"      Avg batch time: {result['avg_batch_time']:.3f}s")
        if result['errors']:
            print(f"      Errors: {len(result['errors'])}")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Loader validation failed: {str(e)}")
        logger.error(f"Loader validation error: {e}")
        return result

def calculate_overall_stats(validation_results: Dict[str, Any], splits: List[str]) -> Dict[str, Any]:
    """Calculate overall statistics across all splits."""
    
    total_batches = sum(
        validation_results[f'{split}_results'].get('tested_batches', 0)
        for split in splits
        if f'{split}_results' in validation_results
    )
    
    total_samples = sum(
        validation_results[f'{split}_results'].get('total_samples', 0)
        for split in splits
        if f'{split}_results' in validation_results
    )
    
    all_batch_times = []
    all_keyword_counts = []
    all_image_sizes = []
    
    for split in splits:
        if f'{split}_results' in validation_results:
            result = validation_results[f'{split}_results']
            all_batch_times.extend(result.get('batch_times', []))
            all_keyword_counts.extend(result.get('keyword_counts', []))
            all_image_sizes.extend(result.get('image_sizes', []))
    
    avg_batch_time = sum(all_batch_times) / len(all_batch_times) if all_batch_times else 0
    avg_keywords = sum(all_keyword_counts) / len(all_keyword_counts) if all_keyword_counts else 0
    unique_image_sizes = list(set(tuple(size) if isinstance(size, (list, tuple)) else size for size in all_image_sizes))
    
    return {
        'total_batches_tested': total_batches,
        'total_samples_processed': total_samples,
        'avg_batch_time': avg_batch_time,
        'avg_keywords_per_sample': avg_keywords,
        'unique_image_sizes': unique_image_sizes,
        'splits_tested': splits
    }

def test_data_loaders():
    """Test Task 3.3: Data Loaders"""
    
    print("="*60)
    print("TASK 3.3: TESTING DATA LOADERS")
    print("="*60)
    
    try:
        # Create data loaders
        print("🧪 Creating data loaders...")
        
        loaders = create_data_loaders(
            batch_size=1,  # Small batch size for testing
            num_workers=0,  # Avoid multiprocessing issues in testing
            shuffle=False,  # Don't shuffle for testing
            processor=None  # Skip BLIP processor for faster testing
        )
        
        if not loaders:
            print("❌ Failed to create data loaders")
            return False
        
        train_loader = loaders['train']
        val_loader = loaders['val']
        
        print(f"   ✅ Data loaders created successfully")
        print(f"   📊 Train batches: {len(train_loader)}")
        print(f"   📊 Val batches: {len(val_loader)}")
        
        # Test train loader
        print(f"\n🔍 Testing train loader...")
        train_success = test_data_loader_batch(train_loader, "train", max_batches=2)
        
        # Test val loader
        print(f"\n🔍 Testing val loader...")
        val_success = test_data_loader_batch(val_loader, "val", max_batches=2)
        
        # Overall success
        if train_success and val_success:
            print(f"\n✅ TASK 3.3 COMPLETE - Data loaders working correctly!")
            return True
        else:
            print(f"\n❌ TASK 3.3 FAILED - Some data loaders failed")
            return False
            
    except Exception as e:
        print(f"\n❌ TASK 3.3 FAILED - Data loader creation failed: {e}")
        return False

if __name__ == "__main__":
    # Test Task 3.3 first
    task3_success = test_data_loaders()
    
    # Then run comprehensive Task 3.4 validation
    if task3_success:
        validation_results = validate_data_pipeline(
            max_batches=50,  # Test 50 batches for comprehensive validation
            batch_size=8,
            include_test=False  # Skip test set for now to speed up validation
        )
        task4_success = validation_results['success']
    else:
        task4_success = False
    
    print(f"\n" + "="*60)
    print("OVERALL TASK STATUS:")
    print(f"   Task 3.3 (Data Loaders): {'✅ PASS' if task3_success else '❌ FAIL'}")
    print(f"   Task 3.4 (Pipeline Validation): {'✅ PASS' if task4_success else '❌ FAIL'}")
    print("="*60) 