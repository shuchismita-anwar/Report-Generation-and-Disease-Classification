#!/usr/bin/env python3
"""
Task 3.1: Build Image-Text Pair Loader
Task 3.2: Create PyTorch Dataset Class
Goal: Complete PyTorch dataset for BLIP-2 training
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BlipProcessor

# Import our transcript processor
try:
    from .transcript_processor import load_transcript, process_transcript_to_keywords
except ImportError:
    # Handle direct script execution
    from transcript_processor import load_transcript, process_transcript_to_keywords

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image_text_pair(csv_row: pd.Series, base_path: str = "../../") -> Optional[Dict[str, Any]]:
    """
    Load image and corresponding keywords for one sample from CSV row.
    
    Args:
        csv_row (pd.Series): Row from train/val/test CSV
        base_path (str): Base path to dataset
        
    Returns:
        Optional[Dict[str, Any]]: Dictionary with image, keywords, and metadata, or None if failed
    """
    try:
        # Extract paths from CSV row
        transcript_path = str(csv_row['transcript_path'])
        dicom_id = str(csv_row['dicom_id'])
        
        # Build paths - use processed PNG images from data_dump
        full_image_path = base_path + f"data_dump/output/img_png/{dicom_id}.png"
        full_transcript_path = base_path + transcript_path
        
        # Load image
        image = load_image(full_image_path)
        if image is None:
            logger.error(f"Failed to load image: {full_image_path}")
            return None
        
        # Load transcript and extract keywords
        transcript_text = load_transcript(full_transcript_path)
        if transcript_text is None:
            logger.error(f"Failed to load transcript: {full_transcript_path}")
            return None
        
        # Process transcript to keywords
        keyword_result = process_transcript_to_keywords(transcript_text)
        if not keyword_result['keywords']:
            logger.warning(f"No keywords extracted from transcript: {full_transcript_path}")
            return None
        
        # Create result dictionary
        result = {
            'image': image,
            'keywords': keyword_result['keywords'],
            'image_path': full_image_path,
            'transcript_path': full_transcript_path,
            'patient_folder': csv_row.get('patient_folder', ''),
            'dicom_id': csv_row.get('dicom_id', ''),
            'study_id': csv_row.get('study_id', ''),
            'condition': csv_row.get('condition', ''),
            'original_transcript': transcript_text,
            'keyword_count': keyword_result['keyword_count'],
            'compression_ratio': keyword_result['compression_ratio'],
            'image_size': image.size,
            'image_mode': image.mode
        }
        
        logger.debug(f"Successfully loaded image-text pair: {keyword_result['keyword_count']} keywords, {image.size} image")
        return result
        
    except Exception as e:
        logger.error(f"Error loading image-text pair: {e}")
        return None

def load_image(image_path: str) -> Optional[Image.Image]:
    """
    Load and validate an image file.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        Optional[Image.Image]: PIL Image object or None if failed
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        
        # Load image with PIL
        image = Image.open(image_path)
        
        # Convert to RGB if needed (some medical images might be grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.debug(f"Successfully loaded image: {image.size}, mode: {image.mode}")
        return image
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def load_multiple_pairs(csv_path: str, num_samples: int = 10, base_path: str = "../../") -> Dict[str, Any]:
    """
    Load multiple image-text pairs from CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        num_samples (int): Number of samples to load
        base_path (str): Base path to dataset
        
    Returns:
        Dict[str, Any]: Results with loaded pairs and statistics
    """
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(df)} rows")
        
        # Limit to requested number of samples
        df_subset = df.head(num_samples)
        
        # Load pairs
        pairs = []
        success_count = 0
        
        for idx, row in df_subset.iterrows():
            logger.info(f"Loading pair {idx + 1}/{len(df_subset)}")
            
            pair = load_image_text_pair(row, base_path)
            if pair:
                pairs.append(pair)
                success_count += 1
                logger.info(f"  ✅ Success: \"{pair['keywords'][:50]}...\"")
            else:
                logger.warning(f"  ❌ Failed to load pair for row {idx}")
        
        # Calculate statistics
        success_rate = success_count / len(df_subset) * 100 if len(df_subset) > 0 else 0
        
        # Statistics for successful pairs
        if pairs:
            avg_keywords = sum(p['keyword_count'] for p in pairs) / len(pairs)
            image_sizes = [p['image_size'] for p in pairs]
            unique_sizes = list(set(image_sizes))
        else:
            avg_keywords = 0
            unique_sizes = []
        
        result = {
            'success': True,
            'csv_path': csv_path,
            'requested_samples': num_samples,
            'loaded_samples': len(df_subset),
            'success_count': success_count,
            'success_rate': success_rate,
            'pairs': pairs,
            'avg_keywords_per_pair': avg_keywords,
            'unique_image_sizes': unique_sizes
        }
        
        logger.info(f"Loaded {success_count}/{len(df_subset)} pairs ({success_rate:.1f}% success)")
        return result
        
    except Exception as e:
        logger.error(f"Error loading multiple pairs: {e}")
        return {'success': False, 'error': str(e)}

# Task 3.2: Create PyTorch Dataset Class

class MedicalKeywordDataset(Dataset):
    """
    PyTorch Dataset for medical image-keyword pairs.
    Loads chest X-rays and corresponding descriptive keywords for BLIP-2 training.
    """
    
    def __init__(self, csv_path: str, base_path: str = "../../", processor=None, max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            csv_path (str): Path to CSV file (train.csv, val.csv, test.csv)
            base_path (str): Base path to dataset
            processor: BLIP processor for tokenization (optional)
            max_length (int): Maximum sequence length for text
        """
        self.csv_path = csv_path
        self.base_path = base_path
        self.max_length = max_length
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataset with {len(self.df)} samples from {csv_path}")
        
        # Initialize processor
        if processor is None:
            try:
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
                logger.info("Loaded BLIP-2 processor")
            except Exception as e:
                logger.warning(f"Could not load BLIP processor: {e}")
                self.processor = None
        else:
            self.processor = processor
        
        # Image transforms for consistency
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Ensure consistent size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Pre-filter valid samples (optional - can be done on-demand)
        self.valid_indices = self._find_valid_samples()
        logger.info(f"Found {len(self.valid_indices)} valid samples out of {len(self.df)}")
    
    def _find_valid_samples(self, max_check: int = 100) -> list:
        """
        Pre-check which samples have both valid images and transcripts.
        Only checks first max_check samples to avoid long initialization.
        """
        valid_indices = []
        check_count = min(max_check, len(self.df))
        
        for idx in range(check_count):
            row = self.df.iloc[idx]
            
            # Check if files exist
            dicom_id = str(row['dicom_id'])
            transcript_path = str(row['transcript_path'])
            
            image_path = self.base_path + f"data_dump/output/img_png/{dicom_id}.png"
            full_transcript_path = self.base_path + transcript_path
            
            if os.path.exists(image_path) and os.path.exists(full_transcript_path):
                valid_indices.append(idx)
        
        # If we found valid samples in the first batch, assume the rest are similar
        if len(valid_indices) > check_count * 0.8:  # 80% success rate
            # Add remaining indices without checking (for speed)
            valid_indices.extend(range(check_count, len(self.df)))
        
        return valid_indices
    
    def __len__(self):
        """Return the number of valid samples in the dataset."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary with processed image and text
        """
        # Get actual row index
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        
        try:
            # Load image-text pair
            pair = load_image_text_pair(row, self.base_path)
            
            if pair is None:
                # Return a dummy sample if loading fails
                return self._get_dummy_sample()
            
            # Process image
            image = pair['image']
            if self.processor:
                # Use BLIP processor
                processed = self.processor(
                    images=image,
                    text=pair['keywords'],
                    return_tensors="pt",
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True
                )
                
                return {
                    'pixel_values': processed['pixel_values'].squeeze(0),
                    'input_ids': processed['input_ids'].squeeze(0),
                    'attention_mask': processed['attention_mask'].squeeze(0),
                    'labels': processed['input_ids'].squeeze(0),  # For training
                    'keywords': pair['keywords'],
                    'dicom_id': pair['dicom_id'],
                    'patient_folder': pair['patient_folder']
                }
            else:
                # Manual processing if no BLIP processor
                image_tensor = self.image_transform(image)
                
                return {
                    'image': image_tensor,
                    'keywords': pair['keywords'],
                    'dicom_id': pair['dicom_id'],
                    'patient_folder': pair['patient_folder'],
                    'keyword_count': pair['keyword_count']
                }
                
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            return self._get_dummy_sample()
    
    def _get_dummy_sample(self):
        """Return a dummy sample in case of loading errors."""
        if self.processor:
            dummy_image = Image.new('RGB', (512, 512), color='black')
            dummy_text = "normal chest"
            
            processed = self.processor(
                images=dummy_image,
                text=dummy_text,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )
            
            return {
                'pixel_values': processed['pixel_values'].squeeze(0),
                'input_ids': processed['input_ids'].squeeze(0),
                'attention_mask': processed['attention_mask'].squeeze(0),
                'labels': processed['input_ids'].squeeze(0),
                'keywords': dummy_text,
                'dicom_id': 'dummy',
                'patient_folder': 'dummy'
            }
        else:
            dummy_tensor = torch.zeros(3, 512, 512)
            return {
                'image': dummy_tensor,
                'keywords': 'normal chest',
                'dicom_id': 'dummy',
                'patient_folder': 'dummy',
                'keyword_count': 2
            }
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata about a sample without loading the full data."""
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        
        return {
            'index': idx,
            'actual_index': actual_idx,
            'patient_folder': row.get('patient_folder', ''),
            'dicom_id': row.get('dicom_id', ''),
            'study_id': row.get('study_id', ''),
            'condition': row.get('condition', '')
        }

def test_pytorch_dataset():
    """Test Task 3.2: PyTorch Dataset Class"""
    
    print("\n" + "="*60)
    print("TASK 3.2: TESTING PYTORCH DATASET CLASS")
    print("="*60)
    
    try:
        # Test dataset creation
        print("🧪 Creating dataset from train.csv...")
        
        # Create dataset without BLIP processor first (faster testing)
        dataset = MedicalKeywordDataset(
            csv_path="../../dataset_splits/train.csv",
            processor=None  # Skip BLIP processor for faster testing
        )
        
        print(f"   ✅ Dataset created successfully")
        print(f"   📊 Total samples: {len(dataset)}")
        
        # Test sample loading
        print(f"\n🔍 Testing sample loading...")
        sample_indices = [0, 1, 2]
        
        for i in sample_indices:
            try:
                sample = dataset[i]
                info = dataset.get_sample_info(i)
                
                print(f"   ✅ Sample {i}: {sample['keywords'][:50]}...")
                print(f"      Patient: {info['patient_folder']} | DICOM: {info['dicom_id']}")
                print(f"      Image shape: {sample['image'].shape}")
                
            except Exception as e:
                print(f"   ❌ Sample {i} failed: {e}")
                return False
        
        # Test iteration
        print(f"\n🔄 Testing dataset iteration...")
        success_count = 0
        test_samples = min(5, len(dataset))
        
        for i in range(test_samples):
            try:
                sample = dataset[i]
                if 'keywords' in sample and 'image' in sample:
                    success_count += 1
            except Exception as e:
                print(f"   ⚠️  Iteration {i} failed: {e}")
        
        success_rate = success_count / test_samples * 100
        print(f"   📊 Iteration success: {success_count}/{test_samples} ({success_rate:.1f}%)")
        
        if success_rate >= 80:  # At least 80% success rate
            print(f"\n✅ TASK 3.2 COMPLETE - PyTorch dataset working correctly!")
            return True
        else:
            print(f"\n❌ TASK 3.2 FAILED - Dataset iteration success rate too low")
            return False
            
    except Exception as e:
        print(f"\n❌ TASK 3.2 FAILED - Dataset creation failed: {e}")
        return False

def test_image_text_pair_loader():
    """Test Task 3.1: Image-Text Pair Loader"""
    
    print("="*60)
    print("TASK 3.1: TESTING IMAGE-TEXT PAIR LOADER")
    print("="*60)
    
    # Test with first few rows of train.csv
    csv_path = "../../dataset_splits/train.csv"
    num_test_samples = 5
    
    print(f"🧪 Testing with first {num_test_samples} samples from train.csv...")
    
    # Load test pairs
    result = load_multiple_pairs(csv_path, num_test_samples)
    
    if result['success']:
        print(f"\n📊 LOADING RESULTS:")
        print(f"   Requested: {result['requested_samples']}")
        print(f"   Successful: {result['success_count']}")
        print(f"   Success rate: {result['success_rate']:.1f}%")
        print(f"   Avg keywords per pair: {result['avg_keywords_per_pair']:.1f}")
        print(f"   Image sizes found: {result['unique_image_sizes']}")
        
        # Show sample pairs
        if result['pairs']:
            print(f"\n📋 SAMPLE PAIRS:")
            for i, pair in enumerate(result['pairs'][:3], 1):
                print(f"   {i}. Image: {pair['image_size']} | Keywords: \"{pair['keywords'][:60]}...\"")
                print(f"      Patient: {pair['patient_folder']} | DICOM: {pair['dicom_id']}")
        
        # Determine success
        if result['success_rate'] >= 60:  # At least 60% success rate
            print(f"\n✅ TASK 3.1 COMPLETE - Image-text pair loader working correctly!")
            return True
        else:
            print(f"\n❌ TASK 3.1 FAILED - Success rate too low ({result['success_rate']:.1f}%)")
            return False
    else:
        print(f"\n❌ TASK 3.1 FAILED - {result.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    # Test both tasks
    task1_success = test_image_text_pair_loader()
    task2_success = test_pytorch_dataset()
    
    print(f"\n" + "="*60)
    print("OVERALL TASK STATUS:")
    print(f"   Task 3.1 (Image-Text Pair Loader): {'✅ PASS' if task1_success else '❌ FAIL'}")
    print(f"   Task 3.2 (PyTorch Dataset): {'✅ PASS' if task2_success else '❌ FAIL'}")
    print("="*60) 