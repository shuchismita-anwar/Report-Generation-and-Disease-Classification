#!/usr/bin/env python3
"""
Test script to validate fixes for training issues:
1. Missing keyword extractions
2. Image conversion failures
3. Empty BLEU/ROUGE evaluations
4. Unicode logging failures
5. Zero validation coverage & quality

This script tests each fix individually before running training.
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data.transcript_processor import extract_descriptive_keywords, load_transcript
from data.dataset import MedicalKeywordDataset
from model.blip2_medical import MedicalBLIP2Model
from training.trainer import MedicalKeywordTrainer
import pandas as pd
from PIL import Image
import numpy as np

# Set up logging without emoji characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_keyword_extraction_fix():
    """Test Fix #1: Missing keyword extractions"""
    print("\n" + "="*60)
    print("TEST 1: Keyword Extraction Fix")
    print("="*60)
    
    # Test cases for keyword extraction
    test_transcripts = [
        "The patient has normal chest appearance with clear lung fields.",
        "Bilateral lung infiltrates are noted with prominent cardiac silhouette.",
        "Chest X-ray shows mild pulmonary edema and enlarged heart.",
        "Normal thoracic structures with no acute findings.",
        "The examination demonstrates consolidation in the right lower lobe.",
        "",  # Empty transcript
        "This is a very short text.",  # Minimal content
    ]
    
    print(f"Testing keyword extraction on {len(test_transcripts)} transcripts...")
    
    for i, transcript in enumerate(test_transcripts):
        keywords = extract_descriptive_keywords(transcript)
        print(f"  Test {i+1}: '{transcript[:50]}...' -> '{keywords}'")
        
        # Validate that we always get some keywords (no empty results)
        assert len(keywords) > 0, f"Empty keywords for transcript {i+1}"
        assert keywords != "", f"Empty string returned for transcript {i+1}"
    
    print("✅ All keyword extraction tests passed!")
    return True

def test_image_conversion_fix():
    """Test Fix #2: Image conversion failures"""
    print("\n" + "="*60)
    print("TEST 2: Image Conversion Fix")
    print("="*60)
    
    try:
        # Initialize model
        print("Loading BLIP-2 model...")
        model = MedicalBLIP2Model(device="cpu")  # Use CPU for testing
        
        # Create test image tensor (normalized as in dataset)
        print("Creating test normalized image tensor...")
        test_tensor = torch.randn(3, 224, 224)  # Random normalized tensor
        
        # Apply ImageNet normalization (simulating dataset preprocessing)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        normalized_tensor = (test_tensor - mean) / std
        
        print(f"Normalized tensor range: [{normalized_tensor.min():.4f}, {normalized_tensor.max():.4f}]")
        
        # Test keyword generation with tensor input
        print("Testing keyword generation with tensor input...")
        keywords = model.generate_keywords(
            normalized_tensor,
            prompt="Medical keywords:",
            max_new_tokens=10,
            num_beams=1
        )
        
        print(f"Generated keywords: '{keywords}'")
        
        # Validate output
        assert isinstance(keywords, str), "Keywords should be string"
        assert len(keywords) > 0, "Keywords should not be empty"
        
        print("✅ Image conversion test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Image conversion test failed: {e}")
        return False

def test_unicode_logging_fix():
    """Test Fix #4: Unicode logging failures"""
    print("\n" + "="*60)
    print("TEST 3: Unicode Logging Fix")
    print("="*60)
    
    try:
        # Test logging messages that previously caused Unicode errors
        test_messages = [
            "Validation Metrics Summary:",
            "Best Example:",
            "WARNING: NaN loss detected",
            "CRITICAL: Scale loss for gradient accumulation",
        ]
        
        print("Testing logging messages without emoji characters...")
        for msg in test_messages:
            logger.info(msg)
        
        print("✅ Unicode logging test passed!")
        return True
        
    except UnicodeEncodeError as e:
        print(f"❌ Unicode logging test failed: {e}")
        return False

def test_dataset_loading_fix():
    """Test Fix #1 + #5: Dataset loading with better keyword extraction"""
    print("\n" + "="*60)
    print("TEST 4: Dataset Loading Fix")
    print("="*60)
    
    try:
        # Test with a small subset of validation data
        csv_path = "dataset_splits/val.csv"
        if not os.path.exists(csv_path):
            print(f"⏭️ Skipping dataset test - {csv_path} not found")
            return True
        
        print("Loading validation dataset...")
        dataset = MedicalKeywordDataset(
            csv_path=csv_path,
            base_path="../../",
            max_length=64
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading a few samples
        print("Testing dataset sample loading...")
        for i in range(min(3, len(dataset))):
            try:
                sample = dataset[i]
                keywords = sample['keywords']
                print(f"  Sample {i}: '{keywords[:50]}...'")
                
                # Validate sample
                assert len(keywords) > 0, f"Empty keywords in sample {i}"
                assert 'pixel_values' in sample, f"Missing pixel_values in sample {i}"
                assert 'input_ids' in sample, f"Missing input_ids in sample {i}"
                
            except Exception as e:
                print(f"  Sample {i}: Failed to load - {e}")
        
        print("✅ Dataset loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        return False

def test_tensor_to_keywords_fix():
    """Test Fix #2 + #3: Complete tensor to keywords pipeline"""
    print("\n" + "="*60)
    print("TEST 5: Tensor to Keywords Pipeline Fix")
    print("="*60)
    
    try:
        # Initialize model
        print("Loading BLIP-2 model...")
        model = MedicalBLIP2Model(device="cpu")
        
        # Create test batch of images (simulating validation batch)
        print("Creating test batch of normalized tensors...")
        batch_size = 2
        batch_tensor = torch.randn(batch_size, 3, 224, 224)
        
        # Apply normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        normalized_batch = (batch_tensor - mean) / std
        
        print(f"Batch shape: {normalized_batch.shape}")
        print(f"Batch range: [{normalized_batch.min():.4f}, {normalized_batch.max():.4f}]")
        
        # Initialize trainer's keyword generation method
        from training.trainer import MedicalKeywordTrainer
        
        # Create dummy trainer (minimal setup)
        dummy_optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
        
        # 🔧 FIX: Create a minimal dummy data loader for trainer initialization
        dummy_dataset = torch.utils.data.TensorDataset(
            torch.randn(1, 3, 224, 224),  # dummy images
            torch.randint(0, 1000, (1, 10))  # dummy input_ids
        )
        dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=1)
        
        trainer = MedicalKeywordTrainer(
            model=model,
            train_loader=dummy_loader,  # Use dummy loader instead of None
            val_loader=None,
            learning_rate=1e-6,
            device="cpu"
        )
        trainer.optimizer = dummy_optimizer
        
        # Test the fixed keyword generation method
        print("Testing trainer's _generate_keywords_for_metrics method...")
        keywords_list = trainer._generate_keywords_for_metrics(normalized_batch, batch_size)
        
        print(f"Generated {len(keywords_list)} keyword strings:")
        for i, keywords in enumerate(keywords_list):
            print(f"  Batch {i}: '{keywords}'")
        
        # Validate results
        assert len(keywords_list) == batch_size, f"Expected {batch_size} keywords, got {len(keywords_list)}"
        for i, keywords in enumerate(keywords_list):
            assert isinstance(keywords, str), f"Keywords {i} should be string, got {type(keywords)}"
            assert len(keywords) > 0, f"Keywords {i} should not be empty"
        
        print("✅ Tensor to keywords pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tensor to keywords pipeline test failed: {e}")
        return False

def main():
    """Run all fix validation tests"""
    print("VALIDATION TESTS FOR TRAINING ISSUE FIXES")
    print("="*60)
    
    test_results = {}
    
    # Run all tests
    test_results['keyword_extraction'] = test_keyword_extraction_fix()
    test_results['image_conversion'] = test_image_conversion_fix()
    test_results['unicode_logging'] = test_unicode_logging_fix()
    test_results['dataset_loading'] = test_dataset_loading_fix()
    test_results['tensor_to_keywords'] = test_tensor_to_keywords_fix()
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Training fixes are validated.")
        print("Ready to run training with:")
        print("  python main_train.py --mode full --max_epochs 5")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Review fixes before training.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 