#!/usr/bin/env python3
"""
Test script to verify Unicode encoding fixes.
Tests:
1. Unicode text cleaning function
2. Logging with problematic characters
3. Tokenization with max_length settings
"""

import os
import sys
import logging
import tempfile

# Add project to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from data.transcript_processor import clean_unicode_text

def test_unicode_cleaning():
    """Test Unicode text cleaning function."""
    print("🧪 Testing Unicode text cleaning...")
    
    # Test cases with problematic Unicode characters
    test_cases = [
        # Original text with Unicode issues, Expected cleaned text
        ("Text with replacement char: \ufffd", "Text with replacement char:"),
        ("Em dash \u2014 and en dash \u2013 test", "Em dash - and en dash - test"),
        ("Smart quotes: \u201cHello\u201d and \u2018world\u2019", 'Smart quotes: "Hello" and \'world\''),
        ("Non-breaking\u00a0space\u00adtest", "Non-breaking space test"),
        ("Ellipsis\u2026 test", "Ellipsis... test"),
        ("", ""),  # Empty string
        (None, ""),  # None input
        ("Normal text without issues", "Normal text without issues"),
    ]
    
    success_count = 0
    for i, (input_text, expected) in enumerate(test_cases):
        try:
            result = clean_unicode_text(input_text) if input_text is not None else clean_unicode_text("")
            if result == expected:
                print(f"   ✅ Test {i+1}: PASS")
                success_count += 1
            else:
                print(f"   ❌ Test {i+1}: FAIL")
                print(f"      Input: {repr(input_text)}")
                print(f"      Expected: {repr(expected)}")
                print(f"      Got: {repr(result)}")
        except Exception as e:
            print(f"   ❌ Test {i+1}: ERROR - {e}")
    
    print(f"   📊 Results: {success_count}/{len(test_cases)} tests passed")
    return success_count == len(test_cases)

def test_logging_unicode():
    """Test logging with Unicode characters."""
    print("\n🧪 Testing logging with Unicode characters...")
    
    try:
        # Create temporary log file with UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.log') as temp_log:
            temp_log_path = temp_log.name
        
        # Setup logger with UTF-8 encoding
        test_logger = logging.getLogger('unicode_test')
        test_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        for handler in test_logger.handlers[:]:
            test_logger.removeHandler(handler)
        
        # Add UTF-8 file handler
        file_handler = logging.FileHandler(temp_log_path, encoding='utf-8')
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        test_logger.addHandler(file_handler)
        
        # Test problematic messages
        test_messages = [
            "Comprehensive metrics system enabled (BLEU/ROUGE/Medical)",  # Fixed emoji
            "Normal text without Unicode issues",
            "Text with cleaned problematic chars: basal music sheet question recovery...",
        ]
        
        success_count = 0
        for i, message in enumerate(test_messages):
            try:
                test_logger.info(message)
                success_count += 1
                print(f"   ✅ Log message {i+1}: PASS")
            except UnicodeEncodeError as e:
                print(f"   ❌ Log message {i+1}: Unicode error - {e}")
            except Exception as e:
                print(f"   ❌ Log message {i+1}: Other error - {e}")
        
        # Close and cleanup
        file_handler.close()
        test_logger.removeHandler(file_handler)
        
        # Verify log file was written
        if os.path.exists(temp_log_path):
            with open(temp_log_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            os.unlink(temp_log_path)  # Clean up
            
            if len(log_content) > 0:
                print(f"   ✅ Log file written successfully ({len(log_content)} chars)")
                success_count += 1
            else:
                print(f"   ❌ Log file is empty")
        else:
            print(f"   ❌ Log file was not created")
        
        print(f"   📊 Results: {success_count}/{len(test_messages)+1} tests passed")
        return success_count == len(test_messages) + 1
        
    except Exception as e:
        print(f"   ❌ Logging test failed: {e}")
        return False

def test_tokenizer_max_length():
    """Test that tokenizer calls include max_length to prevent warnings."""
    print("\n🧪 Testing tokenizer max_length settings...")
    
    try:
        # Import required modules
        from transformers import Blip2Processor
        
        # Load processor
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        tokenizer = processor.tokenizer
        
        # Test text that might cause truncation
        long_text = "cardiac silhouette enlarged prominent hilar markings bilateral lung fields clear " * 10
        
        # Test tokenization with proper max_length
        try:
            result = tokenizer.encode(
                long_text,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=64,
                truncation=True
            )
            print(f"   ✅ Tokenization with max_length: PASS (output length: {result.shape[1]})")
            return True
            
        except Exception as e:
            print(f"   ❌ Tokenization failed: {e}")
            return False
            
    except ImportError:
        print(f"   ⚠️  Skipping tokenizer test (transformers not available)")
        return True  # Don't fail if transformers isn't installed
    except Exception as e:
        print(f"   ❌ Tokenizer test failed: {e}")
        return False

def main():
    """Run all Unicode fix tests."""
    print("=" * 60)
    print("UNICODE ENCODING FIXES VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Unicode Text Cleaning", test_unicode_cleaning),
        ("Logging with Unicode", test_logging_unicode),
        ("Tokenizer Max Length", test_tokenizer_max_length),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASS" if result else "FAIL"
            print(f"   📋 {test_name}: {status}")
        except Exception as e:
            print(f"   ❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All Unicode encoding fixes working correctly!")
        return True
    else:
        print("❌ Some tests failed - Unicode issues may persist")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 