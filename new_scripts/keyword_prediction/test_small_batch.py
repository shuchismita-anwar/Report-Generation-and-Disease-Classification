import json
import os
from ai_cleanup import GeminiAICleanup

def test_very_small_batch():
    """Test with extremely small batch to isolate the problem"""
    
    # Create minimal test data
    test_data = {
        "metadata": {
            "version": "2.1",
            "description": "Minimal test for debugging API issues"
        },
        "conditions": {
            "Atelectasis": {
                "high_confidence": [
                    "collapsed lung",      # Should KEEP
                    "are seen",           # Should REMOVE
                    "atelectasis",        # Should KEEP
                    "noted",              # Should REMOVE
                    "the"                 # Should REMOVE
                ],
                "medium_confidence": [],
                "low_confidence": []
            }
        }
    }
    
    test_file = "tiny_test.json"
    
    try:
        # Save test data
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        print("🧪 Testing with TINY batch (5 keywords only)...")
        print("🎯 This will help isolate if the issue is payload size")
        
        # Initialize cleanup
        cleanup = GeminiAICleanup()
        
        # Test with batch size of 5 (extremely small)
        print("🔄 Running with batch_size=5...")
        result = cleanup.main(input_file=test_file, batch_size=5)
        
        if result:
            print("✅ TINY batch test SUCCEEDED!")
            print("💡 Issue is likely payload size - need even smaller batches")
        else:
            print("❌ TINY batch test FAILED!")
            print("💡 Issue is deeper than payload size")
            
        return result is not None
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists("failed_batches_recovery.json"):
            os.remove("failed_batches_recovery.json")

if __name__ == "__main__":
    success = test_very_small_batch()
    if success:
        print("\n🎉 Small batch worked - try reducing batch size to 50-100")
    else:
        print("\n💥 Even small batch failed - API/prompt issue") 