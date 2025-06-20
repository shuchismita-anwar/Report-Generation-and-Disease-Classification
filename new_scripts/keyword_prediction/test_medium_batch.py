import json
import os
from ai_cleanup import GeminiAICleanup

def test_medium_batch():
    """Test with medium batch (~100 keywords) to verify reliability"""
    
    # Create medium test data - more realistic test
    test_data = {
        "metadata": {
            "version": "2.1",
            "description": "Medium test for batch size verification"
        },
        "conditions": {
            "Atelectasis": {
                "high_confidence": [
                    "collapsed lung", "are seen", "atelectasis", "noted", "the", "bilateral",
                    "lung collapse", "again", "subsegmental", "linear atelectasis", "mild",
                    "streaky opacity", "present", "volume loss", "and", "left lower lobe",
                    "discoid atelectasis", "previously", "compressive", "plate atelectasis",
                    "associated", "lung base opacity", "findings", "basilar atelectasis",
                    "shows", "retrocardiac", "changes", "dependent atelectasis", "demonstrates"
                ],
                "medium_confidence": [
                    "band-like opacity", "bibasilar atelectasis", "subsegmental atelectasis",
                    "consolidation", "appears", "pneumonic infiltrate", "currently",
                    "air bronchogram", "of", "right upper lobe", "lobar pneumonia", "suggests",
                    "inflammatory process", "bronchopneumonia", "severe", "infectious process",
                    "evident", "patchy opacity", "with", "bacterial pneumonia", "recent"
                ],
                "low_confidence": [
                    "lung infection", "aspiration pneumonia", "compatible", "ground glass opacity",
                    "abnormality", "pneumonitis", "reveals", "multifocal", "pattern",
                    "community acquired", "indicates", "pleural effusion", "fluid collection",
                    "costophrenic angle", "blunting", "layering", "loculated", "empyema",
                    "thoracentesis", "drainage", "hydrothorax", "hemothorax", "chylothorax",
                    "transudative", "exudative", "parapneumonic", "malignant effusion"
                ]
            },
            "Pneumonia": {
                "high_confidence": [
                    "consolidation", "pneumonic infiltrate", "air bronchogram", "lobar pneumonia",
                    "inflammatory process", "bronchopneumonia", "infectious process", 
                    "patchy opacity", "bacterial pneumonia", "lung infection", "aspiration pneumonia",
                    "ground glass opacity", "pneumonitis", "multifocal", "community acquired"
                ],
                "medium_confidence": [
                    "appears", "currently", "of", "right upper lobe", "suggests", "severe",
                    "evident", "with", "recent", "compatible", "abnormality", "reveals",
                    "pattern", "indicates", "hospital acquired", "ventilator associated"
                ],
                "low_confidence": [
                    "opacity", "infiltrate", "process", "infection", "acute", "chronic",
                    "bilateral", "unilateral", "upper lobe", "lower lobe", "lingular",
                    "right middle lobe", "segmental", "subsegmental", "multilobar"
                ]
            }
        }
    }
    
    test_file = "medium_test.json"
    
    try:
        # Save test data
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        # Count total keywords
        total_keywords = 0
        for condition, condition_data in test_data['conditions'].items():
            for conf_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
                total_keywords += len(condition_data[conf_level])
        
        print(f"🧪 Testing with MEDIUM batch ({total_keywords} keywords)...")
        print("🎯 This will verify batch_size=50 works reliably")
        
        # Initialize cleanup
        cleanup = GeminiAICleanup()
        
        # Test with default batch size (50)
        print("🔄 Running with batch_size=50 (default)...")
        result = cleanup.main(input_file=test_file)
        
        if result:
            print("✅ MEDIUM batch test SUCCEEDED!")
            print("🚀 Ready to run on full dataset!")
            
            # Show some sample results
            print(f"\n📊 SAMPLE RESULTS:")
            for condition_name, condition_data in result['conditions'].items():
                kept_keywords = []
                for conf_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
                    if conf_level in condition_data:
                        kept_keywords.extend(condition_data[conf_level])
                
                if kept_keywords:
                    sample_kept = kept_keywords[:5]
                    print(f"  {condition_name}: {sample_kept}")
                    
        else:
            print("❌ MEDIUM batch test FAILED!")
            print("💡 Need to reduce batch size further")
            
        return result is not None
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists("failed_batches_recovery.json"):
            os.remove("failed_batches_recovery.json")

if __name__ == "__main__":
    success = test_medium_batch()
    if success:
        print("\n🎉 Medium batch worked perfectly!")
        print("🚀 Run the main script now - it should work reliably!")
    else:
        print("\n💥 Medium batch failed - need smaller batch size") 