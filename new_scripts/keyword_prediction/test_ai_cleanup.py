import json
import os
from ai_cleanup import GeminiAICleanup

def create_test_keywords_file():
    """Create a small test keywords file for testing AI cleanup"""
    
    test_data = {
        "metadata": {
            "version": "2.1",
            "description": "Test keywords for AI cleanup verification",
            "extraction_method": "test_data",
            "total_patients_analyzed": 10,
            "successful_extractions": 10,
            "last_updated": "2025-01-21T10:00:00"
        },
        "conditions": {
            "Atelectasis": {
                "high_confidence": [
                    "collapsed lung",           # Should KEEP - medical term
                    "are seen",                 # Should REMOVE - generic observation  
                    "atelectasis",             # Should KEEP - medical term
                    "noted",                   # Should REMOVE - generic observation
                    "band-like opacity",       # Should KEEP - medical descriptor
                    "the",                     # Should REMOVE - stop word
                    "bilateral",               # Should REMOVE - anatomical location alone
                    "lung collapse",           # Should KEEP - medical term
                    "again",                   # Should REMOVE - temporal descriptor
                    "subsegmental"             # Should KEEP - medical term
                ],
                "medium_confidence": [
                    "linear atelectasis",      # Should KEEP - medical term
                    "mild",                    # Should REMOVE - intensity modifier alone
                    "streaky opacity",         # Should KEEP - medical descriptor
                    "present",                 # Should REMOVE - generic observation
                    "volume loss",             # Should KEEP - medical term
                    "and",                     # Should REMOVE - stop word
                    "left lower lobe",         # Should REMOVE - anatomical location alone
                    "discoid atelectasis",     # Should KEEP - medical term
                    "previously",              # Should REMOVE - temporal descriptor
                    "compressive"              # Should KEEP - medical descriptor
                ],
                "low_confidence": [
                    "plate atelectasis",       # Should KEEP - medical term
                    "associated",              # Should REMOVE - non-medical word
                    "lung base opacity",       # Should KEEP - medical descriptor
                    "findings",                # Should REMOVE - vague descriptor
                    "basilar atelectasis",     # Should KEEP - medical term
                    "shows",                   # Should REMOVE - generic verb
                    "retrocardiac",            # Should KEEP - anatomical descriptor with context
                    "changes",                 # Should REMOVE - vague descriptor
                    "dependent atelectasis",   # Should KEEP - medical term
                    "demonstrates"             # Should REMOVE - generic verb
                ]
            },
            "Pneumonia": {
                "high_confidence": [
                    "consolidation",           # Should KEEP - medical term
                    "appears",                 # Should REMOVE - generic verb
                    "pneumonic infiltrate",    # Should KEEP - medical term
                    "currently",               # Should REMOVE - temporal descriptor
                    "air bronchogram",         # Should KEEP - medical term
                    "of",                      # Should REMOVE - stop word
                    "right upper lobe",        # Should REMOVE - anatomical location alone
                    "lobar pneumonia",         # Should KEEP - medical term
                    "suggests",                # Should REMOVE - generic verb
                    "inflammatory process"     # Should KEEP - medical term
                ],
                "medium_confidence": [
                    "bronchopneumonia",        # Should KEEP - medical term
                    "severe",                  # Should REMOVE - intensity modifier alone
                    "infectious process",      # Should KEEP - medical term
                    "evident",                 # Should REMOVE - generic observation
                    "patchy opacity",          # Should KEEP - medical descriptor
                    "with",                    # Should REMOVE - stop word
                    "bilateral",               # Should REMOVE - anatomical location alone
                    "bacterial pneumonia",     # Should KEEP - medical term
                    "recent",                  # Should REMOVE - temporal descriptor
                    "lung infection"           # Should KEEP - medical term
                ],
                "low_confidence": [
                    "aspiration pneumonia",    # Should KEEP - medical term
                    "compatible",              # Should REMOVE - non-medical word
                    "ground glass opacity",    # Should KEEP - medical term
                    "abnormality",             # Should REMOVE - vague descriptor
                    "pneumonitis",             # Should KEEP - medical term
                    "reveals",                 # Should REMOVE - generic verb
                    "multifocal",              # Should KEEP - medical descriptor
                    "pattern",                 # Should REMOVE - vague descriptor
                    "community acquired",      # Should KEEP - medical descriptor
                    "indicates"                # Should REMOVE - generic verb
                ]
            }
        }
    }
    
    # Save test data
    test_file = "test_keywords.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Created test keywords file: {test_file}")
    
    # Print expected results for verification
    print(f"\n📊 TEST DATA SUMMARY:")
    total_keywords = 0
    expected_keep = 0
    expected_remove = 0
    
    for condition, condition_data in test_data['conditions'].items():
        condition_total = 0
        for conf_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
            condition_total += len(condition_data[conf_level])
        total_keywords += condition_total
        print(f"  {condition}: {condition_total} keywords")
    
    print(f"  TOTAL: {total_keywords} keywords")
    
    # Manual count of expected keeps vs removes based on comments above
    print(f"\n🎯 EXPECTED RESULTS (manual verification):")
    print(f"  Should KEEP: ~{total_keywords // 2} keywords (medical terms)")
    print(f"  Should REMOVE: ~{total_keywords // 2} keywords (generic/stop words)")
    print(f"  Removal rate: ~50%")
    
    return test_file

def run_test():
    """Run the AI cleanup test"""
    print("🧪 Starting AI Cleanup Test...")
    print("=" * 50)
    
    # Create test file
    test_file = create_test_keywords_file()
    
    try:
        # Initialize AI cleanup with small batch size for testing
        print(f"\n🤖 Initializing AI cleanup...")
        cleanup = GeminiAICleanup()
        
        # Run cleanup with small batches (10 keywords per batch for testing)
        print(f"\n🔄 Running AI cleanup with batch_size=10...")
        print(f"📂 Input file: {test_file}")
        
        # This will process in small batches and save results back to test_file
        cleaned_data = cleanup.main(input_file=test_file, batch_size=10)
        
        print(f"\n✅ AI cleanup completed!")
        
        # Analyze results
        print(f"\n📈 ANALYZING RESULTS...")
        original_total = 0
        cleaned_total = 0
        
        # Load the cleaned file to see results
        with open(test_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        for condition_name, condition_data in result_data['conditions'].items():
            condition_original = 60  # 30 keywords per condition from our test data
            condition_cleaned = 0
            
            for conf_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
                if conf_level in condition_data:
                    condition_cleaned += len(condition_data[conf_level])
            
            original_total += condition_original
            cleaned_total += condition_cleaned
            
            print(f"  {condition_name}:")
            print(f"    Original: {condition_original} keywords")
            print(f"    Cleaned: {condition_cleaned} keywords")
            print(f"    Removed: {condition_original - condition_cleaned} keywords")
            print(f"    Removal rate: {((condition_original - condition_cleaned) / condition_original * 100):.1f}%")
        
        total_removed = original_total - cleaned_total
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"  Total original: {original_total} keywords")
        print(f"  Total cleaned: {cleaned_total} keywords") 
        print(f"  Total removed: {total_removed} keywords")
        print(f"  Overall removal rate: {(total_removed / original_total * 100):.1f}%")
        
        # Show some examples of what was kept
        print(f"\n🔍 SAMPLE KEPT KEYWORDS:")
        for condition_name, condition_data in result_data['conditions'].items():
            kept_keywords = []
            for conf_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
                if conf_level in condition_data:
                    kept_keywords.extend(condition_data[conf_level])
            
            if kept_keywords:
                sample_kept = kept_keywords[:5]  # Show first 5
                print(f"  {condition_name}: {sample_kept}")
        
        # Check if AI cleanup metadata was added
        if 'ai_cleanup' in result_data.get('metadata', {}):
            ai_stats = result_data['metadata']['ai_cleanup']
            print(f"\n📋 AI CLEANUP METADATA:")
            print(f"  Model used: {ai_stats.get('model_used')}")
            print(f"  Cleanup date: {ai_stats.get('cleanup_date')}")
            print(f"  Removal rate: {ai_stats.get('removal_rate')}")
        
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Test results saved in: {test_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test file
        try:
            if os.path.exists(test_file):
                print(f"\n🧹 Cleaning up test file: {test_file}")
                os.remove(test_file)
        except:
            print(f"⚠️  Could not remove test file: {test_file}")

if __name__ == "__main__":
    success = run_test()
    if success:
        print(f"\n🎉 AI Cleanup test PASSED!")
    else:
        print(f"\n💥 AI Cleanup test FAILED!") 