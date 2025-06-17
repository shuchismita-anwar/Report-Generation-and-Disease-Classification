#!/usr/bin/env python3
"""
Task 1.2: Examine Real Transcript Content
Goal: Understand transcript JSON structure and content
"""

import pandas as pd
import json
import random
from pathlib import Path

def examine_transcript_content():
    """Examine 10 random transcript.json files from dataset"""
    
    print("="*60)
    print("TASK 1.2: TRANSCRIPT CONTENT EXAMINATION")
    print("="*60)
    
    # Load train.csv to get transcript paths
    train_csv = Path("dataset_splits/train.csv")
    
    if not train_csv.exists():
        print(f"❌ ERROR: {train_csv} does not exist!")
        return False
    
    print(f"📁 Loading transcript paths from: {train_csv}")
    df = pd.read_csv(train_csv)
    
    # Get all transcript paths
    transcript_paths = df['transcript_path'].dropna().tolist()
    print(f"📊 Total transcript paths available: {len(transcript_paths)}")
    
    # Randomly select 10 transcripts
    random.seed(42)  # For reproducible results
    selected_paths = random.sample(transcript_paths, min(10, len(transcript_paths)))
    
    print(f"🎯 Selected {len(selected_paths)} transcripts for examination:")
    
    transcript_data = []
    
    for i, transcript_path in enumerate(selected_paths, 1):
        print(f"\n📄 TRANSCRIPT {i}/10:")
        print(f"   Path: {transcript_path}")
        
        # Try to load the transcript file
        transcript_file = Path(transcript_path)
        
        # Check if file exists (might need to try relative path)
        if not transcript_file.exists():
            # Try relative to current directory
            alt_path = Path(".") / transcript_path
            if alt_path.exists():
                transcript_file = alt_path
            else:
                print(f"   ❌ File not found: {transcript_path}")
                continue
        
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_json = json.load(f)
            
            print(f"   ✅ Successfully loaded JSON")
            
            # Analyze JSON structure
            print(f"   🔑 JSON keys: {list(transcript_json.keys())}")
            
            # Extract full_text if available
            if 'full_text' in transcript_json:
                full_text = transcript_json['full_text']
                print(f"   📝 Full text length: {len(full_text)} characters")
                print(f"   📝 Word count: {len(full_text.split())} words")
                
                # Show first 200 characters of text
                preview = full_text[:200] + "..." if len(full_text) > 200 else full_text
                print(f"   📖 Text preview: \"{preview}\"")
                
                # Store for analysis
                transcript_data.append({
                    'path': transcript_path,
                    'text': full_text,
                    'char_count': len(full_text),
                    'word_count': len(full_text.split()),
                    'json_keys': list(transcript_json.keys())
                })
                
            else:
                print(f"   ⚠️  No 'full_text' key found")
                print(f"   🔑 Available keys: {list(transcript_json.keys())}")
                
                # Try to find text in other keys
                for key, value in transcript_json.items():
                    if isinstance(value, str) and len(value) > 50:
                        print(f"   📖 Text in '{key}': \"{value[:100]}...\"")
            
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON parsing error: {e}")
        except Exception as e:
            print(f"   ❌ Error loading file: {e}")
    
    # Analyze collected transcript data
    if transcript_data:
        print(f"\n📊 TRANSCRIPT ANALYSIS SUMMARY:")
        print(f"   Successfully parsed: {len(transcript_data)}/{len(selected_paths)} transcripts")
        
        # Text length statistics
        char_counts = [t['char_count'] for t in transcript_data]
        word_counts = [t['word_count'] for t in transcript_data]
        
        print(f"   📏 Character count - Min: {min(char_counts)}, Max: {max(char_counts)}, Avg: {sum(char_counts)/len(char_counts):.1f}")
        print(f"   📏 Word count - Min: {min(word_counts)}, Max: {max(word_counts)}, Avg: {sum(word_counts)/len(word_counts):.1f}")
        
        # Analyze common JSON structure
        all_keys = set()
        for t in transcript_data:
            all_keys.update(t['json_keys'])
        print(f"   🔑 All JSON keys found: {sorted(all_keys)}")
        
        # Show examples of medical terminology
        print(f"\n🏥 MEDICAL TERMINOLOGY ANALYSIS:")
        medical_terms = set()
        
        for t in transcript_data:
            text = t['text'].lower()
            # Look for common medical terms
            terms_to_find = [
                'heart', 'lung', 'chest', 'cardiomegaly', 'pneumonia', 'effusion',
                'bilateral', 'opacity', 'consolidation', 'atelectasis', 'pneumothorax',
                'mediastinum', 'hilum', 'costophrenic', 'diaphragm', 'pleural',
                'pulmonary', 'cardiac', 'thoracic', 'ribs', 'clavicle'
            ]
            
            found_terms = [term for term in terms_to_find if term in text]
            medical_terms.update(found_terms)
        
        print(f"   🔍 Medical terms found: {sorted(medical_terms)}")
        
        # Show sample descriptive phrases
        print(f"\n📝 SAMPLE DESCRIPTIVE CONTENT:")
        for i, t in enumerate(transcript_data[:3], 1):
            print(f"   Sample {i}:")
            # Extract sentences with medical terms
            sentences = t['text'].split('.')
            medical_sentences = []
            for sentence in sentences[:5]:  # First 5 sentences
                sentence = sentence.strip()
                if any(term in sentence.lower() for term in ['heart', 'lung', 'chest', 'bilateral', 'opacity']):
                    medical_sentences.append(sentence)
            
            for sentence in medical_sentences[:2]:  # Show first 2 medical sentences
                print(f"     \"{sentence.strip()}\"")
    
    else:
        print(f"\n❌ No transcripts were successfully parsed!")
        return False
    
    print(f"\n✅ TASK 1.2 COMPLETE - Transcript content examined successfully!")
    print(f"   📊 Parsed {len(transcript_data)} real transcript files")
    print(f"   🏥 Identified medical terminology and descriptive content")
    print(f"   📋 Confirmed JSON structure with 'full_text' field")
    
    return True

if __name__ == "__main__":
    examine_transcript_content() 