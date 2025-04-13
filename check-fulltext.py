import os
import json
import pandas as pd

# Function to extract text from transcript files
def extract_transcript_text(transcript_path):
    """Extract text from transcript files, handling both EyeGaze JSON and regular text formats"""
    try:
        # Check if the file exists
        if not os.path.exists(transcript_path):
            print(f"File not found: {transcript_path}")
            return None
            
        # Handle EyeGaze JSON format
        if 'EyeGaze' in transcript_path and transcript_path.endswith('transcript.json'):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract the full_text field
                if 'full_text' in data:
                    return data['full_text']
                else:
                    print(f"No 'full_text' field found in {transcript_path}")
                    return None
        # Handle regular text transcripts
        else:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
                
    except Exception as e:
        print(f"Error processing {transcript_path}: {str(e)}")
        return None

def main():
    print("Loading trans-data.csv...")
    
    try:
        # Read the CSV file
        df = pd.read_csv('trans-data.csv')
        print(f"Successfully loaded {len(df)} rows")
        
        # Create a new column for transcript text
        df['transcript_text'] = None
        
        # Count different transcript types
        eyegaze_count = sum(1 for path in df['transcript_path'] if 'EyeGaze' in path)
        reflacx_count = sum(1 for path in df['transcript_path'] if 'REFLACX' in path)
        print(f"EyeGaze transcripts: {eyegaze_count}")
        print(f"REFLACX transcripts: {reflacx_count}")
        
        # Process a sample of transcripts
        print("\nProcessing sample transcripts...")
        sample_size = 5  # Adjust this number to process more samples
        
        processed_count = 0
        success_count = 0
        eyegaze_processed = 0
        reflacx_processed = 0
        
        for i, row in df.iterrows():
            if processed_count >= sample_size:
                break
                
            transcript_path = row['transcript_path']
            is_eyegaze = 'EyeGaze' in transcript_path
            
            print(f"\nProcessing transcript {processed_count+1} (ID: {row['id']})")
            print(f"Is EyeGaze: {is_eyegaze}")
            
            # Extract text from transcript
            text = extract_transcript_text(transcript_path)
            
            if text is not None:
                success_count += 1
                if is_eyegaze:
                    eyegaze_processed += 1
                else:
                    reflacx_processed += 1
                    
                print(f"Successfully extracted text ({len(text)} chars)")
                print(f"Sample: {text[:100]}...")
                
                # Update the dataframe
                df.at[i, 'transcript_text'] = text
            else:
                print("Failed to extract text")
                
            processed_count += 1
        
        # Print summary
        print("\n===== SUMMARY =====")
        print(f"Processed {processed_count} transcripts")
        print(f"Successfully extracted text from {success_count} transcripts")
        print(f"EyeGaze transcripts processed: {eyegaze_processed}")
        print(f"REFLACX transcripts processed: {reflacx_processed}")
        
        # Save the updated dataframe to a new CSV file
        sample_df = df.head(sample_size)
        sample_df.to_csv('trans-data-with-text.csv', index=False)
        print("\nSaved sample with extracted text to 'trans-data-with-text.csv'")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
