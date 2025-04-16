import pandas as pd
import json
import os
from tqdm import tqdm

# Read the CSV file
df = pd.read_csv('condition_labels.csv')

# Create or overwrite the output file
with open('conditioncrosscheck.txt', 'w', encoding='utf-8') as outfile:
    # Iterate through each row in the CSV
    for index, row in tqdm(df.iterrows(), total=len(df)):
        transcript_path = row['transcript_path']
        conditions = row['conditions']
        
        try:
            # Check if the transcript comes from EyeGaze (in json format) and extract "full_text"
            if "EyeGaze" in transcript_path:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                report_text = json_data["full_text"]
            else:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    report_text = f.read()
            
            # Write the transcript and conditions to the output file
            outfile.write(f"{report_text} --> {conditions}\n")
            
        except Exception as e:
            print(f"Error processing {transcript_path}: {str(e)}")

print("Done")