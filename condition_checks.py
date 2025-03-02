import os
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('condition_analysis.txt', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")

# The predefined conditions
PREDEFINED_CONDITIONS = {
    0: "No acute cardiopulmonary abnormality",
    1: "Cardiomegaly",
    2: "Pulmonary edema",
    3: "Pneumonia",
    4: "Pleural effusion",
    5: "Atelectasis",
    6: "Pneumothorax",
    7: "Consolidation"
}

# Common medical terms to help identify conditions
MEDICAL_KEYWORDS = [
    'acute', 'chronic', 'opacity', 'mass', 'nodule', 'fracture', 'effusion', 
    'pneumonia', 'cardiomegaly', 'edema', 'atelectasis', 'pneumothorax', 
    'consolidation', 'abnormality', 'disease', 'fibrosis', 'emphysema', 
    'congestion', 'infection', 'infiltrate', 'lesion', 'fluid', 'thickening',
    'enlarged', 'distention', 'hyperinflation', 'hypertension', 'failure',
    'tuberculosis', 'tumor', 'cancer', 'malignancy', 'bronchitis', 'copd',
    'asthma', 'embolism', 'aneurysm', 'sarcoidosis', 'silicosis'
]

# Section headers typically found in radiology reports
SECTION_HEADERS = ["INDICATION:", "COMPARISON:", "FINDINGS:", "IMPRESSION:"]

def read_text_file(file_path):
    """Read and return the contents of a text file."""
    if pd.isna(file_path) or file_path == '':
        return ''
    else:
        try:
            if len(file_path) >= 260 and os.name == 'nt' and not file_path.startswith('\\\\?\\'):
                file_path = '\\\\?\\' + os.path.abspath(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return text
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ''

def extract_findings_and_impression(text):
    """Extract the FINDINGS and IMPRESSION sections from a radiology report."""
    findings = ""
    impression = ""
    
    # Split the text into sections based on headers
    current_section = ""
    sections = {}
    lines = text.split('\n')
    
    for line in lines:
        for header in SECTION_HEADERS:
            if header in line or line.strip().upper() == header.strip(':'):
                current_section = header
                if current_section not in sections:
                    sections[current_section] = []
                break
        else:  # This is an 'else' clause for the for loop, not an if-else
            if current_section:
                sections[current_section].append(line)
    
    # Convert lists to strings
    for header, lines in sections.items():
        sections[header] = '\n'.join(lines).strip()
    
    # Extract findings and impression
    findings = sections.get("FINDINGS:", "")
    impression = sections.get("IMPRESSION:", "")
    
    return findings, impression

def extract_potential_conditions(text):
    """Extract potential medical conditions mentioned in the text."""
    if not text:
        return []
    
    # Preprocess text
    text = text.lower()
    
    # Remove common non-condition phrases
    text = re.sub(r'\b(no|without|absence of|negative for)\s+\w+', '', text)
    
    # Find all n-grams that contain medical keywords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w.lower() not in stop_words and len(w) > 2]
    
    # Check for medical keywords
    potential_conditions = []
    for i, token in enumerate(filtered_tokens):
        if token in MEDICAL_KEYWORDS or any(keyword in token for keyword in MEDICAL_KEYWORDS):
            # Try to capture the condition phrase (up to 3 words before and after)
            start = max(0, i - 3)
            end = min(len(filtered_tokens), i + 4)
            phrase = ' '.join(filtered_tokens[start:end])
            potential_conditions.append(phrase)
    
    # Also check for direct mentions of our predefined conditions
    for condition in PREDEFINED_CONDITIONS.values():
        condition_lower = condition.lower()
        if condition_lower in text and condition_lower != "no acute cardiopulmonary abnormality":
            if condition_lower not in [c.lower() for c in potential_conditions]:
                potential_conditions.append(condition_lower)
    
    return potential_conditions

def analyze_reports():
    """Analyze all reports to check for additional conditions."""
    logger.info("Starting condition analysis...")
    
    # Paths
    XAMI_MIMIC_PATH = os.path.abspath(os.path.join(
        os.getcwd(),
        'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0',
        'mimic-eye'
    )).replace('\\', '/')
    
    # Load the dataset
    try:
        df = pd.read_csv('test.csv')
        logger.info(f"Loaded {len(df)} records from test.csv")
    except Exception as e:
        logger.error(f"Error loading test.csv: {e}")
        return
    
    # Prepare report paths
    df['report_path'] = df['report_path'].str.replace(
        '{XAMI_MIMIC_PATH}', XAMI_MIMIC_PATH
    ).str.replace('\\', '/')
    df['report_path'] = df['report_path'].apply(lambda x: os.path.abspath(x) if pd.notna(x) else x)
    
    # Replace mimic-eye path if needed
    df['report_path'] = df['report_path'].str.replace(
        'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye',
        XAMI_MIMIC_PATH,
        regex=False
    )
    
    # Collect all conditions from reports
    all_conditions = []
    condition_frequencies = Counter()
    reports_analyzed = 0
    conditions_by_report = {}
    
    for idx, row in df.iterrows():
        if pd.isna(row['report_path']):
            continue
        
        report_text = read_text_file(row['report_path'])
        if not report_text:
            continue
        
        findings, impression = extract_findings_and_impression(report_text)
        
        # Combine findings and impression for condition extraction
        combined_text = f"{findings}\n{impression}"
        potential_conditions = extract_potential_conditions(combined_text)
        
        # Store conditions by report ID
        report_id = row.get('id', f"report_{idx}")
        conditions_by_report[report_id] = potential_conditions
        
        # Update statistics
        all_conditions.extend(potential_conditions)
        condition_frequencies.update(potential_conditions)
        reports_analyzed += 1
        
        # Log progress every 50 reports
        if reports_analyzed % 50 == 0:
            logger.info(f"Analyzed {reports_analyzed} reports...")
    
    # Compare with predefined conditions
    predefined_set = set([c.lower() for c in PREDEFINED_CONDITIONS.values()])
    all_conditions_set = set(all_conditions)
    
    # Identify potentially missing conditions (those not in predefined list)
    missing_conditions = all_conditions_set - predefined_set
    
    # Calculate condition frequencies
    logger.info(f"\nAnalyzed {reports_analyzed} reports in total.")
    logger.info("\n===== Condition Analysis Results =====")
    
    # Log statistics about predefined conditions
    logger.info("\nPredefined conditions frequency:")
    for condition_id, condition in PREDEFINED_CONDITIONS.items():
        condition_lower = condition.lower()
        count = sum(1 for c in all_conditions if condition_lower in c)
        percentage = (count / reports_analyzed) * 100 if reports_analyzed > 0 else 0
        logger.info(f"- {condition}: {count} occurrences ({percentage:.2f}%)")
    
    # Log top potential missing conditions
    logger.info("\nTop 20 potential additional conditions found:")
    for condition, count in condition_frequencies.most_common(50):
        if not any(pred_cond.lower() in condition for pred_cond in predefined_set):
            percentage = (count / reports_analyzed) * 100 if reports_analyzed > 0 else 0
            logger.info(f"- {condition}: {count} occurrences ({percentage:.2f}%)")
    
    # Generate detailed report for each unique condition
    unique_conditions = defaultdict(int)
    for conditions in conditions_by_report.values():
        for condition in conditions:
            unique_conditions[condition] += 1
    
    logger.info(f"\nTotal unique conditions found: {len(unique_conditions)}")
    
    # Report findings
    logger.info("\n===== Recommendation =====")
    if missing_conditions:
        significant_missing = [c for c, count in unique_conditions.items() 
                              if c in missing_conditions and 
                              count > reports_analyzed * 0.05]  # More than 5% occurrence
        if significant_missing:
            logger.info("Consider adding these conditions to your model:")
            for condition in significant_missing[:10]:  # Top 10
                count = unique_conditions[condition]
                percentage = (count / reports_analyzed) * 100
                logger.info(f"- {condition}: {count} occurrences ({percentage:.2f}%)")
        else:
            logger.info("The current condition list appears to cover the most common conditions in the dataset.")
    else:
        logger.info("The current condition list covers all identified conditions in the dataset.")

if __name__ == "__main__":
    analyze_reports()
