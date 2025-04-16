import pandas as pd
import re
import spacy
from spacy.tokens import Span
from negspacy.negation import Negex
from tqdm import tqdm

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Add Negex component
nlp.add_pipe("negex", config={"ent_types": ["CONDITION"]})

CONDITION_SYNONYMS = {
    'Atelectasis': ['atelectasis', 'atelectatic'],
    'Cardiomegaly': ['cardiomegaly', 'enlarged heart', 'cardiac enlargement'],
    'Consolidation': ['consolidation', 'consolidative', 'airspace consolidation', 'pulmonary consolidation'],
    'Edema': ['edema', 'pulmonary edema', 'fluid overload', 'pulmonary congestion'],
    'Enlarged Cardiomediastinum': ['widened mediastinum', 'mediastinal widening', 'cardiac silhouette enlargement'],
    'Fracture': ['fracture', 'rib fracture'],
    'Lung Lesion': ['lung lesion', 'pulmonary lesion', 'lung mass', 'nodule', 'pulmonary nodule'],
    'Lung Opacity': ['lung opacity', 'opacity', 'opacities', 'pulmonary opacity', 'infiltrate', 'infiltrates', 'hazy opacity'],
    'No Finding': ['no acute', 'normal', 'unremarkable'],
    'Pleural Effusion': ['pleural effusion', 'effusion', 'pleural fluid', 'pleural effusions', 'large pleural effusions', 'large pleural effusion'],
    'Pleural Other': ['pleural abnormality', 'pleural thickening'],
    'Pneumonia': [' pneumonia', 'pneumonic', 'pneumonitis'],
    'Pneumothorax': ['pneumothorax', 'pneumo', 'pneumothoraces'],
    'Support Devices': ['support devices', 'catheter', 'tube', 'ventilator', 'line', 'drain', 'device']
}

# Initialize counts
condition_counts = {condition: 0 for condition in CONDITION_SYNONYMS.keys()}

# Load data
df = pd.read_csv('trans-data.csv')

def check_for_condition(doc, synonyms):
    """Check if any synonym in the doc is non-negated."""
    spans = []
    for synonym in synonyms:
        pattern = r"(?<!\w)" + re.escape(synonym) + r"(?!\w)"
        for match in re.finditer(pattern, doc.text):
            start_char, end_char = match.start(), match.end()
            start_token = end_token = None
            for token in doc:
                if token.idx >= start_char and start_token is None:
                    start_token = token.i
                if token.idx + len(token.text) > end_char and end_token is None:
                    end_token = token.i + 1
            if start_token is not None and end_token is not None:
                spans.append((Span(doc, start_token, end_token, label="CONDITION"), start_char, end_char))

    # Sort spans by length (shorter first) and deduplicate by position
    spans.sort(key=lambda x: (x[2] - x[1], x[1]))  # Sort by length, then start position
    unique_spans = []
    used_tokens = set()
    for span, start, end in spans:
        span_tokens = set(range(span.start, span.end))
        if not span_tokens & used_tokens:  # No overlap with previously used tokens
            unique_spans.append(span)
            used_tokens.update(span_tokens)

    # Set entities and process with pipeline
    doc.ents = unique_spans
    # Process the doc through the pipeline to apply negex
    processed_doc = nlp(doc.text)
    processed_doc.ents = unique_spans

    # Check for non-negated entities
    for ent in processed_doc.ents:
        if ent.label_ == "CONDITION" and not ent._.negex:
            return True
    return False

# Process each report
print("Processing reports...")
results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing reports"):
    transcript_path = row['transcript_path']
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            report_text = f.read()

        # Create a doc without processing the pipeline yet
        doc = nlp.make_doc(report_text.lower())  # Lowercase for consistency
        
        # Initialize conditions for this report
        report_conditions = []

        # Process "No Finding" separately
        if any(re.search(r"(?<!\w)" + re.escape(syn) + r"(?!\w)", doc.text) for syn in CONDITION_SYNONYMS['No Finding']):
            report_conditions.append('No Finding')
            condition_counts['No Finding'] += 1

        # Process other conditions
        for condition, synonyms in CONDITION_SYNONYMS.items():
            if condition != 'No Finding':
                # Create a fresh doc for each condition to avoid entity conflicts
                temp_doc = nlp.make_doc(report_text.lower())
                if check_for_condition(temp_doc, synonyms):
                    report_conditions.append(condition)
                    condition_counts[condition] += 1
        
        # Add to results
        results.append({
            'transcript_path': transcript_path,
            'conditions': '|'.join(report_conditions) if report_conditions else 'None'
        })

    except Exception as e:
        print(f"Error processing {transcript_path}: {e}")
        results.append({
            'transcript_path': transcript_path,
            'conditions': 'Error'
        })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('condition_labels.csv', index=False)
print("\nResults saved to condition_labels.csv")

# Print condition counts
print("\nCondition counts:")
for condition, count in condition_counts.items():
    print(f"{condition}: {count} positive samples")