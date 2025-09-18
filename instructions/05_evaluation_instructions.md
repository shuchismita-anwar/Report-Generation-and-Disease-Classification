# Evaluation Instructions

## Overview

This document provides instructions for evaluating the generated medical reports against ground truth reports.

## Prerequisites

1. Activate virtual environment: `.\venv\Scripts\activate`
2. Ensure inference has been completed (see `04_inference_instructions.md`)
3. Verify ground truth reports are available in `.\cleaned_reports\`
4. For single patient evaluation, ensure `.\main\latest_fixed_analysis.txt` exists

## Evaluation Scripts

### 1. Medical Report Evaluator (Comprehensive)

**Script:** `.\main\evaluation\medical_report_evaluator.py`

**Description:** Comprehensive evaluation of multiple generated reports against ground truth.

**Usage:**

```powershell
python .\main\evaluation\medical_report_evaluator.py
```

**Input Requirements:**

- **Generated Reports:** JSON files in `.\main\real_analysis_results\`
- **Ground Truth Reports:** Text files in `.\cleaned_reports\`

**Output Location:** `.\main\evaluation\output\reports\`

**Output Files:**

- Detailed evaluation metrics per patient
- Aggregate performance statistics
- BLEU, ROUGE, and custom medical similarity scores
- Statistical analysis reports

### 2. Single Patient Evaluation (Terminal Output)

**Script:** `.\main\evaluation\test_specific_patient_evaluation.py`

**Description:** Focused evaluation for single patient analysis from random X-ray analysis.

**Usage:**

```powershell
python .\main\evaluation\test_specific_patient_evaluation.py
```

**Input Requirements:**

- **AI Generated Report:** `.\main\latest_fixed_analysis.txt`
- **Ground Truth Reports:** Corresponding file in `.\cleaned_reports\`

**Output Location:** **Terminal only** (results not saved to file)

**Output Information:**

- BLEU scores (1-4 gram)
- ROUGE scores (ROUGE-L, ROUGE-1, ROUGE-2)
- Medical terminology alignment
- Clinical keyword overlap
- Semantic similarity metrics

## Data Organization for Evaluation

### Required Structure

```
Root/
├── main/
│   ├── real_analysis_results/          # Generated reports (JSON)
│   │   ├── patient_001_analysis.json
│   │   ├── patient_002_analysis.json
│   │   └── ...
│   ├── latest_fixed_analysis.txt       # Single patient report
│   └── evaluation/
│       └── output/
│           └── reports/                # Evaluation results
└── cleaned_reports/                    # Ground truth reports
    ├── patient_001.txt
    ├── patient_002.txt
    └── ...
```

### Data Matching

- AI-generated reports are matched to ground truth by patient ID
- Ensure consistent naming convention between generated and ground truth files
- Missing ground truth files will be noted in evaluation logs

## Statistical Analysis

### 3. Gaze Attention Validation Statistics

**Script:** `.\main\mean-sd.py`

**Description:** Analyzes gaze attention validation metrics across all generated reports.

**Prerequisites:**

- Generated reports with gaze attention validation data in `.\main\real_analysis_results\`
- Reports must contain `gaze_attention_validation` key in JSON structure

**Usage:**

```powershell
python .\main\mean-sd.py
```

**Output:**

- Mean and standard deviation of gaze attention correlation
- Distribution analysis of attention alignment metrics
- Statistical summary of model-human attention agreement

**Key Metrics:**

- Pearson correlation coefficients
- Jensen-Shannon divergence scores
- Spatial attention overlap measures

## Evaluation Metrics

### Lexical Similarity

- **BLEU-1 to BLEU-4:** N-gram precision scores
- **ROUGE-L:** Longest common subsequence
- **ROUGE-1/ROUGE-2:** Unigram and bigram recall

### Semantic Similarity

- **Clinical Keyword Overlap:** Medical terminology alignment
- **Anatomical Region Accuracy:** Spatial finding correlation
- **Confidence Score Alignment:** Prediction certainty matching

### Clinical Relevance

- **Disease Detection Accuracy:** Pathology identification rates
- **False Positive Analysis:** Over-reporting assessment
- **Critical Finding Coverage:** Important pathology capture

### Attention Validation

- **Gaze Correlation:** Human-AI attention alignment
- **Regional Focus Accuracy:** Anatomical attention distribution
- **Attention-Report Consistency:** Spatial findings correlation

## Usage Examples

### Comprehensive Report Evaluation

```powershell
# Activate environment
.\venv\Scripts\activate

# Run comprehensive evaluation
python .\main\evaluation\medical_report_evaluator.py

# Check results
Get-ChildItem .\main\evaluation\output\reports\
```

### Single Patient Quick Evaluation

```powershell
# Activate environment
.\venv\Scripts\activate

# Generate single patient report
python .\main\random_xray_analysis.py

# Evaluate the generated report
python .\main\evaluation\test_specific_patient_evaluation.py
```

### Attention Statistics Analysis

```powershell
# Activate environment
.\venv\Scripts\activate

# Ensure all reports are generated
python .\main\medical_report_generator.py

# Analyze attention validation statistics
python .\main\mean-sd.py
```

## Output Interpretation

### Comprehensive Evaluation Results

**Location:** `.\main\evaluation\output\reports\`

**Files Include:**

- `detailed_report_YYYYMMDD_HHMMSS.md` - Comprehensive evaluation report with detailed metrics
- `executive_summary_YYYYMMDD_HHMMSS.md` - Executive summary of evaluation results

## Troubleshooting

### Common Issues

- **Missing Ground Truth:** Verify files exist in `.\cleaned_reports\`
- **Format Mismatches:** Ensure consistent patient ID mapping
- **JSON Parsing Errors:** Validate generated report JSON structure
- **Evaluation Crashes:** Check for malformed text files or encoding issues

### Debug Steps

1. Verify input data availability and format
2. Check file permissions for output directories
3. Validate JSON structure of generated reports
4. Ensure ground truth reports are properly formatted
5. Monitor memory usage during large-scale evaluation
