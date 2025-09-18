# Inference Instructions

## Overview

This document provides instructions for running inference and generating medical reports using the trained models.

## Prerequisites

1. Activate virtual environment: `.\venv\Scripts\activate`
2. Ensure trained models are available in `.\main\output\`
3. Verify LM Studio is running with API enabled (see `02_environment_setup.md`)
4. Ensure `.env` file is configured in `.\main\` directory

## Inference Scripts

### 1. Medical Report Generator (Batch Processing)

**Script:** `.\main\run_medical_report_generator.py`

**Description:** Main script for generating medical reports for multiple patients with comprehensive analysis.

**Note:** This script imports and uses `.\main\medical_report_generator.py` as a module. Running `medical_report_generator.py` directly will do nothing.

**Usage:**

```powershell
python .\main\run_medical_report_generator.py
```

**Outputs:**

- **Reports (JSON):** `.\main\real_analysis_results\`
- **Gaze Attention Analysis (PNG):** `.\main\real_analysis_results\gaze_attention\`

**Features:**

- Processes multiple chest X-ray images
- Generates structured medical reports
- Creates gaze attention visualizations
- Saves comprehensive analysis data

### 2. Single Patient X-ray Analysis

**Script:** `.\main\random_xray_analysis.py`

**Description:** Script for analyzing a specific patient and generating detailed report.

**Note:** This script currently hardcodes a specific patient's DICOM ID (can be modified to select random patients). It imports and uses `.\main\medical_report_generator.py` as a module for the core functionality.

**Usage:**

```powershell
python .\main\random_xray_analysis.py
```

**Outputs:**

- **AI Generated Report:** `.\main\latest_fixed_analysis.txt`
- **Gaze Attention Analysis:** `.\main\gaze_attention_analysis\`
- **Debug Prompt:** `.\main\debug_ai_prompt.txt`

**Features:**

- Analyzes hardcoded patient DICOM ID (configurable in script)
- Generates single comprehensive report
- Creates detailed attention analysis
- Saves full LLM prompt for debugging

## Output Structures

### 1. Medical Report Generator Outputs

#### Reports Directory: `.\main\real_analysis_results\`

- Contains JSON files with structured report data
- Each file includes:
  - Patient information
  - Disease predictions
  - Generated medical report
  - Metadata and confidence scores
  - Gaze attention validation metrics

#### Gaze Attention Directory: `.\main\real_analysis_results\gaze_attention\`

- PNG visualizations of attention analysis
- Overlay comparisons between AI and human attention
- Spatial attention heatmaps
- Regional attention focus maps

### 2. Single Patient Analysis Outputs

#### Analysis Text: `.\main\latest_fixed_analysis.txt`

- Final generated medical report for single patient
- Formatted as clinical report
- Used as AI-generated ground truth for evaluation

#### Attention Analysis: `.\main\gaze_attention_analysis\`

- Detailed attention visualizations
- Comparison plots between AI and human gaze patterns
- Regional attention distribution analysis

#### Debug Information: `.\main\debug_ai_prompt.txt`

- Complete prompt sent to LLM
- Includes all input data and formatting
- Useful for prompt engineering and debugging

## Key Components and Locations

### Static Resources

#### Anatomical Regions

**Location:** `.\main\keyword_prediction\anatomical_results\anatomical_regions_dict.py`

- Contains `ANATOMICAL_REGIONS` dictionary
- Maps conditions to anatomical locations
- Used for spatial analysis and reporting

#### Clinical Keywords

**Location:** `.\main\keyword_prediction\extracted_keywords_result_final.json`

- Contains condition-specific keywords
- Organized by confidence levels
- Used for clinical terminology alignment

### Model Dependencies

- Trained multimodal models from `.\main\output\`
- Vision Transformer: `google/vit-base-patch16-224-in21k`
- Clinical BERT: `emilyalsentzer/Bio_ClinicalBERT`

### Core Module

**`.\main\medical_report_generator.py`** - Contains the `MedicalReportGenerator` class and core functionality. This is imported by both inference scripts above and should not be run directly.

## Inference Process Flow

1. **Image Loading:** Load chest X-ray from `.\data_dump\output\img_png\`
2. **Model Inference:** Run multimodal disease prediction
3. **Keyword Extraction:** Extract relevant clinical keywords
4. **Attention Analysis:** Generate and analyze attention maps
5. **Region Mapping:** Map predictions to anatomical regions
6. **LLM Report Generation:** Send structured prompt to language model
7. **Report Formatting:** Format and save final medical report

## Usage Examples

### Generate Reports for All Test Patients

```powershell
# Activate environment
.\venv\Scripts\activate

# Run batch processing
python .\main\run_medical_report_generator.py
```

### Analyze Single Specific Patient

```powershell
# Activate environment
.\venv\Scripts\activate

# Run single patient analysis (hardcoded DICOM ID)
python .\main\random_xray_analysis.py
```

### View Generated Report

```powershell
# Check the generated report
Get-Content .\main\latest_fixed_analysis.txt
```

## Configuration Options

### Report Templates

- **Standard:** Balanced clinical report
- **Detailed:** Comprehensive analysis with extended findings
- **Concise:** Brief summary for screening

### Attention Analysis Settings

- Gaze attention correlation thresholds
- Spatial attention weighting
- Regional attention mapping parameters

## Quality Assurance Features

### Confidence-Based Reporting

- **>70%:** Report directly and confidently
- **50-70%:** Use appropriate clinical uncertainty
- **<50%:** Do not report finding

### Medical Safety Features

- Attention data prohibition in clinical reports
- Validation against human radiologist patterns
- Clinical terminology alignment
- Device finding detection and reporting

## Troubleshooting

### Common Issues

- **LM Studio Connection:** Verify API server is running on correct port
- **Model Loading:** Ensure trained models exist in output directory
- **Memory Issues:** Reduce batch processing size or use GPU optimization
- **Output Permissions:** Verify write permissions to output directories

### Debug Information

- Check `.\main\debug_ai_prompt.txt` for prompt issues
- Review logs in respective output directories
- Monitor GPU memory usage during inference
- Verify image paths and data availability
