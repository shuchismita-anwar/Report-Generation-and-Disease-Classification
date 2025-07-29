# Medical Report Generation and Disease Classification - Instructions

## Overview

This repository contains the complete codebase for **Medical Report Generation and Disease Classification** using multimodal deep learning with eye-gaze attention mechanisms.

## 🚨 Important Notes

- **No datasets provided** - As instructed, datasets are not included in the supplementary zip
- **No trained models provided** - Trained models are not included in the supplementary zip
- **Barebones code only** - Supervisors indicated they won't actually run the code
- **Complete documentation** - All necessary instructions are provided for full reproduction

## Instructions Overview

This instructions folder contains comprehensive documentation organized into separate files:

### 📁 Instruction Files

| File                              | Description                            | Purpose                                |
| --------------------------------- | -------------------------------------- | -------------------------------------- |
| **01_dataset_setup.md**           | Dataset organization and preprocessing | Required data structure and setup      |
| **02_environment_setup.md**       | Virtual environment and configuration  | Python environment and LM Studio setup |
| **03_training_instructions.md**   | Model training with hyperparameters    | Training all 8 model variants          |
| **04_inference_instructions.md**  | Report generation and analysis         | Running inference and creating reports |
| **05_evaluation_instructions.md** | Performance evaluation and metrics     | Evaluating generated reports           |
| **06_prompt_template.md**         | LLM prompt engineering details         | Complete prompt template specification |

## 🚀 Quick Start Guide

### 1. Environment Setup

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure LM Studio

Create `.env` file in `.\main\` directory:

```env
LM_STUDIO_HOST=http://127.0.0.1
LM_STUDIO_PORT=1234
LM_STUDIO_MODEL=<your_model_name>
```

### 3. Dataset Preparation

- Download MIMIC-Eye dataset
- Organize in `.\dataset_splits\` (train/val/test)
- Run preprocessing: `python .\scripts\preprocess.py`

### 4. Training Models

Run training scripts in `.\main\` directory:

```powershell
# Enhanced gaze model (recommended)
python .\main\0.(full+enhanced_gaze)_training_mimic_on_chexpert_optimized.py

# Full ablation study
python .\main\1.(full)_training_mimic_on_chexpert_optimized.py
python .\main\2.(fixation_removed)_training_mimic_on_chexpert_optimized.py
python .\main\3.(transcript_removed)_training_mimic_on_chexpert_optimized.py
python .\main\4.(bbox_removed)_training_mimic_on_chexpert_optimized.py
```

### 5. Generate Reports

```powershell
# Batch processing
python .\main\run_medical_report_generator.py

# Single patient analysis (hardcoded DICOM ID)
python .\main\random_xray_analysis.py
```

### 6. Evaluate Performance

```powershell
# Comprehensive evaluation
python .\main\evaluation\medical_report_evaluator.py

# Single patient evaluation
python .\main\evaluation\test_specific_patient_evaluation.py

# Attention statistics
python .\main\mean-sd.py
```

## 📂 Project Structure

```
Root/
├── instructions/                    # This documentation
│   ├── README.md                   # This file
│   ├── 01_dataset_setup.md
│   ├── 02_environment_setup.md
│   ├── 03_training_instructions.md
│   ├── 04_inference_instructions.md
│   ├── 05_evaluation_instructions.md
│   └── 06_prompt_template.md
├── main/                           # Main scripts (formerly new_scripts)
│   ├── [training_scripts].py       # 8 model training variants
│   ├── medical_report_generator.py # Core report generation module
│   ├── run_medical_report_generator.py # Batch report generation script
│   ├── random_xray_analysis.py     # Single patient analysis (hardcoded DICOM ID)
│   ├── evaluation/                 # Evaluation scripts
│   ├── output/                     # Trained models (created during training)
│   ├── real_analysis_results/      # Generated reports (JSON)
│   ├── gaze_attention_analysis/    # Attention visualizations
│   └── latest_fixed_analysis.txt   # Single patient report
├── dataset_splits/                 # Training data organization
├── data_dump/output/               # Preprocessed data
├── cleaned_reports/                # Ground truth reports
├── venv/                          # Virtual environment
└── [other project files]
```

## 🔬 Model Variants

### Enhanced Gaze Models

1. **Standard Enhanced Gaze** - Primary multimodal model with enhanced attention
2. **Experiment (Train+Val)** - Uses combined train+validation for training

### Ablation Study Models

3. **Full Model** - Complete baseline with all modalities
4. **Fixation Removed** - Without eye fixation sequences
5. **Transcript Removed** - Without radiologist transcripts
6. **Bounding Box Removed** - Without anatomical bounding boxes

### Baseline Models

7. **Original MIMIC** - Original training configuration
8. **ViT Only** - Pure vision transformer without multimodality

## 📊 Key Output Locations

| Component              | Location                                       | Description                                       |
| ---------------------- | ---------------------------------------------- | ------------------------------------------------- |
| **Trained Models**     | `.\main\output\`                               | All model checkpoints and weights                 |
| **Generated Reports**  | `.\main\real_analysis_results\`                | JSON reports from batch processing                |
| **Single Report**      | `.\main\latest_fixed_analysis.txt`             | Text report for evaluation                        |
| **Attention Analysis** | `.\main\real_analysis_results\gaze_attention\` | Visual attention comparisons                      |
| **Evaluation Results** | `.\main\evaluation\output\reports\`            | Detailed and executive summary reports (Markdown) |
| **Debug Information**  | `.\main\debug_ai_prompt.txt`                   | LLM prompt debugging                              |

## 🧩 Key Components

### Static Medical Knowledge

- **Anatomical Regions:** `.\main\keyword_prediction\anatomical_results\anatomical_regions_dict.py`
- **Clinical Keywords:** `.\main\keyword_prediction\extracted_keywords_result_final.json`

### Data Dependencies

- **Image Data:** `.\data_dump\output\img_png\`
- **Fixation Data:** `.\data_dump\output\fix_seq\`
- **Bounding Boxes:** `.\data_dump\output\bbox_mask\`
- **Statistics:** `.\fixstats.npz`

## ⚙️ Hyperparameters Summary

### Enhanced Gaze Models

- Learning Rate: `6e-6`
- Batch Size: `8`/`32` (memory dependent)
- Epochs: `40`
- Scheduler: `cosine`

### Baseline Models (Only "training_mimic_on_chexpert.py" and "vit_chexpert_training.py")

- Learning Rate: `5e-6` (MIMIC) / `5e-5` (ViT)
- Batch Size: `8`/`32` (multimodal) / `128` (ViT)
- Epochs: `35` (MIMIC) / `20` (ViT)

## 🎯 Evaluation Metrics

### Report Quality

- **BLEU-1 to BLEU-4** - N-gram precision
- **ROUGE-L/1/2** - Recall-oriented metrics
- **Clinical Keyword Overlap** - Medical terminology alignment

### Attention Validation

- **Pearson Correlation** - Human-AI attention alignment
- **Jensen-Shannon Divergence** - Attention distribution similarity
- **Spatial Overlap** - Regional attention agreement

## 🛠️ Troubleshooting

### Common Issues

- **Memory Errors:** Reduce batch size or enable gradient checkpointing
- **LM Studio Connection:** Verify API server and port configuration
- **Data Loading:** Check preprocessing and file paths
- **Model Loading:** Ensure sufficient disk space for outputs

### Debug Resources

- Check individual instruction files for detailed troubleshooting
- Review `.\main\debug_ai_prompt.txt` for LLM prompt issues
- Monitor training logs in respective output directories
- Verify virtual environment activation and dependencies

## 📄 Citation and Usage

This codebase implements the methodology described in our conference paper submission. The complete pipeline includes:

1. **Multimodal Training** with eye-gaze attention mechanisms
2. **Disease Classification** using Vision Transformers and Clinical BERT
3. **Attention Analysis** comparing AI and human radiologist patterns
4. **Report Generation** via optimized clinical prompt templates
5. **Comprehensive Evaluation** with medical-specific metrics

For detailed implementation specifics, refer to the individual instruction files in this directory.

---

**Note:** This is supplementary material for conference submission. All code is provided as-is for reproducibility and research purposes.
