# Appendix A.1: MIMIC-Eye Dataset Specifications and Processing Details

## A.1.1 MIMIC-Eye Dataset Composition

### Dataset Statistics

The MIMIC-Eye dataset represents a comprehensive multimodal medical imaging dataset integrating chest X-rays with eye-gaze fixations and radiologist transcripts. The dataset composition includes:

- **Total Records**: 3,654 medical imaging studies
- **Unique Patients**: 3,192 individual patients
- **Image Coverage**: 100% (3,654 records with chest X-ray images)
- **Report Coverage**: 100% (3,654 records with radiological reports)

### Patient Demographics and Clinical Metadata

The dataset incorporates extensive clinical metadata for each patient, including:

**Numerical Clinical Variables:**

- Age (years)
- Vital signs: Temperature (°F), Heart rate (bpm), Respiratory rate (breaths/min)
- Oxygen saturation (O2Sat) percentage
- Blood pressure: Systolic (SBP) and Diastolic (DBP) in mmHg
- Clinical acuity score

**Categorical Clinical Variables:**

- Gender classification
- Additional clinical categorical features

**Statistical Normalization:** The dataset includes pre-computed normalization parameters for clinical numerical variables based on training set statistics:

- Age: mean = 62.9 years, std = 18.5 years
- Temperature: mean = 98.1°F, std = 2.7°F
- Heart rate: mean = 86.0 bpm, std = 19.0 bpm
- Respiratory rate: mean = 18.2 breaths/min, std = 2.6 breaths/min
- O2Sat: mean = 97.9%, std = 2.6%
- SBP: mean = 133.1 mmHg, std = 25.5 mmHg
- DBP: mean = 74.0 mmHg, std = 16.4 mmHg
- Acuity: mean = 2.3, std = 0.7

### Condition Distribution

The dataset contains 8 primary medical conditions after filtering:

1. **Atelectasis**: Partial lung collapse condition
2. **Cardiomegaly**: Enlarged heart condition
3. **Edema**: Fluid accumulation in lungs
4. **Lung Opacity**: General lung opacity findings
5. **No Finding**: Normal chest X-rays without pathological findings
6. **Pleural Effusion**: Fluid in pleural space
7. **Pneumonia**: Lung infection/inflammation
8. **Support Devices**: Medical devices visible in chest X-rays

## A.1.2 Eye-Gaze Data Harmonization: REFLACX vs EyeGaze Format Alignment

### Coordinate System Unification

The MIMIC-Eye dataset integrates two distinct eye-gaze data collection paradigms requiring systematic harmonization to ensure consistent multimodal analysis:

#### REFLACX Format Specification

```
timestamp_start_fixation,timestamp_end_fixation,x_position,y_position,pupil_area_normalized,
angular_resolution_x_pixels_per_degree,angular_resolution_y_pixels_per_degree,window_width,
window_level,xmin_shown_from_image,ymin_shown_from_image,xmax_shown_from_image,
ymax_shown_from_image,xmin_in_screen_coordinates,ymin_in_screen_coordinates,
xmax_in_screen_coordinates,ymax_in_screen_coordinates
```

#### EyeGaze Format Specification

```
SESSION_ID,MEDIA_ID,DICOM_ID,CNT,Time (in secs),TIMETICK,FPOGX,FPOGY,FPOGS,FPOGD,
FPOGID,FPOGV,BPOGX,BPOGY,BPOGV,LPCX,LPCY,LPD,LPS,LPV,RPCX,RPCY,RPD,RPS,RPV,
BKID,BKDUR,BKPMIN,LPMM,LPMMV,RPMM,RPMMV,SACCADE_MAG,SACCADE_DIR,VID_FRAME,
X_ORIGINAL,Y_ORIGINAL
```

### Coordinate Normalization Implementation

**REFLACX to [0,1] Format Conversion:**

```python
# Coordinate normalization from REFLACX to [0,1] format
x_norm = (x_position - xmin_shown_from_image) / (xmax_shown_from_image - xmin_shown_from_image)
y_norm = (y_position - ymin_shown_from_image) / (ymax_shown_from_image - ymin_shown_from_image)

# Safety clipping to ensure valid range
x_norm = np.clip(x_norm, 0.0, 1.0)
y_norm = np.clip(y_norm, 0.0, 1.0)
```

**EyeGaze Format Alignment:**

```python
# EyeGaze data already provides normalized coordinates
x_norm = FPOGX  # Fixation Point of Gaze X (0-1 normalized)
y_norm = FPOGY  # Fixation Point of Gaze Y (0-1 normalized)

# Alternative: Use pixel coordinates if available
x_pixel = X_ORIGINAL  # Raw gaze coordinates in image pixel space
y_pixel = Y_ORIGINAL  # Raw gaze coordinates in image pixel space
```

### Pupil Area Normalization

**Baseline Calculation Methodology:**
The pupil area normalization employs a baseline calculation using the first 1-2 seconds of data collection to establish a subject-specific reference point:

```python
# REFLACX: Direct normalized values
pupil_area_normalized = pupil_area_normalized  # Pre-computed baseline normalization

# EyeGaze: Computed normalization
# Step 1: Filter valid pupil diameter values
valid_data = df[(df['LPV'] == 1) & (df['RPV'] == 1)]

# Step 2: Extract baseline window (first 1-2 seconds)
baseline_window = valid_data[valid_data['Time (in secs)'] <= 2.0]

# Step 3: Compute average pupil area
left_area = np.pi * (baseline_window['LPD'] / 2) ** 2
right_area = np.pi * (baseline_window['RPD'] / 2) ** 2
avg_area = (left_area + right_area) / 2
baseline_avg_area = avg_area.mean()

# Step 4: Normalize all pupil measurements
pupil_area_normalized = avg_area / baseline_avg_area
```

**Rationale for 1-2 Second Baseline:**

- **Initial Resting State**: Minimal cognitive load before active image interpretation
- **Stable Lighting Conditions**: Consistent display brightness and contrast settings
- **Task-Independent Measurements**: Excludes cognitive and emotional pupil dilation effects

### Fixation Sequence Processing

**L_MAX=100 Sequence Standardization:**

```python
# Configuration constants from preprocessing pipeline
L_MAX = 100  # Maximum number of fixations per sequence
EPS = 1e-6   # Epsilon for numerical stability

# Feature extraction and standardization
def process_fixations(fix_path, mu, sigma, out_dir):
    # Load standardized columns
    df = pd.read_csv(fix_path, usecols=["x_norm", "y_norm", "gaze_duration", "pupil_area_normalized"])
    arr = df.values.astype(np.float32)

    # Coordinate and duration clipping
    arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)  # x_norm
    arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)  # y_norm
    arr[:, 2] = np.clip(arr[:, 2], 0.0, None)  # gaze_duration
    arr[:, 3] = np.clip(arr[:, 3], 0.0, 1.0)  # pupil_area_normalized

    # Z-score standardization with epsilon guard
    arr = (arr - mu) / (sigma + EPS)

    # Sequence padding/truncation to L_MAX
    length = min(arr.shape[0], L_MAX)
    seq = np.zeros((L_MAX, 4), dtype=np.float32)
    mask = np.zeros((L_MAX,), dtype=np.bool_)
    seq[:length] = arr[:length]
    mask[:length] = True

    return seq, mask
```

**Temporal Alignment Strategies:**

- **Silent Phase Processing**: Fixations before radiologist begins verbal reporting
- **Reporting Phase Processing**: Fixations during active radiological interpretation
- **Complete Sequence Processing**: Full temporal sequence regardless of speech activity

### Screen-to-Image Coordinate Mapping

**Angular Resolution Conversion:**
The dataset incorporates angular resolution metadata for precise visual angle calculations:

```python
# Angular resolution parameters from REFLACX
angular_resolution_x = angular_resolution_x_pixels_per_degree
angular_resolution_y = angular_resolution_y_pixels_per_degree

# Visual angle calculation
visual_angle_x = pixel_distance_x / angular_resolution_x
visual_angle_y = pixel_distance_y / angular_resolution_y
```

**Screen Space to Image Space Transformation:**

```python
# Bounding box mapping for REFLACX data
screen_to_image_x = (screen_x - xmin_in_screen_coordinates) / (xmax_in_screen_coordinates - xmin_in_screen_coordinates)
screen_to_image_y = (screen_y - ymin_in_screen_coordinates) / (ymax_in_screen_coordinates - ymin_in_screen_coordinates)

# Map to actual image pixel coordinates
image_x = xmin_shown_from_image + screen_to_image_x * (xmax_shown_from_image - xmin_shown_from_image)
image_y = ymin_shown_from_image + screen_to_image_y * (ymax_shown_from_image - ymin_shown_from_image)
```

### Unified Dataset Schema

**Common Data Structure:**

```python
# Harmonized eye-gaze record format
harmonized_record = {
    'source': ['REFLACX', 'EyeGaze'],
    'timestamp_start': float,        # Start time in seconds
    'timestamp_end': float,          # End time in seconds
    'duration': float,               # Fixation duration
    'x_norm': float,                 # Normalized X coordinate [0,1]
    'y_norm': float,                 # Normalized Y coordinate [0,1]
    'pupil_area_normalized': float,  # Baseline-normalized pupil area
    'dicom_id': str,                 # Study identifier
    'validity_flags': dict           # Data quality indicators
}
```

**Statistical Harmonization Parameters:**

- **Global Mean (μ)**: Computed across all fixation sequences for z-score normalization
- **Global Standard Deviation (σ)**: Population-level variance for standardization
- **Missing Data Handling**: Zero-padding with validity masks for incomplete sequences
- **Outlier Management**: Statistical clipping using 3-sigma thresholds

### Heatmap Generation Pipeline

**Gaussian Kernel Convolution:**

```python
def get_heatmap(fixations_dict, display_size, pupil_weighting=False):
    # Gaussian kernel parameters
    gwh = 200  # Gaussian window height/width
    gsdwh = gwh/6  # Gaussian standard deviation
    gaus = gaussian(gwh, gsdwh)

    # Initialize heatmap matrix
    strt = int(gwh/2)
    heatmapsize = (int(display_size[1] + 2*strt), int(display_size[0] + 2*strt))
    heatmap = np.zeros(heatmapsize, dtype=float)

    # Accumulate Gaussian contributions
    for i in range(len(fixations_dict['dur'])):
        x = strt + int(fixations_dict['x'][i]) - int(gwh/2)
        y = strt + int(fixations_dict['y'][i]) - int(gwh/2)

        if pupil_weighting:
            weight = fixations_dict['pupil'][i]
        else:
            weight = fixations_dict['dur'][i]

        heatmap[y:y+gwh, x:x+gwh] += gaus * weight

    return heatmap[strt:display_size[1]+strt, strt:display_size[0]+strt]
```

This comprehensive harmonization framework ensures consistent multimodal integration while preserving the unique characteristics and temporal dynamics of both REFLACX and EyeGaze data collection paradigms.

## A.4 Evaluation Metrics and Validation

### A.4.1 Disease Classification Metrics

The MIMIC-Eye dataset evaluation framework employs a comprehensive suite of metrics specifically designed for multi-label medical image classification. The evaluation targets 8 primary pathological conditions after systematic filtering, ensuring robust performance assessment across diverse clinical presentations.

#### Target Medical Conditions

The evaluation framework focuses on the following clinically significant conditions:

```python
# Primary evaluation conditions for MIMIC-Eye classification
EVALUATION_CONDITIONS = [
    'Atelectasis',           # Lung collapse or incomplete expansion
    'Cardiomegaly',         # Enlarged heart
    'Edema',                # Fluid accumulation in tissues
    'Lung Opacity',         # Abnormal lung opacification
    'No Finding',           # Normal chest X-ray
    'Pleural Effusion',     # Fluid in pleural space
    'Pneumonia',            # Lung infection and inflammation
    'Support Devices'       # Medical device presence
]
```

#### Per-Condition Performance Metrics

Individual condition evaluation provides granular insight into model performance across pathological presentations:

```python
def compute_per_condition_metrics(labels, predictions, probabilities, conditions):
    """
    Calculate comprehensive per-condition classification metrics.

    Args:
        labels: Ground truth binary labels [N, num_conditions]
        predictions: Binary predictions [N, num_conditions]
        probabilities: Prediction probabilities [N, num_conditions]
        conditions: List of condition names

    Returns:
        Dict containing per-condition metrics
    """
    per_condition_metrics = {}

    for i, condition in enumerate(conditions):
        condition_labels = labels[:, i]
        condition_preds = predictions[:, i]
        condition_probs = probabilities[:, i]

        # Core classification metrics
        precision = precision_score(condition_labels, condition_preds, zero_division=0)
        recall = recall_score(condition_labels, condition_preds, zero_division=0)
        f1 = f1_score(condition_labels, condition_preds, zero_division=0)
        support = np.sum(condition_labels)

        # Area Under ROC Curve (handling edge cases)
        try:
            if len(np.unique(condition_labels)) > 1:
                auc = roc_auc_score(condition_labels, condition_probs)
            else:
                auc = np.nan  # Single class present
        except ValueError:
            auc = np.nan  # Insufficient data

        # Per-condition accuracy
        accuracy = np.mean(condition_labels == condition_preds)

        # True/False Positive/Negative rates
        tn, fp, fn, tp = confusion_matrix(condition_labels, condition_preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        per_condition_metrics[condition] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'accuracy': accuracy,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'support': int(support),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }

    return per_condition_metrics
```

#### Macro-Averaged Metrics

System-level performance assessment using macro-averaging across all conditions:

```python
def compute_macro_averaged_metrics(labels, predictions, probabilities):
    """
    Calculate macro-averaged metrics across all conditions.
    Provides unweighted average performance across pathological classes.
    """
    # Macro-averaged core metrics
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    macro_recall = recall_score(labels, predictions, average="macro", zero_division=0)
    macro_precision = precision_score(labels, predictions, average="macro", zero_division=0)

    # Macro-averaged AUC calculation
    auc_scores = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            try:
                auc = roc_auc_score(labels[:, i], probabilities[:, i])
                auc_scores.append(auc)
            except ValueError:
                continue  # Skip problematic classes

    macro_auc = np.mean(auc_scores) if auc_scores else 0.0

    # Additional macro metrics
    macro_accuracy = np.mean([
        np.mean(labels[:, i] == predictions[:, i])
        for i in range(labels.shape[1])
    ])

    # Hamming accuracy (exact match ratio)
    hamming_accuracy = np.mean(np.all(labels == predictions, axis=1))

    return {
        'macro_f1': macro_f1,
        'macro_recall': macro_recall,
        'macro_precision': macro_precision,
        'macro_auc': macro_auc,
        'macro_accuracy': macro_accuracy,
        'hamming_accuracy': hamming_accuracy,
        'exact_match_ratio': hamming_accuracy
    }
```

#### Confusion Matrix Analysis

Detailed classification error pattern analysis for systematic performance understanding:

```python
def generate_confusion_matrix_analysis(labels, predictions, conditions):
    """
    Generate comprehensive confusion matrix analysis for multi-label classification.
    Provides detailed error pattern insights for each medical condition.
    """
    confusion_analysis = {}

    for i, condition in enumerate(conditions):
        condition_labels = labels[:, i]
        condition_preds = predictions[:, i]

        # Generate confusion matrix
        cm = confusion_matrix(condition_labels, condition_preds)
        tn, fp, fn, tp = cm.ravel()

        # Calculate error rates and diagnostic performance
        total_samples = len(condition_labels)
        positive_samples = np.sum(condition_labels)
        negative_samples = total_samples - positive_samples

        # Error analysis
        false_positive_rate = fp / negative_samples if negative_samples > 0 else 0.0
        false_negative_rate = fn / positive_samples if positive_samples > 0 else 0.0
        true_positive_rate = tp / positive_samples if positive_samples > 0 else 0.0
        true_negative_rate = tn / negative_samples if negative_samples > 0 else 0.0

        # Clinical interpretation metrics
        positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Diagnostic likelihood ratios
        likelihood_ratio_positive = (true_positive_rate / false_positive_rate
                                   if false_positive_rate > 0 else float('inf'))
        likelihood_ratio_negative = (false_negative_rate / true_negative_rate
                                   if true_negative_rate > 0 else 0.0)

        confusion_analysis[condition] = {
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'true_positive_rate': true_positive_rate,
            'true_negative_rate': true_negative_rate,
            'positive_predictive_value': positive_predictive_value,
            'negative_predictive_value': negative_predictive_value,
            'likelihood_ratio_positive': likelihood_ratio_positive,
            'likelihood_ratio_negative': likelihood_ratio_negative,
            'prevalence': positive_samples / total_samples
        }

    return confusion_analysis
```

#### Statistical Significance Testing

Rigorous statistical assessment for model comparison and performance validation:

```python
def perform_statistical_significance_testing(baseline_results, comparison_results,
                                           baseline_predictions, comparison_predictions,
                                           true_labels):
    """
    Comprehensive statistical significance testing for model comparison.
    Implements multiple statistical tests for robust performance assessment.
    """
    from scipy import stats
    from statsmodels.stats.contingency_tables import mcnemar
    import numpy as np

    statistical_tests = {}

    # 1. McNemar's Test for paired binary classification comparison
    mcnemar_results = {}
    for i, condition in enumerate(CONDITIONS):
        condition_labels = true_labels[:, i]
        baseline_preds = baseline_predictions[:, i]
        comparison_preds = comparison_predictions[:, i]

        # Create contingency table for McNemar's test
        baseline_correct = (baseline_preds == condition_labels)
        comparison_correct = (comparison_preds == condition_labels)

        # 2x2 contingency table
        both_correct = np.sum(baseline_correct & comparison_correct)
        baseline_only = np.sum(baseline_correct & ~comparison_correct)
        comparison_only = np.sum(~baseline_correct & comparison_correct)
        both_wrong = np.sum(~baseline_correct & ~comparison_correct)

        contingency_table = np.array([[both_correct, baseline_only],
                                    [comparison_only, both_wrong]])

        # Perform McNemar's test
        try:
            mcnemar_result = mcnemar(contingency_table, exact=True)
            mcnemar_results[condition] = {
                'statistic': float(mcnemar_result.statistic),
                'p_value': float(mcnemar_result.pvalue),
                'significant': mcnemar_result.pvalue < 0.05,
                'contingency_table': contingency_table.tolist()
            }
        except Exception as e:
            mcnemar_results[condition] = {
                'error': str(e),
                'contingency_table': contingency_table.tolist()
            }

    # 2. Paired t-test for metric differences
    paired_t_tests = {}
    for metric in ['f1_score', 'precision', 'recall', 'auc_roc']:
        baseline_values = [baseline_results[condition][metric]
                          for condition in CONDITIONS
                          if not np.isnan(baseline_results[condition][metric])]
        comparison_values = [comparison_results[condition][metric]
                           for condition in CONDITIONS
                           if not np.isnan(comparison_results[condition][metric])]

        if len(baseline_values) == len(comparison_values) and len(baseline_values) > 1:
            t_stat, p_value = stats.ttest_rel(comparison_values, baseline_values)

            # Effect size (Cohen's d)
            differences = np.array(comparison_values) - np.array(baseline_values)
            pooled_std = np.sqrt((np.var(baseline_values) + np.var(comparison_values)) / 2)
            cohens_d = np.mean(differences) / pooled_std if pooled_std > 0 else 0.0

            paired_t_tests[metric] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'effect_size_cohens_d': float(cohens_d),
                'mean_difference': float(np.mean(differences)),
                'std_difference': float(np.std(differences))
            }

    # 3. Wilcoxon signed-rank test (non-parametric alternative)
    wilcoxon_tests = {}
    for metric in ['f1_score', 'precision', 'recall', 'auc_roc']:
        baseline_values = [baseline_results[condition][metric]
                          for condition in CONDITIONS
                          if not np.isnan(baseline_results[condition][metric])]
        comparison_values = [comparison_results[condition][metric]
                           for condition in CONDITIONS
                           if not np.isnan(comparison_results[condition][metric])]

        if len(baseline_values) == len(comparison_values) and len(baseline_values) > 1:
            try:
                w_stat, p_value = stats.wilcoxon(comparison_values, baseline_values)
                wilcoxon_tests[metric] = {
                    'statistic': float(w_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            except ValueError as e:
                wilcoxon_tests[metric] = {'error': str(e)}

    return {
        'mcnemar_tests': mcnemar_results,
        'paired_t_tests': paired_t_tests,
        'wilcoxon_tests': wilcoxon_tests
    }
```

#### Performance Reporting Framework

Standardized reporting system for comprehensive evaluation documentation:

```python
def generate_evaluation_report(per_condition_metrics, macro_metrics,
                             confusion_analysis, statistical_tests,
                             model_name="Multimodal Model"):
    """
    Generate comprehensive evaluation report with tabulated results.
    Creates publication-ready performance summaries.
    """
    from tabulate import tabulate

    # Per-condition performance table
    headers = ["Condition", "Precision", "Recall", "F1-score", "AUC-ROC", "Support"]
    rows = []

    for condition, metrics in per_condition_metrics.items():
        auc_str = f"{metrics['auc_roc']:.3f}" if not np.isnan(metrics['auc_roc']) else "N/A"
        rows.append([
            condition,
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1_score']:.3f}",
            auc_str,
            f"{metrics['support']}"
        ])

    # Add macro-averaged row
    rows.append([
        "Macro Average",
        f"{macro_metrics['macro_precision']:.3f}",
        f"{macro_metrics['macro_recall']:.3f}",
        f"{macro_metrics['macro_f1']:.3f}",
        f"{macro_metrics['macro_auc']:.3f}",
        "All Conditions"
    ])

    performance_table = tabulate(rows, headers=headers, tablefmt="grid")

    # Statistical significance summary
    significance_summary = {
        'mcnemar_significant_conditions': [
            condition for condition, test in statistical_tests['mcnemar_tests'].items()
            if test.get('significant', False)
        ],
        'significant_metric_improvements': [
            metric for metric, test in statistical_tests['paired_t_tests'].items()
            if test.get('significant', False) and test.get('mean_difference', 0) > 0
        ]
    }

    return {
        'model_name': model_name,
        'performance_table': performance_table,
        'macro_metrics': macro_metrics,
        'per_condition_metrics': per_condition_metrics,
        'confusion_analysis': confusion_analysis,
        'statistical_significance': statistical_tests,
        'significance_summary': significance_summary,
        'evaluation_timestamp': datetime.now().isoformat()
    }
```

This comprehensive evaluation framework ensures rigorous assessment of multimodal medical AI models, providing both clinical interpretability and statistical rigor necessary for medical imaging applications. The multi-faceted approach enables detailed understanding of model performance across diverse pathological presentations while maintaining statistical validity for comparative analysis.

### A.4.2 Report Generation Evaluation

The MIMIC-Eye system employs sophisticated natural language evaluation metrics specifically tailored for medical report generation. The evaluation framework addresses both lexical similarity and semantic coherence while incorporating domain-specific medical knowledge validation.

#### BLEU Scores: N-gram Precision Analysis

Bilingual Evaluation Understudy (BLEU) scores measure n-gram precision between generated and reference reports:

```python
def calculate_bleu_scores(generated_reports, reference_reports):
    """
    Calculate comprehensive BLEU scores for medical report evaluation.
    Implements BLEU-1 through BLEU-4 with medical text optimization.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk

    bleu_metrics = {
        'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': [],
        'corpus_bleu': {'bleu_1': 0, 'bleu_2': 0, 'bleu_3': 0, 'bleu_4': 0}
    }

    smooth_fn = SmoothingFunction().method1  # Add-one smoothing for medical texts

    # Sentence-level BLEU calculation
    for gen_report, ref_report in zip(generated_reports, reference_reports):
        if not gen_report.strip() or not ref_report.strip():
            continue

        # Tokenize with medical text preprocessing
        reference_tokens = nltk.word_tokenize(ref_report.lower())
        hypothesis_tokens = nltk.word_tokenize(gen_report.lower())

        # Calculate BLEU-1 through BLEU-4
        try:
            bleu_1 = sentence_bleu([reference_tokens], hypothesis_tokens,
                                 weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
            bleu_2 = sentence_bleu([reference_tokens], hypothesis_tokens,
                                 weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
            bleu_3 = sentence_bleu([reference_tokens], hypothesis_tokens,
                                 weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
            bleu_4 = sentence_bleu([reference_tokens], hypothesis_tokens,
                                 weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)

            bleu_metrics['bleu_1'].append(bleu_1)
            bleu_metrics['bleu_2'].append(bleu_2)
            bleu_metrics['bleu_3'].append(bleu_3)
            bleu_metrics['bleu_4'].append(bleu_4)

        except Exception as e:
            logging.warning(f"BLEU calculation error: {e}")
            continue

    # Calculate corpus-level averages
    for metric in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']:
        if bleu_metrics[metric]:
            bleu_metrics['corpus_bleu'][metric] = np.mean(bleu_metrics[metric])

    return bleu_metrics
```

#### ROUGE Metrics: Recall-Oriented Evaluation

Recall-Oriented Understudy for Gisting Evaluation (ROUGE) emphasizes content coverage and recall:

```python
def calculate_rouge_metrics(generated_reports, reference_reports):
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L metrics for medical reports.
    Focuses on recall-oriented evaluation with medical text considerations.
    """
    from rouge_score import rouge_scorer

    rouge_metrics = {
        'rouge_1': {'precision': [], 'recall': [], 'f1': []},
        'rouge_2': {'precision': [], 'recall': [], 'f1': []},
        'rouge_l': {'precision': [], 'recall': [], 'f1': []}
    }

    # Initialize ROUGE scorer with stemming for medical terminology
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for gen_report, ref_report in zip(generated_reports, reference_reports):
        if not gen_report.strip() or not ref_report.strip():
            continue

        try:
            scores = scorer.score(ref_report, gen_report)

            # ROUGE-1 (unigram overlap)
            rouge_metrics['rouge_1']['precision'].append(scores['rouge1'].precision)
            rouge_metrics['rouge_1']['recall'].append(scores['rouge1'].recall)
            rouge_metrics['rouge_1']['f1'].append(scores['rouge1'].fmeasure)

            # ROUGE-2 (bigram overlap)
            rouge_metrics['rouge_2']['precision'].append(scores['rouge2'].precision)
            rouge_metrics['rouge_2']['recall'].append(scores['rouge2'].recall)
            rouge_metrics['rouge_2']['f1'].append(scores['rouge2'].fmeasure)

            # ROUGE-L (longest common subsequence)
            rouge_metrics['rouge_l']['precision'].append(scores['rougeL'].precision)
            rouge_metrics['rouge_l']['recall'].append(scores['rougeL'].recall)
            rouge_metrics['rouge_l']['f1'].append(scores['rougeL'].fmeasure)

        except Exception as e:
            logging.warning(f"ROUGE calculation error: {e}")
            continue

    # Calculate corpus-level averages
    final_rouge_metrics = {}
    for metric_type in ['rouge_1', 'rouge_2', 'rouge_l']:
        final_rouge_metrics[metric_type] = {}
        for score_type in ['precision', 'recall', 'f1']:
            scores = rouge_metrics[metric_type][score_type]
            final_rouge_metrics[metric_type][score_type] = np.mean(scores) if scores else 0.0

    return final_rouge_metrics
```

#### Clinical Keyword Overlap: Medical Terminology Alignment

Domain-specific evaluation measuring alignment of medical terminology between generated and reference reports:

```python
def calculate_clinical_keyword_overlap(generated_reports, reference_reports,
                                     condition_keywords, medical_vocabulary):
    """
    Calculate clinical keyword overlap scoring for medical terminology alignment.
    Measures how well generated reports use appropriate medical vocabulary.
    """

    clinical_metrics = {
        'exact_medical_terms': [],
        'condition_specific_keywords': [],
        'anatomical_terminology': [],
        'clinical_impression_terms': [],
        'overall_medical_vocabulary_score': []
    }

    for gen_report, ref_report in zip(generated_reports, reference_reports):
        # Extract medical terms from both reports
        gen_medical_terms = extract_medical_terminology(gen_report, medical_vocabulary)
        ref_medical_terms = extract_medical_terminology(ref_report, medical_vocabulary)

        # Exact medical term overlap
        exact_overlap = len(gen_medical_terms & ref_medical_terms)
        total_ref_terms = len(ref_medical_terms)
        exact_score = exact_overlap / total_ref_terms if total_ref_terms > 0 else 0.0

        # Condition-specific keyword scoring
        condition_score = calculate_condition_keyword_alignment(
            gen_report, ref_report, condition_keywords
        )

        # Anatomical terminology assessment
        anatomical_score = evaluate_anatomical_terminology_usage(
            gen_report, ref_report
        )

        # Clinical impression terminology
        impression_score = assess_clinical_impression_alignment(
            gen_report, ref_report
        )

        # Overall medical vocabulary coherence
        vocab_coherence = calculate_medical_vocabulary_coherence(
            gen_report, medical_vocabulary
        )

        clinical_metrics['exact_medical_terms'].append(exact_score)
        clinical_metrics['condition_specific_keywords'].append(condition_score)
        clinical_metrics['anatomical_terminology'].append(anatomical_score)
        clinical_metrics['clinical_impression_terms'].append(impression_score)
        clinical_metrics['overall_medical_vocabulary_score'].append(vocab_coherence)

    # Calculate aggregated scores
    final_clinical_metrics = {}
    for metric_name, scores in clinical_metrics.items():
        final_clinical_metrics[metric_name] = {
            'mean': np.mean(scores) if scores else 0.0,
            'std': np.std(scores) if scores else 0.0,
            'median': np.median(scores) if scores else 0.0
        }

    return final_clinical_metrics

def extract_medical_terminology(text, medical_vocabulary):
    """Extract validated medical terms from text using comprehensive vocabulary."""
    import re

    # Tokenize and normalize text
    tokens = re.findall(r'\b\w+\b', text.lower())

    # Identify medical terms
    medical_terms = set()
    for token in tokens:
        if token in medical_vocabulary['conditions']:
            medical_terms.add(token)
        elif token in medical_vocabulary['anatomical_terms']:
            medical_terms.add(token)
        elif token in medical_vocabulary['clinical_descriptors']:
            medical_terms.add(token)
        elif token in medical_vocabulary['diagnostic_terms']:
            medical_terms.add(token)

    return medical_terms

def calculate_condition_keyword_alignment(gen_report, ref_report, condition_keywords):
    """Calculate alignment score for condition-specific keywords."""
    alignment_scores = []

    for condition, keywords in condition_keywords.items():
        gen_condition_terms = extract_condition_keywords(gen_report, keywords)
        ref_condition_terms = extract_condition_keywords(ref_report, keywords)

        if ref_condition_terms:  # Only evaluate if reference contains condition keywords
            overlap = len(gen_condition_terms & ref_condition_terms)
            alignment_score = overlap / len(ref_condition_terms)
            alignment_scores.append(alignment_score)

    return np.mean(alignment_scores) if alignment_scores else 0.0
```

#### Semantic Similarity: Sentence-BERT Embeddings

Advanced semantic coherence evaluation using contextual embeddings:

```python
def calculate_semantic_similarity(generated_reports, reference_reports,
                                model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Calculate semantic similarity using Sentence-BERT embeddings.
    Measures deep semantic coherence beyond lexical overlap.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Load pre-trained Sentence-BERT model
    model = SentenceTransformer(model_name)

    semantic_metrics = {
        'sentence_bert_similarity': [],
        'section_wise_similarity': {'findings': [], 'impression': []},
        'semantic_coherence_score': [],
        'contextual_embedding_distance': []
    }

    for gen_report, ref_report in zip(generated_reports, reference_reports):
        if not gen_report.strip() or not ref_report.strip():
            continue

        # Overall report similarity
        gen_embedding = model.encode([gen_report])
        ref_embedding = model.encode([ref_report])

        similarity = cosine_similarity(gen_embedding, ref_embedding)[0][0]
        semantic_metrics['sentence_bert_similarity'].append(similarity)

        # Section-wise similarity analysis
        gen_sections = extract_report_sections(gen_report)
        ref_sections = extract_report_sections(ref_report)

        for section in ['findings', 'impression']:
            if section in gen_sections and section in ref_sections:
                gen_sec_emb = model.encode([gen_sections[section]])
                ref_sec_emb = model.encode([ref_sections[section]])

                section_sim = cosine_similarity(gen_sec_emb, ref_sec_emb)[0][0]
                semantic_metrics['section_wise_similarity'][section].append(section_sim)

        # Semantic coherence within generated report
        coherence_score = calculate_internal_semantic_coherence(gen_report, model)
        semantic_metrics['semantic_coherence_score'].append(coherence_score)

        # Contextual embedding distance (L2 norm)
        embedding_distance = np.linalg.norm(gen_embedding - ref_embedding)
        semantic_metrics['contextual_embedding_distance'].append(embedding_distance)

    # Calculate aggregated semantic metrics
    final_semantic_metrics = {}
    for metric_name, scores in semantic_metrics.items():
        if isinstance(scores, dict):  # Handle section-wise metrics
            final_semantic_metrics[metric_name] = {}
            for section, section_scores in scores.items():
                final_semantic_metrics[metric_name][section] = {
                    'mean': np.mean(section_scores) if section_scores else 0.0,
                    'std': np.std(section_scores) if section_scores else 0.0
                }
        else:
            final_semantic_metrics[metric_name] = {
                'mean': np.mean(scores) if scores else 0.0,
                'std': np.std(scores) if scores else 0.0,
                'median': np.median(scores) if scores else 0.0
            }

    return final_semantic_metrics

def extract_report_sections(report_text):
    """Extract FINDINGS and IMPRESSION sections from medical reports."""
    import re

    sections = {}

    # Extract FINDINGS section
    findings_match = re.search(r'FINDINGS:\s*(.*?)(?=IMPRESSION:|$)',
                              report_text, re.IGNORECASE | re.DOTALL)
    if findings_match:
        sections['findings'] = findings_match.group(1).strip()

    # Extract IMPRESSION section
    impression_match = re.search(r'IMPRESSION:\s*(.*?)$',
                                report_text, re.IGNORECASE | re.DOTALL)
    if impression_match:
        sections['impression'] = impression_match.group(1).strip()

    return sections

def calculate_internal_semantic_coherence(text, model):
    """Calculate internal semantic coherence within a single report."""
    sentences = text.split('.')
    if len(sentences) < 2:
        return 1.0  # Single sentence is perfectly coherent

    # Get embeddings for all sentences
    sentence_embeddings = model.encode(sentences)

    # Calculate pairwise similarities
    similarities = []
    for i in range(len(sentences) - 1):
        sim = cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i+1]])[0][0]
        similarities.append(sim)

    return np.mean(similarities) if similarities else 0.0
```

### A.4.3 Attention Analysis Validation

The attention validation framework provides comprehensive assessment of spatial attention alignment between human radiologist gaze patterns and AI model attention mechanisms, ensuring clinically relevant focus areas are properly learned.

#### Spatial Correlation: Pearson Correlation Analysis

Quantitative measurement of spatial attention alignment using statistical correlation:

```python
def calculate_spatial_correlation_metrics(human_gaze_heatmaps, model_attention_maps):
    """
    Calculate comprehensive spatial correlation metrics between human gaze and AI attention.
    Implements multiple correlation measures for robust spatial alignment assessment.
    """
    from scipy.stats import pearsonr, spearmanr
    from scipy.spatial.distance import jensenshannon
    import cv2

    correlation_metrics = {
        'pearson_correlation': [],
        'spearman_correlation': [],
        'spatial_overlap_scores': [],
        'attention_centroid_distances': [],
        'top_region_alignment': []
    }

    for human_heatmap, model_attention in zip(human_gaze_heatmaps, model_attention_maps):
        # Ensure spatial alignment
        if human_heatmap.shape != model_attention.shape:
            model_attention = cv2.resize(model_attention,
                                       (human_heatmap.shape[1], human_heatmap.shape[0]))

        # Flatten for correlation calculations
        human_flat = human_heatmap.flatten()
        model_flat = model_attention.flatten()

        # Pearson correlation (linear relationship)
        pearson_corr, pearson_p = pearsonr(human_flat, model_flat)
        correlation_metrics['pearson_correlation'].append({
            'coefficient': pearson_corr,
            'p_value': pearson_p,
            'significant': pearson_p < 0.05
        })

        # Spearman correlation (monotonic relationship)
        spearman_corr, spearman_p = spearmanr(human_flat, model_flat)
        correlation_metrics['spearman_correlation'].append({
            'coefficient': spearman_corr,
            'p_value': spearman_p,
            'significant': spearman_p < 0.05
        })

        # Spatial overlap using intersection over union
        overlap_score = calculate_spatial_overlap(human_heatmap, model_attention)
        correlation_metrics['spatial_overlap_scores'].append(overlap_score)

        # Attention centroid distance
        centroid_distance = calculate_attention_centroid_distance(
            human_heatmap, model_attention
        )
        correlation_metrics['attention_centroid_distances'].append(centroid_distance)

        # Top attention region alignment
        top_alignment = calculate_top_region_alignment(
            human_heatmap, model_attention, top_percentile=0.2
        )
        correlation_metrics['top_region_alignment'].append(top_alignment)

    # Aggregate correlation statistics
    aggregated_metrics = {}
    for metric_name, metric_values in correlation_metrics.items():
        if isinstance(metric_values[0], dict):  # Handle correlation with p-values
            coefficients = [item['coefficient'] for item in metric_values]
            p_values = [item['p_value'] for item in metric_values]
            significant_count = sum([item['significant'] for item in metric_values])

            aggregated_metrics[metric_name] = {
                'mean_coefficient': np.mean(coefficients),
                'std_coefficient': np.std(coefficients),
                'median_coefficient': np.median(coefficients),
                'mean_p_value': np.mean(p_values),
                'significant_proportion': significant_count / len(metric_values)
            }
        else:
            aggregated_metrics[metric_name] = {
                'mean': np.mean(metric_values),
                'std': np.std(metric_values),
                'median': np.median(metric_values),
                'min': np.min(metric_values),
                'max': np.max(metric_values)
            }

    return aggregated_metrics

def calculate_spatial_overlap(heatmap1, heatmap2, threshold=0.5):
    """Calculate spatial overlap between two attention heatmaps."""
    # Binarize heatmaps using threshold
    binary1 = (heatmap1 > threshold).astype(float)
    binary2 = (heatmap2 > threshold).astype(float)

    # Calculate intersection over union
    intersection = np.sum(binary1 * binary2)
    union = np.sum(np.maximum(binary1, binary2))

    return intersection / union if union > 0 else 0.0

def calculate_attention_centroid_distance(heatmap1, heatmap2):
    """Calculate Euclidean distance between attention centroids."""
    def compute_centroid(heatmap):
        h, w = heatmap.shape
        total_weight = np.sum(heatmap)

        if total_weight == 0:
            return (h // 2, w // 2)  # Default to center

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        centroid_y = np.sum(y_coords * heatmap) / total_weight
        centroid_x = np.sum(x_coords * heatmap) / total_weight

        return (centroid_y, centroid_x)

    centroid1 = compute_centroid(heatmap1)
    centroid2 = compute_centroid(heatmap2)

    # Normalized distance (0-1 scale)
    max_distance = np.sqrt(heatmap1.shape[0]**2 + heatmap1.shape[1]**2)
    distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)

    return distance / max_distance

def calculate_top_region_alignment(heatmap1, heatmap2, top_percentile=0.2):
    """Calculate alignment of top attention regions."""
    # Get top-k% regions for both heatmaps
    threshold1 = np.percentile(heatmap1, (1 - top_percentile) * 100)
    threshold2 = np.percentile(heatmap2, (1 - top_percentile) * 100)

    top_regions1 = (heatmap1 >= threshold1).astype(float)
    top_regions2 = (heatmap2 >= threshold2).astype(float)

    # Calculate Jaccard similarity for top regions
    intersection = np.sum(top_regions1 * top_regions2)
    union = np.sum(np.maximum(top_regions1, top_regions2))

    return intersection / union if union > 0 else 0.0
```

#### Distribution Similarity: Jensen-Shannon Divergence

Statistical measurement of attention distribution similarity:

```python
def calculate_distribution_similarity_metrics(human_gaze_heatmaps, model_attention_maps):
    """
    Calculate distribution similarity using Jensen-Shannon divergence and related metrics.
    Measures how similarly attention is distributed across spatial regions.
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import entropy, wasserstein_distance

    distribution_metrics = {
        'jensen_shannon_divergence': [],
        'kl_divergence_human_to_model': [],
        'kl_divergence_model_to_human': [],
        'wasserstein_distance': [],
        'bhattacharyya_distance': [],
        'attention_entropy_comparison': []
    }

    for human_heatmap, model_attention in zip(human_gaze_heatmaps, model_attention_maps):
        # Ensure same dimensions and normalize as probability distributions
        if human_heatmap.shape != model_attention.shape:
            model_attention = cv2.resize(model_attention,
                                       (human_heatmap.shape[1], human_heatmap.shape[0]))

        # Convert to probability distributions
        epsilon = 1e-8
        human_flat = human_heatmap.flatten()
        model_flat = model_attention.flatten()

        # Normalize to probability distributions
        human_prob = (human_flat + epsilon) / (human_flat.sum() + epsilon * len(human_flat))
        model_prob = (model_flat + epsilon) / (model_flat.sum() + epsilon * len(model_flat))

        # Jensen-Shannon divergence
        js_distance = jensenshannon(human_prob, model_prob)
        distribution_metrics['jensen_shannon_divergence'].append(js_distance)

        # KL divergences (both directions)
        kl_human_to_model = entropy(human_prob, model_prob)
        kl_model_to_human = entropy(model_prob, human_prob)
        distribution_metrics['kl_divergence_human_to_model'].append(kl_human_to_model)
        distribution_metrics['kl_divergence_model_to_human'].append(kl_model_to_human)

        # Wasserstein distance (Earth Mover's Distance)
        wasserstein_dist = wasserstein_distance(human_flat, model_flat)
        distribution_metrics['wasserstein_distance'].append(wasserstein_dist)

        # Bhattacharyya distance
        bhattacharyya_dist = calculate_bhattacharyya_distance(human_prob, model_prob)
        distribution_metrics['bhattacharyya_distance'].append(bhattacharyya_dist)

        # Entropy comparison
        human_entropy = entropy(human_prob)
        model_entropy = entropy(model_prob)
        entropy_comparison = {
            'human_entropy': human_entropy,
            'model_entropy': model_entropy,
            'entropy_ratio': model_entropy / human_entropy if human_entropy > 0 else 0.0,
            'entropy_difference': abs(human_entropy - model_entropy)
        }
        distribution_metrics['attention_entropy_comparison'].append(entropy_comparison)

    # Aggregate distribution metrics
    aggregated_metrics = {}
    for metric_name, metric_values in distribution_metrics.items():
        if metric_name == 'attention_entropy_comparison':
            # Handle entropy comparison separately
            human_entropies = [item['human_entropy'] for item in metric_values]
            model_entropies = [item['model_entropy'] for item in metric_values]
            entropy_ratios = [item['entropy_ratio'] for item in metric_values]
            entropy_diffs = [item['entropy_difference'] for item in metric_values]

            aggregated_metrics[metric_name] = {
                'mean_human_entropy': np.mean(human_entropies),
                'mean_model_entropy': np.mean(model_entropies),
                'mean_entropy_ratio': np.mean(entropy_ratios),
                'mean_entropy_difference': np.mean(entropy_diffs),
                'std_entropy_difference': np.std(entropy_diffs)
            }
        else:
            aggregated_metrics[metric_name] = {
                'mean': np.mean(metric_values),
                'std': np.std(metric_values),
                'median': np.median(metric_values),
                'min': np.min(metric_values),
                'max': np.max(metric_values)
            }

    return aggregated_metrics

def calculate_bhattacharyya_distance(p, q):
    """Calculate Bhattacharyya distance between two probability distributions."""
    bhattacharyya_coefficient = np.sum(np.sqrt(p * q))
    return -np.log(bhattacharyya_coefficient) if bhattacharyya_coefficient > 0 else float('inf')
```

#### Regional Analysis: Anatomical Region-Specific Attention

Detailed analysis of attention patterns within clinically relevant anatomical regions:

```python
def calculate_regional_attention_analysis(human_gaze_heatmaps, model_attention_maps,
                                        anatomical_regions, region_masks):
    """
    Calculate region-specific attention analysis for anatomical structures.
    Evaluates attention alignment within clinically relevant areas.
    """

    regional_metrics = {
        'region_wise_correlations': {},
        'region_attention_ratios': {},
        'clinical_focus_alignment': {},
        'pathology_specific_attention': {}
    }

    for region_name, region_mask in region_masks.items():
        region_correlations = []
        region_attention_ratios = []

        for human_heatmap, model_attention in zip(human_gaze_heatmaps, model_attention_maps):
            # Ensure spatial alignment
            if human_heatmap.shape != model_attention.shape:
                model_attention = cv2.resize(model_attention,
                                           (human_heatmap.shape[1], human_heatmap.shape[0]))

            if region_mask.shape != human_heatmap.shape:
                current_region_mask = cv2.resize(region_mask.astype(float),
                                                (human_heatmap.shape[1], human_heatmap.shape[0]))
                current_region_mask = (current_region_mask > 0.5).astype(float)
            else:
                current_region_mask = region_mask

            # Extract region-specific attention values
            human_region = human_heatmap * current_region_mask
            model_region = model_attention * current_region_mask

            # Calculate region-specific correlation
            if np.sum(current_region_mask) > 0:  # Ensure region exists
                human_region_flat = human_region[current_region_mask > 0]
                model_region_flat = model_region[current_region_mask > 0]

                if len(human_region_flat) > 1 and np.var(human_region_flat) > 0:
                    region_corr, _ = pearsonr(human_region_flat, model_region_flat)
                    region_correlations.append(region_corr if not np.isnan(region_corr) else 0.0)

                # Calculate attention ratio (region attention / total attention)
                human_ratio = np.sum(human_region) / np.sum(human_heatmap) if np.sum(human_heatmap) > 0 else 0.0
                model_ratio = np.sum(model_region) / np.sum(model_attention) if np.sum(model_attention) > 0 else 0.0

                region_attention_ratios.append({
                    'human_ratio': human_ratio,
                    'model_ratio': model_ratio,
                    'ratio_difference': abs(human_ratio - model_ratio)
                })

        # Aggregate region-specific metrics
        regional_metrics['region_wise_correlations'][region_name] = {
            'mean_correlation': np.mean(region_correlations) if region_correlations else 0.0,
            'std_correlation': np.std(region_correlations) if region_correlations else 0.0,
            'sample_count': len(region_correlations)
        }

        if region_attention_ratios:
            human_ratios = [item['human_ratio'] for item in region_attention_ratios]
            model_ratios = [item['model_ratio'] for item in region_attention_ratios]
            ratio_diffs = [item['ratio_difference'] for item in region_attention_ratios]

            regional_metrics['region_attention_ratios'][region_name] = {
                'mean_human_ratio': np.mean(human_ratios),
                'mean_model_ratio': np.mean(model_ratios),
                'mean_ratio_difference': np.mean(ratio_diffs),
                'ratio_correlation': pearsonr(human_ratios, model_ratios)[0] if len(human_ratios) > 1 else 0.0
            }

    return regional_metrics
```

#### Temporal Dynamics: Fixation Sequence Alignment

Advanced analysis of temporal attention patterns and fixation sequence coherence:

```python
def calculate_temporal_dynamics_analysis(human_fixation_sequences, model_attention_trajectories):
    """
    Calculate temporal dynamics and fixation sequence alignment metrics.
    Analyzes how attention evolves over time and sequence coherence.
    """
    from dtaidistance import dtw
    from scipy.stats import pearsonr

    temporal_metrics = {
        'sequence_alignment_scores': [],
        'temporal_correlation_coefficients': [],
        'fixation_duration_analysis': [],
        'attention_trajectory_similarity': [],
        'scanpath_similarity_scores': []
    }

    for human_sequence, model_trajectory in zip(human_fixation_sequences, model_attention_trajectories):
        # Extract temporal features from human fixation sequence
        if len(human_sequence) > 0:
            # Human fixation temporal features
            human_temporal_features = extract_temporal_features(human_sequence)

            # Model attention temporal trajectory
            model_temporal_features = extract_model_temporal_features(model_trajectory)

            # Dynamic Time Warping for sequence alignment
            dtw_score = calculate_dtw_alignment(human_temporal_features, model_temporal_features)
            temporal_metrics['sequence_alignment_scores'].append(dtw_score)

            # Temporal correlation analysis
            temporal_corr = calculate_temporal_correlation(human_sequence, model_trajectory)
            temporal_metrics['temporal_correlation_coefficients'].append(temporal_corr)

            # Fixation duration analysis
            duration_analysis = analyze_fixation_durations(human_sequence, model_trajectory)
            temporal_metrics['fixation_duration_analysis'].append(duration_analysis)

            # Attention trajectory similarity
            trajectory_similarity = calculate_attention_trajectory_similarity(
                human_sequence, model_trajectory
            )
            temporal_metrics['attention_trajectory_similarity'].append(trajectory_similarity)

            # Scanpath similarity using string-based comparison
            scanpath_similarity = calculate_scanpath_similarity(human_sequence, model_trajectory)
            temporal_metrics['scanpath_similarity_scores'].append(scanpath_similarity)

    # Aggregate temporal metrics
    aggregated_temporal_metrics = {}
    for metric_name, metric_values in temporal_metrics.items():
        if metric_name == 'fixation_duration_analysis':
            # Handle duration analysis separately
            duration_corrs = [item['duration_correlation'] for item in metric_values if item]
            duration_diffs = [item['mean_duration_difference'] for item in metric_values if item]

            aggregated_temporal_metrics[metric_name] = {
                'mean_duration_correlation': np.mean(duration_corrs) if duration_corrs else 0.0,
                'mean_duration_difference': np.mean(duration_diffs) if duration_diffs else 0.0,
                'std_duration_difference': np.std(duration_diffs) if duration_diffs else 0.0
            }
        else:
            valid_values = [v for v in metric_values if v is not None and not np.isnan(v)]
            aggregated_temporal_metrics[metric_name] = {
                'mean': np.mean(valid_values) if valid_values else 0.0,
                'std': np.std(valid_values) if valid_values else 0.0,
                'median': np.median(valid_values) if valid_values else 0.0,
                'sample_count': len(valid_values)
            }

    return aggregated_temporal_metrics

def extract_temporal_features(fixation_sequence):
    """Extract temporal features from human fixation sequence."""
    temporal_features = []

    for fixation in fixation_sequence:
        # Assuming fixation format: [x, y, duration, pupil_area]
        features = {
            'x_position': fixation[0],
            'y_position': fixation[1],
            'duration': fixation[2],
            'pupil_area': fixation[3] if len(fixation) > 3 else 1.0,
            'timestamp': fixation[4] if len(fixation) > 4 else 0.0
        }
        temporal_features.append(features)

    return temporal_features

def calculate_dtw_alignment(human_features, model_features):
    """Calculate Dynamic Time Warping alignment score."""
    if len(human_features) == 0 or len(model_features) == 0:
        return 0.0

    # Create coordinate sequences for DTW
    human_coords = [(f['x_position'], f['y_position']) for f in human_features]
    model_coords = [(f['x_position'], f['y_position']) for f in model_features]

    try:
        # Calculate DTW distance (lower is better alignment)
        dtw_distance = dtw.distance(human_coords, model_coords)

        # Normalize DTW distance to similarity score (0-1, higher is better)
        max_possible_distance = np.sqrt(2)  # Maximum distance in normalized coordinates
        normalized_distance = dtw_distance / (max_possible_distance * max(len(human_coords), len(model_coords)))
        alignment_score = max(0.0, 1.0 - normalized_distance)

        return alignment_score

    except Exception:
        return 0.0

def analyze_fixation_durations(human_sequence, model_trajectory):
    """Analyze fixation duration patterns between human and model."""
    if len(human_sequence) == 0:
        return None

    human_durations = [fixation[2] for fixation in human_sequence if len(fixation) > 2]

    if len(human_durations) == 0:
        return None

    # Estimate model "fixation" durations from trajectory
    model_durations = estimate_model_fixation_durations(model_trajectory)

    if len(model_durations) == 0:
        return None

    # Calculate duration statistics
    duration_analysis = {
        'human_mean_duration': np.mean(human_durations),
        'model_mean_duration': np.mean(model_durations),
        'duration_correlation': pearsonr(human_durations[:min(len(human_durations), len(model_durations))],
                                       model_durations[:min(len(human_durations), len(model_durations))])[0] if min(len(human_durations), len(model_durations)) > 1 else 0.0,
        'mean_duration_difference': abs(np.mean(human_durations) - np.mean(model_durations))
    }

    return duration_analysis

def calculate_scanpath_similarity(human_sequence, model_trajectory):
    """Calculate scanpath similarity using string-based representation."""
    if len(human_sequence) == 0:
        return 0.0

    # Convert sequences to grid-based string representations
    human_string = convert_sequence_to_string(human_sequence)
    model_string = convert_sequence_to_string(model_trajectory)

    # Calculate string similarity (e.g., using edit distance)
    similarity = calculate_string_similarity(human_string, model_string)

    return similarity

def convert_sequence_to_string(sequence, grid_size=8):
    """Convert spatial sequence to string representation for scanpath comparison."""
    if len(sequence) == 0:
        return ""

    string_repr = []
    for point in sequence:
        # Convert continuous coordinates to discrete grid positions
        x = int(point[0] * grid_size)
        y = int(point[1] * grid_size)

        # Clamp to grid boundaries
        x = max(0, min(x, grid_size - 1))
        y = max(0, min(y, grid_size - 1))

        # Convert to single character representation
        grid_position = y * grid_size + x
        string_repr.append(chr(ord('A') + grid_position))

    return ''.join(string_repr)

def calculate_string_similarity(str1, str2):
    """Calculate normalized edit distance similarity between two strings."""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0

    # Calculate Levenshtein distance
    edit_distance = calculate_edit_distance(str1, str2)
    max_length = max(len(str1), len(str2))

    # Convert to similarity score
    similarity = 1.0 - (edit_distance / max_length)

    return similarity

def calculate_edit_distance(str1, str2):
    """Calculate Levenshtein edit distance between two strings."""
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]
```

This comprehensive evaluation framework provides rigorous assessment across all modalities of the MIMIC-Eye system, ensuring both technical validity and clinical relevance in multimodal medical AI evaluation.

## A.2 Training Configuration Details

### A.2.1 Hyperparameter Settings

#### Enhanced Gaze Models Configuration

The enhanced gaze attention models employ optimized hyperparameters for multimodal learning:

```python
# Enhanced gaze models hyperparameters
LR = 6e-6                    # Learning rate: 6×10⁻⁶
EPOCHS = 40                  # Training epochs
EARLY_STOPPING_PATIENCE = 10  # Early stopping patience
WARMUP_STEPS = 175           # Learning rate warmup steps

# Memory-adaptive batch configuration
if low_memory or DEVICE == "cpu":
    BATCH_SIZE = 8           # Low memory batch size
    GRAD_ACCUM = 12          # Higher gradient accumulation
    DATALOADER_WORKERS = 0   # Disable multiprocessing
    FP16_ENABLED = False     # Disable mixed precision
else:
    BATCH_SIZE = 32          # Standard batch size
    GRAD_ACCUM = 6           # Standard gradient accumulation
    DATALOADER_WORKERS = 2   # Reduced workers for memory efficiency
    FP16_ENABLED = DEVICE == "cuda"  # Enable mixed precision on GPU
```

#### Baseline Model Configurations

Comparative baseline models utilize different hyperparameter settings:

```python
# MIMIC baseline configuration
MIMIC_LR = 5e-6              # Learning rate: 5×10⁻⁶
MIMIC_EPOCHS = 35            # Reduced epochs for baseline

# Vision Transformer only configuration
VIT_LR = 5e-5                # Higher learning rate: 5×10⁻⁵
VIT_BATCH_SIZE = 128         # Larger batch size for single modality
VIT_EPOCHS = 20              # Fewer epochs required
```

#### AdamW Optimizer Configuration

All models employ AdamW optimizer with carefully tuned parameters:

```python
# AdamW optimizer settings
training_args = TrainingArguments(
    optim="adamw_torch",
    adam_beta1=0.9,          # β₁ = 0.9
    adam_beta2=0.98,         # β₂ = 0.98
    adam_epsilon=1e-8,       # ε = 1×10⁻⁸
    weight_decay=8e-4,       # Weight decay parameter
    max_grad_norm=0.4,       # Gradient clipping threshold
    lr_scheduler_type="cosine", # Cosine annealing scheduler
)
```

#### Gradient Accumulation and Memory Optimization Strategies

Comprehensive memory management ensures scalable training:

```python
# Memory optimization strategies
training_args = TrainingArguments(
    gradient_accumulation_steps=GRAD_ACCUM,
    dataloader_pin_memory=not low_memory,
    dataloader_persistent_workers=not low_memory,
    eval_accumulation_steps=4,  # Batch evaluation for memory efficiency
    fp16=FP16_ENABLED,         # Mixed precision training
    save_on_each_node=False,   # Reduce I/O overhead
    include_inputs_for_metrics=False,  # Memory-efficient metrics
    ignore_data_skip=True,     # Skip problematic samples
)

# CUDA optimization for memory efficiency
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = False    # Deterministic behavior
    torch.backends.cudnn.deterministic = True # Reproducible results
```

### A.2.2 Data Augmentation and Preprocessing

#### Image Normalization Standards

Consistent image preprocessing across all models ensures compatibility:

```python
# Standard image normalization parameters
image_mean = [0.485, 0.456, 0.406]  # ImageNet mean
image_std = [0.229, 0.224, 0.225]   # ImageNet standard deviation

# Alternative normalization for some models
alternative_mean = [0.5, 0.5, 0.5]  # Centered normalization
alternative_std = [0.5, 0.5, 0.5]   # Symmetric scaling
```

#### Training Data Augmentation Pipeline

Memory-optimized augmentation strategies differentiate between training and validation:

```python
# Training augmentation (memory-adaptive)
if split == "train" and not low_memory:
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])
else:
    # Simplified transforms for validation or low memory
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])
```

#### Fixation Normalization Procedures

Standardized fixation sequence processing ensures consistent multimodal integration:

```python
# Fixation statistics normalization
def process_fixations(fix_path, mu, sigma, out_dir):
    # Load standardized fixation features
    df = pd.read_csv(fix_path, usecols=["x_norm", "y_norm", "gaze_duration", "pupil_area_normalized"])
    arr = df.values.astype(np.float32)

    # Apply coordinate and duration constraints
    arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)  # x_norm clipping
    arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)  # y_norm clipping
    arr[:, 2] = np.clip(arr[:, 2], 0.0, None) # duration clipping
    arr[:, 3] = np.clip(arr[:, 3], 0.0, 1.0)  # pupil area clipping

    # Z-score standardization with epsilon guard
    SIGMA = SIGMA + 1e-6  # Numerical stability
    arr = (arr - mu) / SIGMA

    # Handle edge cases and invalid values
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr
```

#### Pupil Area Harmonization Procedures

Unified pupil area processing across data sources:

```python
# REFLACX pupil area (pre-normalized)
pupil_area_reflacx = row['pupil_area_normalized']

# EyeGaze pupil area harmonization
def harmonize_eyegaze_pupil(left_diameter, right_diameter, baseline_area):
    # Compute average pupil area from diameters
    left_area = np.pi * (left_diameter / 2) ** 2
    right_area = np.pi * (right_diameter / 2) ** 2
    avg_area = (left_area + right_area) / 2

    # Normalize against subject-specific baseline
    pupil_area_normalized = avg_area / baseline_area

    return pupil_area_normalized
```

#### Bounding Box Completion Using YOLO for REFLACX Data

The MIMIC-Eye dataset employs an advanced bounding box completion strategy using YOLOv8 to augment sparse REFLACX anatomical annotations. This approach addresses the inherent limitation of REFLACX data, where only subset of anatomical regions may be annotated by radiologists during interpretation, by systematically generating comprehensive spatial attention maps for all relevant chest anatomical structures.

##### Anatomical Region Framework

The system targets 17 standardized anatomical regions based on clinical chest radiography interpretation protocols:

```python
# Standardized anatomical regions for chest X-ray analysis
ANATOMICAL_REGIONS = [
    'cardiac_silhouette',           # Central cardiac structures
    'left_clavicle', 'right_clavicle',  # Superior thoracic landmarks
    'left_costophrenic_angle', 'right_costophrenic_angle',  # Lower pleural spaces
    'left_hilar_structures', 'right_hilar_structures',      # Central vascular regions
    'left_lower_lung_zone', 'right_lower_lung_zone',        # Lower lung fields
    'left_lung', 'right_lung',                              # Complete lung regions
    'left_mid_lung_zone', 'right_mid_lung_zone',           # Middle lung fields
    'left_upper_lung_zone', 'right_upper_lung_zone',       # Upper lung fields
    'trachea',                       # Central airway
    'upper_mediastinum'              # Superior mediastinal structures
]
```

##### YOLOv8 Model Configuration

The completion system utilizes a specialized YOLOv8 model trained specifically for chest X-ray anatomical region detection:

```python
# YOLO-based bounding box completion system
from ultralytics import YOLO
import pandas as pd
from pathlib import Path

# Load pre-trained chest X-ray YOLO model
model = YOLO(weights_path="chestxray_runs/exp1/weights/best.pt")

# Optimized prediction parameters for medical imaging
prediction_config = {
    "conf": 0.05,           # Low confidence threshold for medical sensitivity
    "iou": 0.5,             # Standard IoU for anatomical overlap handling
    "device": 0,            # GPU acceleration
    "verbose": False,       # Silent operation for batch processing
    "save": False,          # Memory-efficient inference
    "save_txt": False       # No auxiliary file generation
}
```

##### Completion Algorithm Implementation

The bounding box completion process follows a systematic approach to enhance spatial coverage while preserving existing radiologist annotations:

```python
def complete_reflacx_bboxes(image_path, existing_reflacx_bboxes):
    """
    Complete missing anatomical bounding boxes using YOLO inference
    while preserving existing REFLACX radiologist annotations.
    """
    # Step 1: Run YOLO inference for comprehensive anatomical detection
    predictions = model.predict(source=image_path, **prediction_config)[0]

    # Step 2: Extract highest confidence detection per anatomical class
    yolo_detections = {}
    for box in predictions.boxes:
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Retain only the most confident detection per anatomical region
        if cls_id not in yolo_detections or yolo_detections[cls_id]["confidence"] < confidence:
            yolo_detections[cls_id] = {
                "study_id": extract_study_id(image_path),
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "class_name": get_anatomical_class_name(cls_id),
                "source": "YOLO_completion"
            }

    # Step 3: Merge with existing REFLACX annotations using priority system
    completed_annotations = merge_annotations_with_priority(
        existing_reflacx_bboxes,
        yolo_detections
    )

    return completed_annotations

def merge_annotations_with_priority(reflacx_bboxes, yolo_detections):
    """
    Merge REFLACX and YOLO annotations with radiologist priority.
    REFLACX annotations take precedence over YOLO detections.
    """
    merged_annotations = {}

    # Priority 1: Preserve all existing REFLACX radiologist annotations
    for region, bbox_data in reflacx_bboxes.items():
        merged_annotations[region] = {
            **bbox_data,
            "source": "REFLACX_radiologist",
            "priority": "high"
        }

    # Priority 2: Add YOLO detections for missing anatomical regions
    for cls_id, detection in yolo_detections.items():
        region_name = detection["class_name"]

        # Only add YOLO detection if region not covered by REFLACX
        if region_name not in merged_annotations:
            merged_annotations[region_name] = {
                **detection,
                "priority": "medium"
            }

    return merged_annotations
```

##### Spatial Attention Integration

The completed bounding boxes are converted to spatial attention masks for multimodal model integration:

```python
def generate_spatial_attention_masks(completed_bboxes, image_height, image_width, target_size=224):
    """
    Convert completed bounding boxes to spatial attention masks
    for integration with multimodal attention mechanisms.
    """
    # Initialize attention mask tensor
    num_regions = len(completed_bboxes)
    attention_masks = torch.zeros((num_regions, target_size, target_size), dtype=torch.float32)

    # Generate mask for each anatomical region
    for i, (region_name, bbox_data) in enumerate(completed_bboxes.items()):
        x1, y1, x2, y2 = bbox_data["bbox"]

        # Normalize coordinates to target image size
        x1_norm = int((x1 / image_width) * target_size)
        y1_norm = int((y1 / image_height) * target_size)
        x2_norm = int((x2 / image_width) * target_size)
        y2_norm = int((y2 / image_height) * target_size)

        # Create binary mask for region
        attention_masks[i, y1_norm:y2_norm, x1_norm:x2_norm] = 1.0

        # Apply Gaussian smoothing for soft attention boundaries
        attention_masks[i] = apply_gaussian_smoothing(attention_masks[i], sigma=1.0)

    return attention_masks
```

##### Quality Assurance and Validation

The completion system includes comprehensive quality control measures:

```python
def validate_bbox_completion(completed_bboxes, image_dimensions):
    """
    Validate completed bounding box annotations for spatial consistency
    and anatomical plausibility.
    """
    validation_metrics = {
        "spatial_coverage": 0.0,
        "anatomical_completeness": 0.0,
        "confidence_distribution": [],
        "source_breakdown": {"REFLACX": 0, "YOLO": 0}
    }

    # Calculate spatial coverage percentage
    total_image_area = image_dimensions[0] * image_dimensions[1]
    covered_area = sum([
        calculate_bbox_area(bbox_data["bbox"])
        for bbox_data in completed_bboxes.values()
    ])
    validation_metrics["spatial_coverage"] = covered_area / total_image_area

    # Assess anatomical completeness
    essential_regions = ['cardiac_silhouette', 'left_lung', 'right_lung', 'trachea']
    completed_essential = sum([
        1 for region in essential_regions
        if region in completed_bboxes
    ])
    validation_metrics["anatomical_completeness"] = completed_essential / len(essential_regions)

    # Analyze confidence and source distribution
    for region, bbox_data in completed_bboxes.items():
        validation_metrics["confidence_distribution"].append(bbox_data.get("confidence", 1.0))
        validation_metrics["source_breakdown"][bbox_data["source"]] += 1

    return validation_metrics
```

##### Clinical Integration Benefits

This YOLO-based completion strategy provides several advantages for multimodal medical AI:

1. **Comprehensive Spatial Coverage**: Ensures all anatomically relevant regions are represented, addressing radiologist attention variability
2. **Preserved Expert Knowledge**: Maintains original REFLACX annotations while supplementing missing regions
3. **Standardized Representation**: Creates consistent 17-region anatomical maps across all studies
4. **Enhanced Model Performance**: Provides richer spatial context for attention mechanisms and cross-modal learning
5. **Scalable Processing**: Enables automated completion for large-scale dataset processing

The integration of YOLO-completed bounding boxes with original REFLACX annotations creates a hybrid annotation system that combines the precision of expert radiologist interpretation with the comprehensiveness of automated anatomical region detection, ultimately enhancing the spatial attention capabilities of multimodal medical AI models.

#### Mask Generation and Spatial Processing

Bounding box to mask conversion for spatial attention mechanisms:

```python
# Mask transformation pipeline
mask_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

# Spatial mask generation from bounding boxes
def generate_spatial_mask(bboxes, image_height, image_width):
    num_objects = len(bboxes)
    masks = torch.zeros((num_objects, image_height, image_width), dtype=torch.uint8)

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        masks[i, y1:y2, x1:x2] = 1

    return masks
```

#### Training Loop Configuration

Comprehensive training strategy with advanced monitoring:

```python
# Training arguments with memory optimization
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_steps=WARMUP_STEPS,
    learning_rate=LR,
    weight_decay=WD,

    # Evaluation and saving strategy
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,

    # Logging and monitoring
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=10,
    report_to="none",

    # Performance optimization
    dataloader_num_workers=DATALOADER_WORKERS,
    remove_unused_columns=False,
    prediction_loss_only=False,
)

# Advanced callback configuration
callbacks = [
    EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
    LossBasedEarlyStoppingCallback(patience=20, min_delta=0.001),
    OverfittingMonitorCallback()
]
```

This training configuration framework ensures reproducible, memory-efficient, and performance-optimized multimodal model training while maintaining compatibility across different hardware configurations and model architectures.

## A.1.3 Data Modalities: Multimodal Integration

### Chest X-ray Images

- **Format**: JPEG images
- **Processing**: Converted to RGB format for neural network compatibility
- **Standardization**: Normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Resolution**: Standardized to 512×512 pixels for model input
- **Coverage**: 100% of dataset records (3,654 images)

### Eye-Gaze Fixation Data

- **Total Records with Eye-Gaze**: 1,100 studies
- **Unique Patients with Eye-Gaze**: 1,038 patients
- **Format**: CSV files containing normalized fixation coordinates
- **Processing Modes**:
  - Silent phase: Fixations before radiologist begins reporting
  - Reporting phase: Fixations during active radiological interpretation
  - All phases: Complete fixation sequence
- **Heatmap Generation**: Fixations converted to spatial attention heatmaps using Gaussian kernels
- **Coordinate Normalization**: Fixation coordinates normalized to image dimensions

### Anatomical Bounding Boxes

- **Source**: REFLACX dataset annotations
- **Format**: CSV files with bounding box coordinates (xmin, ymin, xmax, ymax)
- **Coverage**: Available for REFLACX subset (2,617 records)
- **Coordinate System**: Pixel-based coordinates with automatic resizing for different image dimensions
- **Certainty Levels**: Annotations include radiologist certainty scores

### Radiologist Transcripts

- **Coverage**: 100% of dataset records
- **Format**: JSON files containing structured transcription data
- **Sources**:
  - REFLACX transcripts: Expert radiologist verbal reports during image interpretation
  - Eye-Gaze transcripts: Audio segmentation transcripts synchronized with gaze data
- **Processing**: Text extraction and tokenization for natural language processing tasks

### Multimodal Data Availability

- **REFLACX-only records**: 2,554 studies (REFLACX data without eye-gaze)
- **Eye-Gaze-only records**: 1,037 studies (Eye-gaze data without REFLACX)
- **Full multimodal records**: 63 studies (Both REFLACX and eye-gaze data)
- **Image-only records**: 0 studies (All records have at least image + report)

## A.1.3 Quality Control: Data Cleaning and Preprocessing

### Missing Data Handling

- **Strategy**: Complete case analysis with systematic removal of records containing any missing values
- **Implementation**: Applied `pandas.DataFrame.dropna()` for comprehensive missing data removal
- **Impact**: Ensures consistent multimodal data availability across all retained records

### Outlier Detection and Removal

- **Clinical Variables**: Applied statistical outlier detection using z-score thresholds (>3 standard deviations)
- **Image Quality**: Automated filtering for corrupted or unreadable image files
- **Fixation Data**: Removal of physiologically impossible fixation coordinates (outside image boundaries)

### Data Validation Procedures

- **Path Verification**: Systematic validation of all file paths for images, transcripts, and metadata
- **Format Consistency**: Standardized file formats across modalities (JPEG for images, CSV for tabular data, JSON for transcripts)
- **Coordinate Validation**: Verification of bounding box coordinates within image dimensions
- **Temporal Validation**: Synchronization checks between eye-gaze timestamps and transcript timing

### Dataset Integrity Checks

- **Unique Identifier Validation**: Verification of DICOM IDs and study IDs for data consistency
- **Cross-Modal Verification**: Confirmation of data availability across modalities for multimodal records
- **Patient Privacy Compliance**: De-identification verification following HIPAA guidelines

## A.1.4 Condition Filtering: Systematic Condition Removal

### Removed Conditions

The following 6 medical conditions were systematically filtered from the original dataset:

1. **Consolidation**: Removed due to clinical overlap with pneumonia and potential annotation inconsistencies
2. **Pneumothorax**: Filtered due to low prevalence and specialized clinical context
3. **Lung Lesion**: Removed for clinical specificity (consolidated into more general "Lung Opacity")
4. **Enlarged Cardiomediastinum**: Filtered due to overlap with cardiomegaly classification
5. **Fracture**: Removed as orthopedic finding outside primary chest pathology focus
6. **Pleural Other**: Filtered due to ambiguous classification (consolidated into "Pleural Effusion")

### Filtering Implementation

```python
# Systematic condition removal process
unwanted_conditions = [
    "Consolidation", "Pneumothorax", "Lung Lesion",
    "Enlarged Cardiomediastinum", "Fracture", "Pleural Other"
]

def contains_unwanted(condition_str):
    if pd.isna(condition_str):
        return False
    return any(cond in condition_str.split('|') for cond in unwanted_conditions)

# Applied filtering with impact tracking
original_count = len(original_dataset)
filtered_dataset = original_dataset[~original_dataset['condition'].apply(contains_unwanted)]
removed_count = original_count - len(filtered_dataset)
```

### Multi-Label Condition Handling

- **Format**: Conditions stored as pipe-separated strings (e.g., "Cardiomegaly|Support Devices")
- **Processing**: Binary feature encoding for each individual condition
- **Multi-label Prevalence**: Significant portion of records contain multiple concurrent conditions
- **Clinical Relevance**: Reflects realistic clinical scenarios where patients present with multiple pathological findings

### Final Condition Set Rationale

The 8 retained conditions represent:

- **High Clinical Relevance**: Most common pathological findings in chest radiography
- **Clear Diagnostic Criteria**: Well-defined radiological features with established interpretation guidelines
- **Balanced Representation**: Mixture of cardiac, pulmonary, and normal findings
- **Educational Value**: Conditions commonly encountered in clinical practice and medical education

### Dataset Splitting Strategy

- **Train Set**: 70% of data (stratified by condition distribution)
- **Test Set**: 15% of data (maintaining condition balance)
- **Validation Set**: 15% of data (preserving rare condition representation)
- **Stratification Method**: Condition-aware splitting to maintain statistical balance across all 8 conditions
- **Random Seed**: Fixed seed (42) for reproducible dataset splits

## A.1.5 Technical Implementation Details

### Data Loading Architecture

- **Dataset Class**: `ReflacxDataset` implementing PyTorch Dataset interface
- **Multimodal Integration**: Unified data loader supporting selective modality inclusion
- **Memory Optimization**: Lazy loading with on-demand data preprocessing
- **Batch Processing**: Efficient collation functions for variable-size inputs

### Image Preprocessing Pipeline

1. **Loading**: PIL Image loading with RGB conversion
2. **Augmentation**: Training-time augmentations (horizontal flip, rotation, color jitter)
3. **Resizing**: Bilinear interpolation to 512×512 resolution
4. **Normalization**: ImageNet-based normalization for transfer learning compatibility
5. **Tensor Conversion**: PyTorch tensor format for model input

### Eye-Gaze Processing Pipeline

1. **CSV Parsing**: Fixation coordinate extraction from normalized CSV files
2. **Temporal Filtering**: Phase-specific fixation selection (silent/reporting/all)
3. **Heatmap Generation**: Gaussian kernel convolution for spatial attention maps
4. **Resolution Matching**: Heatmap resizing to match image dimensions
5. **Normalization**: Consistent preprocessing with image modality

This comprehensive dataset specification demonstrates the MIMIC-Eye dataset's robust multimodal architecture, systematic quality control procedures, and clinical relevance for advancing multimodal medical AI research.
