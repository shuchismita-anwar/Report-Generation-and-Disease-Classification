import os
import warnings
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

# Local module imports
from data_utils import load_preprocess_image, read_text_file
from datasets import ImageTextDataset, DiseaseClassificationDataset
from models import ClinicalDataEncoder, DiseaseClassifier
from llm_loader import load_llm_model
from metrics import calculate_text_generation_metrics

# Import MedCLIP from wherever it is installed
from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPTextModel

warnings.filterwarnings('ignore')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Using device: {device}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    g = torch.Generator()
    g.manual_seed(42)

    # Paths and reading data
    XAMI_MIMIC_PATH = os.path.join(
        'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0',
        'mimic-eye'
    ).replace('\\', '/')
    MIMIC_EYE_PATH = XAMI_MIMIC_PATH

    df = pd.read_csv('test.csv')
    df = df[df['eye_tracking_data_discarded'] == False]

    critical_columns = [
        'image_path', 'bbox_path', 'fixation_path', 'report_path', 
        'reflacx_transcript_path', 'eyegaze_transcript_path', 
        'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 
        'dbp', 'pain', 'acuity', 'chiefcomplaint', 'age'
    ]
    df = df.dropna(subset=critical_columns)

    for path_col in [
        'image_path', 'bbox_path', 'fixation_path', 
        'report_path','reflacx_transcript_path', 'eyegaze_transcript_path'
    ]:
        if path_col in df.columns:
            df[path_col] = df[path_col].str.replace(
                '{XAMI_MIMIC_PATH}', XAMI_MIMIC_PATH
            ).str.replace('\\', '/')

    for col in ['report_path', 'reflacx_transcript_path', 'eyegaze_transcript_path']:
        if col in df.columns:
            df[col] = df[col].str.replace(
                'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye',
                MIMIC_EYE_PATH,
                regex=False
            )

    clinical_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'age']
    df[clinical_cols] = df[clinical_cols].astype(float)
    df[clinical_cols] = df[clinical_cols].fillna(df[clinical_cols].mean())
    scaler = StandardScaler()
    df[clinical_cols] = scaler.fit_transform(df[clinical_cols])

    # Initialize MedCLIP
    from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPTextModel
    try:
        vision_model = MedCLIPVisionModel()
        text_model = MedCLIPTextModel()
        medclip_model = MedCLIPModel().to(device)
        print("Successfully initialized MedCLIP models.")
    except Exception as e:
        print(f"Error initializing MedCLIP models: {e}")
        raise e

    # Datasets and embeddings
    disease_labels = [
        'Atelectasis', 'Pneumothorax', 'Consolidation', 'Pulmonary edema',
        'Enlarged cardiac silhouette', 'Fibrosis', 'Fracture', 'Support devices'
    ]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        model_max_length=77
    )

    unified_embeddings = []
    feature_dataset = ImageTextDataset(df, tokenizer, device)

    feature_dataloader = DataLoader(
        feature_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        generator=g
    )

    clinical_data = df[clinical_cols].values.astype(np.float32)
    with torch.no_grad():
        batch_idx_local = 0
        for batch in feature_dataloader:
            images, input_ids, attention_mask = batch
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            image_features = medclip_model.vision_model(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = medclip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            unified_embedding = (image_features + text_features) / 2
            unified_embedding = unified_embedding / unified_embedding.norm(dim=-1, keepdim=True)
            unified_embedding = unified_embedding.cpu()

            start_idx = batch_idx_local * feature_dataloader.batch_size
            for i in range(unified_embedding.size(0)):
                idx = start_idx + i
                if idx >= len(df):
                    break
                row_data = df.iloc[idx]
                unified_embeddings.append({
                    'id': row_data['id'],
                    'unified_embedding': unified_embedding[i].unsqueeze(0),
                    'labels': row_data,
                    'fixation_path': row_data['fixation_path'],
                    'clinical_data': torch.tensor(clinical_data[idx])
                })
            batch_idx_local += 1

    for entry in unified_embeddings:
        labels = []
        row_data = entry['labels']
        for label in disease_labels:
            try:
                value = float(row_data[label]) if pd.notna(row_data[label]) else 0.0
            except (ValueError, KeyError):
                value = 0.0
            labels.append(value)
        entry['disease_labels'] = torch.tensor(labels, dtype=torch.float32)

    total_length = len(unified_embeddings)
    train_length = int(0.8 * total_length)
    val_length = total_length - train_length

    train_data, val_data = random_split(
        unified_embeddings, 
        [train_length, val_length], 
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset_cls = DiseaseClassificationDataset(train_data)
    val_dataset_cls = DiseaseClassificationDataset(val_data)

    train_dataloader_cls = DataLoader(
        train_dataset_cls,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        generator=g
    )

    val_dataloader_cls = DataLoader(
        val_dataset_cls,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        generator=g
    )

    clinical_encoder = ClinicalDataEncoder(input_dim=len(clinical_cols)).to(device)
    classifier_model = DiseaseClassifier(
        input_dim=512,
        clinical_dim=128,
        num_classes=len(disease_labels)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_cls = optim.Adam(
        list(classifier_model.parameters()) + list(clinical_encoder.parameters()), 
        lr=1e-4
    )
    epochs_cls = 5

    for epoch in range(epochs_cls):
        classifier_model.train()
        clinical_encoder.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (embeddings, clinical_data_batch, labels) in enumerate(train_dataloader_cls):
            embeddings = embeddings.to(device)
            clinical_data_batch = clinical_data_batch.to(device)
            labels = labels.to(device)

            optimizer_cls.zero_grad()

            clinical_embedding = clinical_encoder(clinical_data_batch)
            outputs = classifier_model(embeddings, clinical_embedding)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_cls.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Validation
        classifier_model.eval()
        clinical_encoder.eval()
        all_labels = []
        all_probs = []
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for embeddings, clinical_data_batch, labels in val_dataloader_cls:
                embeddings = embeddings.to(device)
                clinical_data_batch = clinical_data_batch.to(device)
                labels = labels.to(device)

                clinical_embedding = clinical_encoder(clinical_data_batch)
                outputs = classifier_model(embeddings, clinical_embedding)

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                num_val_batches += 1

                probs = torch.sigmoid(outputs).cpu().numpy()
                labels = labels.cpu().numpy()

                all_probs.append(probs)
                all_labels.append(labels)

        avg_val_loss = val_loss / num_val_batches
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Calculate metrics
        aurocs = []
        f1s = []
        precisions = []
        recalls = []

        from sklearn.exceptions import UndefinedMetricWarning
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        for i, disease in enumerate(disease_labels):
            try:
                if len(np.unique(all_labels[:, i])) > 1:
                    auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                    aurocs.append(auroc)
                else:
                    aurocs.append(0.5)

                disease_preds = (all_probs[:, i] >= 0.5).astype(int)
                disease_labels_true = all_labels[:, i]

                f1 = f1_score(disease_labels_true, disease_preds, average='weighted', zero_division=0)
                precision = precision_score(disease_labels_true, disease_preds, average='weighted', zero_division=0)
                recall = recall_score(disease_labels_true, disease_preds, average='weighted', zero_division=0)

                f1s.append(f1)
                precisions.append(precision)
                recalls.append(recall)

            except Exception as e:
                aurocs.append(0.5)
                f1s.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)

        macro_auroc = np.mean(aurocs)
        macro_f1 = np.mean(f1s)
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        all_preds = (all_probs >= 0.5).astype(int)
        accuracy = accuracy_score(all_labels.ravel(), all_preds.ravel())

    # Now load the LLM
    print("\nLoading LLM model...")
    try:
        tokenizer_llm, llm_model = load_llm_model(device)
    except Exception as e:
        print("Failed to load LLM model. Exiting.")
        return

    llm_model.model.eval()

    # Generate reports
    print("\nGenerating reports...")
    reports = []
    reference_reports = []  # Suppose you have references; left empty if not provided

    prompt_template = """[INST]You are an experienced radiologist writing a chest X-ray report. Based on the following findings:

Disease Probabilities:
{}

Clinical Data:
{}

Write a clear, professional report with two sections:
FINDINGS: Describe all observations from the X-ray, including both normal and abnormal findings. If no abnormalities are found, clearly state that the chest X-ray appears normal and describe the normal structures visible.

IMPRESSION: Provide a concise summary of key findings and their clinical significance. For normal studies, state that the chest X-ray is within normal limits.[/INST]

FINDINGS:"""

    disease_labels = [
        'Atelectasis', 'Pneumothorax', 'Consolidation', 'Pulmonary edema',
        'Enlarged cardiac silhouette', 'Fibrosis', 'Fracture', 'Support devices'
    ]
    clinical_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'age']

    for entry in unified_embeddings:
        print(f"\nGenerating report for ID {entry['id']}:")
        print("-" * 50)
        try:
            classifier_model.eval()
            clinical_encoder.eval()
            with torch.no_grad():
                embedding = entry['unified_embedding'].to(device)
                clinical_data_tensor = entry['clinical_data'].to(device)
                clinical_embedding = clinical_encoder(clinical_data_tensor.unsqueeze(0))
                logits = classifier_model(embedding.unsqueeze(0), clinical_embedding)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            # Build disease probabilities text
            disease_probs = dict(zip(disease_labels, probs))
            disease_prob_str = "High Probability Findings (>50%):\n"
            high_prob = [(d, p) for d, p in disease_probs.items() if p > 0.5]
            if high_prob:
                for disease, prob in sorted(high_prob, key=lambda x: x[1], reverse=True):
                    disease_prob_str += f"* {disease}: {prob:.1%}\n"
            else:
                disease_prob_str += "* No high probability findings\n"
            
            disease_prob_str += "\nModerate Probability Findings (30-50%):\n"
            mod_prob = [(d, p) for d, p in disease_probs.items() if 0.3 <= p <= 0.5]
            if mod_prob:
                for disease, prob in sorted(mod_prob, key=lambda x: x[1], reverse=True):
                    disease_prob_str += f"* {disease}: {prob:.1%}\n"
            else:
                disease_prob_str += "* No moderate probability findings\n"

            # Build clinical data text
            clinical_data_np = entry['clinical_data'].cpu().numpy()
            clinical_data_dict = dict(zip(clinical_cols, clinical_data_np))
            clinical_data_str = "Clinical Measurements:\n"
            for col, val in clinical_data_dict.items():
                if abs(val) > 1.5:
                    status = "Significantly " + ("High" if val > 0 else "Low")
                    clinical_data_str += f"* {col}: {val:.1f} ({status})\n"
                elif abs(val) > 1.0:
                    status = "Mildly " + ("High" if val > 0 else "Low")
                    clinical_data_str += f"* {col}: {val:.1f} ({status})\n"
                else:
                    clinical_data_str += f"* {col}: {val:.1f} (Normal)\n"

            prompt = prompt_template.format(disease_prob_str, clinical_data_str)
            
            sequences = llm_model(
                prompt,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                num_return_sequences=1,
                max_new_tokens=300,
                repetition_penalty=1.1,
                pad_token_id=tokenizer_llm.pad_token_id,
                eos_token_id=tokenizer_llm.eos_token_id,
                min_new_tokens=100,
                no_repeat_ngram_size=3
            )

            generated_text = sequences[0]['generated_text']
            report_start = generated_text.find("FINDINGS:")
            if report_start != -1:
                report_text = generated_text[report_start:]
                if "IMPRESSION:" not in report_text:
                    findings = report_text.strip()
                    if not findings:
                        findings = ("FINDINGS:\nChest X-ray appears normal. "
                                    "Heart size and pulmonary vasculature are within normal limits. "
                                    "No focal consolidation, effusion, or pneumothorax.")
                    report_text = (
                        findings +
                        "\n\nIMPRESSION:\nNormal chest X-ray. No acute cardiopulmonary disease."
                    )
                report_text = report_text.replace("[/INST]", "").strip()
                print(report_text)
                reports.append(report_text)
            else:
                default_report = """FINDINGS:
Chest X-ray appears normal. Heart size and pulmonary vasculature are within normal limits. No focal consolidation, effusion, or pneumothorax. Osseous structures are intact.

IMPRESSION:
Normal chest X-ray. No acute cardiopulmonary disease."""
                print(default_report)
                reports.append(default_report)

        except Exception as e:
            print(f"Error generating report: {str(e)}")
            error_report = """FINDINGS:
Unable to fully analyze the chest X-ray due to technical limitations. Basic structures appear grossly normal.

IMPRESSION:
Limited study. Clinical correlation recommended."""
            reports.append(error_report)
            continue

    # Suppose you have reference_reports for metric calculation
    generated_texts = [report for report in reports]
    reference_texts = [rep for rep in reference_reports]  # If you have them

    text_metrics = calculate_text_generation_metrics(generated_texts, reference_texts)
    for metric_name, value in text_metrics.items():
        pass  # Do something with the metrics

if __name__ == '__main__':
    main()
