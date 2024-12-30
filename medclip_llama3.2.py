import os
import pandas as pd
import torch
from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPTextModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import warnings
from torchvision import transforms

# -------------------------------------------------------------------------
# Step 1: Configuration and Setup
# -------------------------------------------------------------------------

warnings.filterwarnings('ignore')

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    model_max_length=77
)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------------------------------------------------
# Step 2: Helper Functions
# -------------------------------------------------------------------------

def read_text_file(file_path):
    if pd.isna(file_path) or file_path == '':
        return ''
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return text
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ''

def load_preprocess_image(path, device):
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        img = Image.new('RGB', (224, 224))
    img = image_transform(img)
    return img.to(device)

# -------------------------------------------------------------------------
# Step 3: Dataset Classes
# -------------------------------------------------------------------------

class ImageTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, device):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        image = load_preprocess_image(image_path, self.device)

        report_text = read_text_file(row['report_path'])
        reflacx_text = read_text_file(row['reflacx_transcript_path'])
        eyegaze_text = read_text_file(row['eyegaze_transcript_path'])
        combined_text = ' '.join([report_text, reflacx_text, eyegaze_text])

        encoded_text = self.tokenizer(
            combined_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=77
        )
        input_ids = encoded_text['input_ids'].squeeze(0).to(self.device)
        attention_mask = encoded_text['attention_mask'].squeeze(0).to(self.device)

        return image, input_ids, attention_mask

class DiseaseClassificationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = self.data[idx]['unified_embedding'].squeeze(0)
        labels = self.data[idx]['disease_labels']
        clinical_data = self.data[idx]['clinical_data'].float()
        return embedding, clinical_data, labels

# -------------------------------------------------------------------------
# Step 4: Model Definitions
# -------------------------------------------------------------------------

class ClinicalDataEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super(ClinicalDataEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.fc(x)

class DiseaseClassifier(nn.Module):
    def __init__(self, input_dim=512, clinical_dim=128, num_classes=8):
        super(DiseaseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim + clinical_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, clinical_data):
        if x.dim() == 3:
            x = x.squeeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        
        if clinical_data.dim() == 3:
            clinical_data = clinical_data.squeeze(0)
        elif clinical_data.dim() == 1:
            clinical_data = clinical_data.unsqueeze(0)
        
        if x.size(0) != clinical_data.size(0):
            if x.size(0) == 1:
                x = x.expand(clinical_data.size(0), -1)
            elif clinical_data.size(0) == 1:
                clinical_data = clinical_data.expand(x.size(0), -1)
        
        x = torch.cat([x, clinical_data], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------------------------------------------------------
# Step 5: Load LLM Model
# -------------------------------------------------------------------------

def load_llm_model(device):
    try:
        model_name = "meta-llama/Llama-3.2-3B"
        hf_token = "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"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
        model.to(device)
        
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            model.resize_token_embeddings(len(tokenizer))
        
        model.config.pad_token_id = tokenizer.pad_token_id
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device=0 if torch.cuda.is_available() else -1,
        )
        
        return tokenizer, pipe
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        raise

# -------------------------------------------------------------------------
# Step 6: Text Generation Metrics
# -------------------------------------------------------------------------

def calculate_text_generation_metrics(generated_reports, reference_reports):
    metrics = {
        'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': [],
        'rouge_1_p': [], 'rouge_1_r': [], 'rouge_1_f': [],
        'rouge_2_p': [], 'rouge_2_r': [], 'rouge_2_f': [],
        'rouge_l_p': [], 'rouge_l_r': [], 'rouge_l_f': [],
        'avg_length': []
    }

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth_fn = SmoothingFunction().method1

    for gen_report, ref_report in zip(generated_reports, reference_reports):
        if not gen_report.strip() or not ref_report.strip():
            continue

        metrics['avg_length'].append(len(gen_report.split()))
        
        reference = nltk.word_tokenize(ref_report.lower())
        hypothesis = nltk.word_tokenize(gen_report.lower())
        
        try:
            metrics['bleu_1'].append(sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0), smoothing_function=smooth_fn))
            metrics['bleu_2'].append(sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn))
            metrics['bleu_3'].append(sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn))
            metrics['bleu_4'].append(sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn))
        except Exception as e:
            print(f"Error calculating BLEU scores: {str(e)}")

        try:
            scores = scorer.score(ref_report, gen_report)
            
            metrics['rouge_1_p'].append(scores['rouge1'].precision)
            metrics['rouge_1_r'].append(scores['rouge1'].recall)
            metrics['rouge_1_f'].append(scores['rouge1'].fmeasure)
            
            metrics['rouge_2_p'].append(scores['rouge2'].precision)
            metrics['rouge_2_r'].append(scores['rouge2'].recall)
            metrics['rouge_2_f'].append(scores['rouge2'].fmeasure)
            
            metrics['rouge_l_p'].append(scores['rougeL'].precision)
            metrics['rouge_l_r'].append(scores['rougeL'].recall)
            metrics['rouge_l_f'].append(scores['rougeL'].fmeasure)
        except Exception as e:
            print(f"Error calculating ROUGE scores: {str(e)}")
    
    final_metrics = {}
    for key in metrics:
        if metrics[key]:
            final_metrics[key] = np.mean(metrics[key])
    return final_metrics

# -------------------------------------------------------------------------
# Step 7: Main Execution Function
# -------------------------------------------------------------------------

def main():
    # Device and Seed Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Using device: {device}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    g = torch.Generator()
    g.manual_seed(42)

    # Path Configurations
    XAMI_MIMIC_PATH = os.path.join(
        'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0',
        'mimic-eye'
    ).replace('\\', '/')
    MIMIC_EYE_PATH = XAMI_MIMIC_PATH

    # Data Loading and Preprocessing
    df = pd.read_csv('test.csv')
    df = df[df['eye_tracking_data_discarded'] == False]

    critical_columns = [
        'image_path', 'bbox_path', 'fixation_path', 'report_path', 'reflacx_transcript_path',
        'eyegaze_transcript_path', 'temperature', 'heartrate', 'resprate',
        'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'chiefcomplaint', 'age'
    ]
    df = df.dropna(subset=critical_columns)

    for path_col in [
        'image_path', 'bbox_path', 'fixation_path', 'report_path',
        'reflacx_transcript_path', 'eyegaze_transcript_path'
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

    # Initialize MedCLIP Models
    try:
        vision_model = MedCLIPVisionModel()
        text_model = MedCLIPTextModel()
        medclip_model = MedCLIPModel().to(device)
        print("Successfully initialized MedCLIP models.")
    except Exception as e:
        print(f"Error initializing MedCLIP models: {e}")
        raise e

    disease_labels = [
        'Atelectasis', 'Pneumothorax', 'Consolidation', 'Pulmonary edema',
        'Enlarged cardiac silhouette', 'Fibrosis', 'Fracture', 'Support devices'
    ]

    # Prepare Unified Embeddings
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

    # Add Disease Labels to Embeddings
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

    # Split Data into Training and Validation Sets
    total_length = len(unified_embeddings)
    train_length = int(0.8 * total_length)
    val_length = total_length - train_length

    train_data, val_data = random_split(unified_embeddings, [train_length, val_length], generator=torch.Generator().manual_seed(42))

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

    # Initialize Classification Models
    clinical_encoder = ClinicalDataEncoder(input_dim=len(clinical_cols)).to(device)
    classifier_model = DiseaseClassifier(input_dim=512, clinical_dim=128, num_classes=len(disease_labels)).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_cls = optim.Adam(
        list(classifier_model.parameters()) + list(clinical_encoder.parameters()), 
        lr=1e-4
    )
    epochs_cls = 5

    # -------------------------------------------------------------------------
    # Step 8: Training Loop
    # -------------------------------------------------------------------------

    for epoch in range(epochs_cls):
        classifier_model.train()
        clinical_encoder.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (embeddings, clinical_data, labels) in enumerate(train_dataloader_cls):
            embeddings = embeddings.to(device)
            clinical_data = clinical_data.to(device)
            labels = labels.to(device)

            optimizer_cls.zero_grad()

            clinical_embedding = clinical_encoder(clinical_data)
            outputs = classifier_model(embeddings, clinical_embedding)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_cls.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        classifier_model.eval()
        clinical_encoder.eval()
        all_labels = []
        all_probs = []
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for embeddings, clinical_data, labels in val_dataloader_cls:
                embeddings = embeddings.to(device)
                clinical_data = clinical_data.to(device)
                labels = labels.to(device)

                clinical_embedding = clinical_encoder(clinical_data)
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

    # -------------------------------------------------------------------------
    # Step 9: Load and Prepare LLM Model
    # -------------------------------------------------------------------------

    print("\nLoading LLM model...")
    try:
        tokenizer_llm, llm_model = load_llm_model(device)
    except Exception as e:
        print("Failed to load LLM model. Exiting.")
        return

    llm_model.model.eval()

    # -------------------------------------------------------------------------
    # Step 10: Generate Reports
    # -------------------------------------------------------------------------

    print("\nGenerating reports...")
    reports = []
    reference_reports = []

    prompt_template = """[INST]You are a radiologist. Write a chest X-ray report based on these findings:

Disease Probabilities:
{}

Clinical Data:
{}

Write a clear and concise report with two sections:
FINDINGS: Describe any abnormalities seen on the X-ray, focusing on those with high probabilities.
IMPRESSION: Summarize the key findings and their clinical significance.

Use standard radiological terminology and be specific.[/INST]

FINDINGS:"""

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

            # Format disease probabilities
            disease_probs = dict(zip(disease_labels, probs))
            disease_prob_str = "High Probability Findings (>50%):\n"
            high_prob = [(d, p) for d, p in disease_probs.items() if p > 0.5]
            if high_prob:
                for disease, prob in sorted(high_prob, key=lambda x: x[1], reverse=True):
                    disease_prob_str += f"* {disease}: {prob:.1%}\n"
            else:
                disease_prob_str += "* None\n"
            
            disease_prob_str += "\nModerate Probability Findings (30-50%):\n"
            mod_prob = [(d, p) for d, p in disease_probs.items() if 0.3 <= p <= 0.5]
            if mod_prob:
                for disease, prob in sorted(mod_prob, key=lambda x: x[1], reverse=True):
                    disease_prob_str += f"* {disease}: {prob:.1%}\n"
            else:
                disease_prob_str += "* None\n"

            # Format clinical data with interpretations
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
                max_new_tokens=400,
                repetition_penalty=1.1,
                pad_token_id=tokenizer_llm.pad_token_id,
                eos_token_id=tokenizer_llm.eos_token_id,
                min_new_tokens=100,
                no_repeat_ngram_size=3
            )

            generated_text = sequences[0]['generated_text']
            
            # Clean up the generated text
            report_start = generated_text.find("FINDINGS:")
            if report_start != -1:
                report = generated_text[report_start:].strip()
            else:
                report = "FINDINGS:\nNo significant abnormalities noted.\n\nIMPRESSION:\n1. Normal chest radiograph."
            
            # Remove special tokens and formatting
            report = report.replace("[/INST]", "").replace("[/SYS]", "").replace("[ASST]", "").strip()
            report = report.split("[")[0].strip()
            
            # Remove any lines that look like formatting instructions
            report_lines = report.split("\n")
            cleaned_lines = []
            for line in report_lines:
                if not any(x in line.lower() for x in ["list any", "summarize", "use standard", "write a", "generate a"]):
                    cleaned_lines.append(line)
            report = "\n".join(cleaned_lines).strip()
            
            # Ensure proper structure
            if "IMPRESSION:" not in report:
                findings = report[9:].strip()  # Get text after "FINDINGS:"
                if any(term in findings.lower() for term in ["abnormal", "disease", "condition", "pathology"]):
                    report += "\n\nIMPRESSION:\n1. Abnormal findings as described above, clinical correlation recommended."
                else:
                    report += "\n\nIMPRESSION:\n1. No acute cardiopulmonary abnormalities."
                
            print(report)
            print("-" * 50)
            
            reports.append({
                'id': entry['id'],
                'report': report
            })

            reference_report = read_text_file(entry['labels']['report_path'])
            reference_reports.append(reference_report)

        except Exception as e:
            continue

    # -------------------------------------------------------------------------
    # Step 11: Calculate Text Generation Metrics
    # -------------------------------------------------------------------------

    generated_texts = [report['report'] for report in reports]
    reference_texts = [rep for rep in reference_reports]
    text_metrics = calculate_text_generation_metrics(generated_texts, reference_texts)
    for metric_name, value in text_metrics.items():
        pass


if __name__ == '__main__':
    main()
