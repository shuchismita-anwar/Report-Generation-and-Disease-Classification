import os
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM, pipeline
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import warnings

# Download required NLTK data
nltk.download('punkt')
warnings.filterwarnings('ignore')


class ImageTextDataset(Dataset):
    def __init__(self, dataframe, processor, device):
        self.data = dataframe.reset_index(drop=True)
        self.processor = processor
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            image = Image.new('RGB', (224, 224))

        report_text = read_text_file(row['report_path'])
        reflacx_text = read_text_file(row['reflacx_transcript_path'])
        eyegaze_text = read_text_file(row['eyegaze_transcript_path'])
        combined_text = ' '.join([report_text, reflacx_text, eyegaze_text])

        inputs = self.processor(
            text=combined_text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )

        for k in inputs.keys():
            inputs[k] = inputs[k].squeeze(0).to(self.device)

        return inputs


class DiseaseClassificationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = self.data[idx]['unified_embedding'].squeeze(0)
        labels = self.data[idx]['labels']
        clinical_data = self.data[idx]['clinical_data'].float()
        return embedding, clinical_data, labels


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
        # Ensure x is 2D
        if x.dim() == 3:
            x = x.squeeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Ensure clinical_data is 2D
        if clinical_data.dim() == 3:
            clinical_data = clinical_data.squeeze(0)
        elif clinical_data.dim() == 1:
            clinical_data = clinical_data.unsqueeze(0)
        
        # Ensure both tensors have the same batch size
        if x.size(0) != clinical_data.size(0):
            if x.size(0) == 1:
                x = x.expand(clinical_data.size(0), -1)
            elif clinical_data.size(0) == 1:
                clinical_data = clinical_data.expand(x.size(0), -1)
        
        # Now concatenate
        x = torch.cat([x, clinical_data], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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


def load_clip_model(device):
    try:
        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        model.to(device)
        print("Successfully loaded CLIP model")
        return processor, model
    except OSError as e:
        print(f"Error loading CLIP model: {e}")
        raise


def load_llm_model(device):
    try:
        print("Loading language model...")
        model_name = "meta-llama/Llama-3.2-3B"  # Ensure this model is accessible
        hf_token = "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"
        
        # Load tokenizer with authentication token
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        
        # Load the language model for causal language modeling with authentication token
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
        model.to(device)
        
        # Create the pipeline without passing `use_auth_token` again
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,  # Adjust based on GPU capabilities
            device=0 if torch.cuda.is_available() else -1,
            # Removed `use_auth_token` to prevent it from being passed during generation
        )
        
        print("Successfully loaded LLM model")
        return tokenizer, pipe
    except ImportError as e:
        print(f"Error loading LLM model: {e}")
        print("Please install 'accelerate' by running: pip install 'accelerate>=0.26.0'")
        raise
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        raise


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
            
        # Calculate report length
        metrics['avg_length'].append(len(gen_report.split()))
        
        # Tokenize
        reference = nltk.word_tokenize(ref_report.lower())
        hypothesis = nltk.word_tokenize(gen_report.lower())
        
        # Calculate BLEU scores
        try:
            metrics['bleu_1'].append(sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0), smoothing_function=smooth_fn))
            metrics['bleu_2'].append(sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn))
            metrics['bleu_3'].append(sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn))
            metrics['bleu_4'].append(sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn))
        except Exception as e:
            print(f"Error calculating BLEU scores: {str(e)}")
        
        # Calculate ROUGE scores
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
    
    # Calculate averages
    final_metrics = {}
    for key in metrics:
        if metrics[key]:
            final_metrics[key] = np.mean(metrics[key])
    
    return final_metrics


def train_clip_model(model, train_dataloader, device, epochs=5):
    print("\nTraining CLIP model:")
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch [{epoch+1}/{epochs}]')
        
        for batch_idx, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=True
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")


def main():
    # Force CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Create a generator for CPU (DataLoader requirement)
    g = torch.Generator()
    g.manual_seed(42)

    XAMI_MIMIC_PATH = os.path.join(
        'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0',
        'mimic-eye'
    ).replace('\\', '/')
    MIMIC_EYE_PATH = XAMI_MIMIC_PATH

    # Load the dataset
    df = pd.read_csv('test.csv')
    df = df[df['eye_tracking_data_discarded'] == False]

    # Define critical columns and drop rows with missing values in these columns
    critical_columns = [
        'image_path', 'bbox_path', 'fixation_path', 'report_path', 'reflacx_transcript_path',
        'eyegaze_transcript_path', 'temperature', 'heartrate', 'resprate',
        'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'chiefcomplaint', 'age'
    ]
    df = df.dropna(subset=critical_columns)

    # Update path columns
    for path_col in [
        'image_path', 'bbox_path', 'fixation_path', 'report_path',
        'reflacx_transcript_path', 'eyegaze_transcript_path'
    ]:
        if path_col in df.columns:
            df[path_col] = df[path_col].str.replace(
                '{XAMI_MIMIC_PATH}', XAMI_MIMIC_PATH
            ).str.replace('\\', '/')

    # Specific replacements for certain columns
    for col in ['report_path', 'reflacx_transcript_path', 'eyegaze_transcript_path']:
        if col in df.columns:
            df[col] = df[col].str.replace(
                'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye',
                MIMIC_EYE_PATH,
                regex=False
            )

    # Clinical columns
    clinical_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'age']

    # Handle missing clinical data and normalize
    df[clinical_cols] = df[clinical_cols].astype(float)
    df[clinical_cols] = df[clinical_cols].fillna(df[clinical_cols].mean())
    scaler = StandardScaler()
    df[clinical_cols] = scaler.fit_transform(df[clinical_cols])

    # Initialize CLIP model
    processor, clip_model = load_clip_model(device)

    # Initialize the dataset and dataloader for training CLIP
    train_dataset = ImageTextDataset(df, processor, device)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        generator=g
    )

    # Train CLIP model
    train_clip_model(clip_model, train_dataloader, device, epochs=5)

    # Extract features
    print("\nExtracting features...")
    unified_embeddings = []
    
    # Initialize ClinicalDataEncoder and DiseaseClassifier for feature extraction
    clinical_encoder = ClinicalDataEncoder(input_dim=len(clinical_cols)).to(device)
    clinical_encoder.eval()
    classifier_model = DiseaseClassifier(input_dim=512, clinical_dim=128, num_classes=8).to(device)
    classifier_model.eval()
    
    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Embeddings"):
            image_path = row['image_path']
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                image = Image.new('RGB', (224, 224))
            
            report_text = read_text_file(row['report_path'])
            reflacx_text = read_text_file(row['reflacx_transcript_path'])
            eyegaze_text = read_text_file(row['eyegaze_transcript_path'])
            combined_text = ' '.join([report_text, reflacx_text, eyegaze_text])
            
            inputs = processor(
                text=combined_text,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            
            outputs = clip_model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            unified_embedding = (image_embeds + text_embeds) / 2
            unified_embedding = unified_embedding / unified_embedding.norm(dim=1, keepdim=True)
            
            # Get clinical data
            clinical_data = row[clinical_cols].astype(np.float32).values
            clinical_data = torch.tensor(clinical_data)
            
            unified_embeddings.append({
                'id': row['id'],
                'unified_embedding': unified_embedding.cpu(),
                'labels': row,
                'fixation_path': row['fixation_path'],
                'clinical_data': clinical_data
            })
    
    print(f"Extracted {len(unified_embeddings)} unified embeddings.")

    # Define disease label columns
    disease_labels = [
        'Atelectasis', 'Pneumothorax', 'Consolidation', 'Pulmonary edema',
        'Enlarged cardiac silhouette', 'Fibrosis', 'Fracture', 'Support devices'
    ]
    print(f"Available disease labels: {disease_labels}")

    # Process labels
    for entry in unified_embeddings:
        labels = []
        row_data = entry['labels']
        for label in disease_labels:
            # Convert to float and handle any missing or invalid values
            try:
                value = float(row_data[label]) if pd.notna(row_data[label]) else 0.0
            except (ValueError, KeyError):
                value = 0.0
                print(f"Warning: Missing label {label} for ID {entry['id']}, setting to 0.0")
            labels.append(value)
        entry['labels'] = torch.tensor(labels, dtype=torch.float32)

    # Split data into training and validation sets
    total_length = len(unified_embeddings)
    train_length = int(0.8 * total_length)
    val_length = total_length - train_length

    train_data, val_data = random_split(unified_embeddings, [train_length, val_length], generator=torch.Generator().manual_seed(42))

    # Create datasets
    train_dataset_cls = DiseaseClassificationDataset(train_data)
    val_dataset_cls = DiseaseClassificationDataset(val_data)

    # Create dataloaders
    train_dataloader_cls = DataLoader(
        train_dataset_cls,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        generator=g
    )

    val_dataloader_cls = DataLoader(
        val_dataset_cls,
        batch_size=16,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        generator=g
    )

    # Initialize models
    clinical_encoder = ClinicalDataEncoder(input_dim=len(clinical_cols)).to(device)
    classifier_model = DiseaseClassifier(input_dim=512, clinical_dim=128, num_classes=len(disease_labels)).to(device)

    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer_cls = optim.Adam(
        list(classifier_model.parameters()) + list(clinical_encoder.parameters()), 
        lr=1e-4
    )
    epochs_cls = 5

    print("\nTraining Disease Classifier:")
    for epoch in range(epochs_cls):
        classifier_model.train()
        clinical_encoder.train()
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_dataloader_cls, desc=f'Disease Classification Epoch [{epoch+1}/{epochs_cls}]')
        
        for batch_idx, (embeddings, clinical_data, labels) in enumerate(progress_bar):
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
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f"Disease Classification Epoch [{epoch+1}/{epochs_cls}], Average Loss: {avg_loss:.4f}")
        
        # Validation
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
                
                # Calculate validation loss
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
        
        print(f"\nValidation Metrics:")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        # Per-disease metrics
        aurocs = []
        f1s = []
        precisions = []
        recalls = []
        
        for i, disease in enumerate(disease_labels):
            try:
                # Check if we have both classes present
                if len(np.unique(all_labels[:, i])) > 1:
                    # AUROC
                    auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                    aurocs.append(auroc)
                else:
                    auroc = None
                    aurocs.append(0.5)  # Default value for single-class case
                
                # Convert probabilities to predictions for this disease
                disease_preds = (all_probs[:, i] >= 0.5).astype(int)
                disease_labels_true = all_labels[:, i]
                
                # Calculate metrics for this disease
                f1 = f1_score(disease_labels_true, disease_preds, average='weighted', zero_division=0)
                precision = precision_score(disease_labels_true, disease_preds, average='weighted', zero_division=0)
                recall = recall_score(disease_labels_true, disease_preds, average='weighted', zero_division=0)
                
                f1s.append(f1)
                precisions.append(precision)
                recalls.append(recall)
                
                print(f"\n{disease}:")
                print(f"AUROC: {auroc if auroc is not None else 'N/A (single class)'}")
                print(f"F1-Score: {f1:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                
            except Exception as e:
                print(f"\nError calculating metrics for {disease}: {str(e)}")
                aurocs.append(0.5)
                f1s.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
        
        # Overall metrics (macro averaging)
        macro_auroc = np.mean(aurocs)
        macro_f1 = np.mean(f1s)
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        
        # Convert all probabilities to predictions for accuracy
        all_preds = (all_probs >= 0.5).astype(int)
        accuracy = accuracy_score(all_labels.ravel(), all_preds.ravel())
        
        print(f"\nOverall Metrics:")
        print(f"Macro AUROC: {macro_auroc:.4f}")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

    # Load the LLM model correctly
    print("\nLoading LLM model...")
    try:
        tokenizer_llm, llm_model = load_llm_model(device)
    except Exception as e:
        print("Failed to load LLM model. Exiting.")
        return

    # Ensure the pipeline is in evaluation mode
    llm_model.model.eval()

    # Generate reports
    print("\nGenerating reports...")
    reports = []
    reference_reports = []
    
    example_report = """
    Example Report:
    Chest X-ray Findings:
    1. Heart size: Mildly enlarged cardiac silhouette with cardiothoracic ratio of 0.55
    2. Lung fields: Clear lung fields bilaterally without focal consolidation
    3. Pleural spaces: No pleural effusion or pneumothorax
    4. Osseous structures: No acute fracture or dislocation
    5. Support devices: None visualized
    
    Impression:
    Mild cardiomegaly without other acute cardiopulmonary findings."""
    
    prompt_template = """You are an experienced radiologist analyzing a chest X-ray. Generate a detailed medical report based on the following disease probabilities and clinical data. Your report should reflect the actual findings from these probabilities, NOT the example format below.
    
Disease Probabilities (IMPORTANT - Base your findings on these values):
{}

Clinical Data (Use this information to provide context):
{}

Key Instructions:
1. Focus on diseases with high probabilities (>50%) first
2. For each significant finding, describe:
   - Specific location and distribution
   - Severity level
   - Detailed characteristics
3. Address each abnormal finding from the disease probabilities
4. Comment on normal findings only if relevant
5. Provide a concise impression summarizing key abnormalities

The report should follow this structure, but with YOUR OWN findings based on the probabilities above:

EXAMPLE FORMAT (DO NOT COPY THE CONTENT, ONLY THE STRUCTURE):
{}

IMPORTANT: Generate a unique report based on the actual disease probabilities above. Do not copy the example findings."""

    for entry in tqdm(unified_embeddings, desc="Generating Reports"):
        print(f"\nGenerated report for ID {entry['id']}:")
        print("-" * 50)
        
        try:
            # Get disease predictions
            classifier_model.eval()
            clinical_encoder.eval()
            with torch.no_grad():
                embedding = entry['unified_embedding'].to(device)
                clinical_data = entry['clinical_data'].to(device)
                clinical_embedding = clinical_encoder(clinical_data.unsqueeze(0))
                logits = classifier_model(embedding.unsqueeze(0), clinical_embedding)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Format disease probabilities
            disease_probs = dict(zip(disease_labels, probs))
            disease_prob_str = "\n".join([f"{disease}: {prob:.2%}" for disease, prob in disease_probs.items()])
            
            # Format clinical data
            clinical_data_np = entry['clinical_data'].cpu().numpy()
            clinical_data_dict = dict(zip(clinical_cols, clinical_data_np))
            clinical_data_str = "\n".join([f"{col}: {val:.2f}" for col, val in clinical_data_dict.items()])
            
            # Debug prints
            print("\nDisease Probabilities:")
            print(disease_prob_str)
            print("\nClinical Data:")
            print(clinical_data_str)
            
            # Create prompt
            prompt = prompt_template.format(disease_prob_str, clinical_data_str, example_report)
            
            # Generate report using pipeline
            sequences = llm_model(
                prompt,
                do_sample=True,
                temperature=0.8,  # Adjusted temperature for better sampling
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
                max_new_tokens=256,
                repetition_penalty=1.2,
                pad_token_id=tokenizer_llm.pad_token_id,
                eos_token_id=tokenizer_llm.eos_token_id,
                # No `use_auth_token` passed here
            )
            
            # Extract generated text
            generated_text = sequences[0]['generated_text']
            
            # Remove the prompt from the generated text to get just the report
            report = generated_text[len(prompt):].strip()
            
            # Post-processing: Remove incomplete sentences and template markers
            report = report.replace("[/INST]", "").strip()
            sentences = nltk.sent_tokenize(report)
            if sentences:
                report = ' '.join(sentences)
            else:
                report = report.strip()
            
            print(report)
            print("-" * 50)
            
            reports.append({
                'id': entry['id'],
                'report': report
            })
            
            # Get the reference report
            reference_report = read_text_file(entry['labels']['report_path'])
            reference_reports.append(reference_report)
            
        except Exception as e:
            print(f"Error generating report for ID {entry['id']}: {str(e)}")
            continue

    print(f"\nSuccessfully generated {len(reports)} reports")

    print("\nReport Generation Metrics:")
    generated_texts = [report['report'] for report in reports]
    reference_texts = [report for report in reference_reports]
    text_metrics = calculate_text_generation_metrics(generated_texts, reference_texts)
    for metric_name, value in text_metrics.items():
        print(f"{metric_name}: {value:.4f}")


if __name__ == '__main__':
    main()