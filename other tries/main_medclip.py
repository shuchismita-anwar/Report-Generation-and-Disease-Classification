import os
import pandas as pd
import torch
from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPTextModel
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import nltk
import warnings
from torchvision import transforms
from transformers import AutoTokenizer

# Download required NLTK data
nltk.download('punkt')
warnings.filterwarnings('ignore')

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

# Define image transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def load_preprocess_image(path, device):
    # Convert from grayscale ('L') to RGB by replicating channels if necessary
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        img = Image.new('RGB', (224, 224))
    img = image_transform(img)
    return img.to(device)  # Removed unsqueeze(0)

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

class ImageTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_transform, medclip_model, device):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.medclip_model = medclip_model
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

        # Tokenize text
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
        embedding = self.data[idx]['unified_embedding']
        labels = self.data[idx]['labels']
        clinical_data = self.data[idx]['clinical_data'].float()
        return embedding, clinical_data, labels

def extract_features(dataset, dataloader, medclip_model, device):
    unified_embeddings = []
    clinical_data = dataset.data[['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'age']].values.astype(np.float32)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            images, input_ids, attention_mask = batch
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Extract image features
            image_features = medclip_model.vision_model(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Extract text features
            text_features = medclip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Combine embeddings
            unified_embedding = (image_features + text_features) / 2
            unified_embedding = unified_embedding / unified_embedding.norm(dim=-1, keepdim=True)

            # Move embeddings to CPU
            unified_embedding = unified_embedding.cpu()

            # Adjust index for the current batch
            start_idx = batch_idx * dataloader.batch_size
            end_idx = start_idx + unified_embedding.size(0)

            for i in range(unified_embedding.size(0)):
                idx = start_idx + i
                if idx >= len(dataset.data):
                    break
                unified_embeddings.append({
                    'id': dataset.data.iloc[idx]['id'],
                    'unified_embedding': unified_embedding[i],
                    'labels': dataset.data.iloc[idx],
                    'fixation_path': dataset.data.iloc[idx]['fixation_path'],
                    'clinical_data': torch.tensor(clinical_data[idx])
                })
    return unified_embeddings

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
        # Concatenate embeddings and clinical data
        x = torch.cat([x, clinical_data], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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
    df = pd.read_csv('test22222.csv')
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

    # Initialize MedCLIPModel with pretrained weights
    try:
        medclip_model = MedCLIPModel.from_pretrained("flax-community/medclip-roco").to(device)
        print("Successfully loaded MedCLIP models via from_pretrained.")
    except AttributeError:
        # If from_pretrained is not available, initialize differently
        try:
            vision_model = MedCLIPVisionModel()
            text_model = MedCLIPTextModel()

            medclip_model = MedCLIPModel().to(device)  # Adjust based on actual MedCLIPModel's constructor

            print("Successfully initialized MedCLIP models without from_pretrained.")
        except Exception as e:
            print(f"Error initializing MedCLIP models: {e}")
            raise e

    diseases = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
                "Pneumonia", "Pneumothorax", "Pleural Effusion", "No Finding"]

    # Initialize the dataset and dataloader for feature extraction
    feature_dataset = ImageTextDataset(df, tokenizer, image_transform, medclip_model, device)
    feature_dataloader = DataLoader(
        feature_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4  # Adjust based on your system
    )

    # Extract features
    unified_embeddings = extract_features(feature_dataset, feature_dataloader, medclip_model, device)

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
    train_dataset = DiseaseClassificationDataset(train_data)
    val_dataset = DiseaseClassificationDataset(val_data)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        generator=g,  # CPU generator
        num_workers=4  # Adjust based on your system
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4  # Adjust based on your system
    )

    clinical_input_dim = len(clinical_cols)
    clinical_output_dim = 128

    classifier_model = DiseaseClassifier(input_dim=512, clinical_dim=clinical_output_dim, num_classes=len(disease_labels)).to(device)
    clinical_encoder = ClinicalDataEncoder(input_dim=clinical_input_dim, output_dim=clinical_output_dim).to(device)

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
        progress_bar = tqdm(train_dataloader, desc=f'Disease Classification Epoch [{epoch+1}/{epochs_cls}]')

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
            for embeddings, clinical_data, labels in val_dataloader:
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

if __name__ == '__main__':
    main()
