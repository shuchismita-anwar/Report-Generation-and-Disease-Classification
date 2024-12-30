import os
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel, LlamaTokenizer, LlamaForCausalLM
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
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
)
MIMIC_EYE_PATH = XAMI_MIMIC_PATH

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

df['report_path'] = df['report_path'].str.replace(
    'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye',
    MIMIC_EYE_PATH.replace('\\', '/'),
    regex=False
)
df['reflacx_transcript_path'] = df['reflacx_transcript_path'].str.replace(
    'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye',
    MIMIC_EYE_PATH.replace('\\', '/'),
    regex=False
)
df['eyegaze_transcript_path'] = df['eyegaze_transcript_path'].str.replace(
    'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0\\mimic-eye',
    MIMIC_EYE_PATH.replace('\\', '/'),
    regex=False
)

# Clinical columns
clinical_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'age']

# Handle missing clinical data and normalize
df[clinical_cols] = df[clinical_cols].astype(float)
df[clinical_cols] = df[clinical_cols].fillna(df[clinical_cols].mean())
scaler = StandardScaler()
df[clinical_cols] = scaler.fit_transform(df[clinical_cols])

def load_clip_model(device):
    import os
    from transformers import CLIPProcessor, CLIPModel
    
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    
    model.to(device)
    print("Successfully loaded CLIP model")
    
    return processor, model

processor, clip_model = load_clip_model(device)

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
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)
        self.processor = processor
    
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
            inputs[k] = inputs[k].squeeze(0)
        
        return inputs

def train_clip_model(model, train_dataloader, epochs=5):
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

train_dataset = ImageTextDataset(df)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True,
    generator=g,  # CPU generator
    worker_init_fn=lambda x: torch.manual_seed(42)
)

train_clip_model(clip_model, train_dataloader)

clip_model.eval()
unified_embeddings = []

with torch.no_grad():
    for index, row in df.iterrows():
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
        clinical_data = row[clinical_cols].values.astype(np.float32)
        clinical_data = torch.tensor(clinical_data)
        
        unified_embeddings.append({
            'id': row['id'],
            'unified_embedding': unified_embedding.cpu(),
            'labels': row,
            'fixation_path': row['fixation_path'],
            'clinical_data': clinical_data
        })

first_sample = unified_embeddings[0]
print("Unified embedding shape:", first_sample['unified_embedding'].shape)

# Step 4: Disease Classification

# Get actual disease labels from DataFrame
disease_labels = []
for col in df.columns:
    if col in ['Atelectasis', 'Pneumothorax', 'Consolidation', 'Pulmonary edema',
               'Enlarged cardiac silhouette', 'Fibrosis', 'Fracture', 'Support devices']:
        disease_labels.append(col)

print(f"Available disease labels: {disease_labels}")

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

# Step 5: Clinical Data Integration

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

classification_dataset = DiseaseClassificationDataset(unified_embeddings)
classification_dataloader = DataLoader(
    classification_dataset, 
    batch_size=16, 
    shuffle=True,
    generator=g,  # CPU generator
    worker_init_fn=lambda x: torch.manual_seed(42)
)

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
        x = self.fc(x)
        return x

class DiseaseClassifier(nn.Module):
    def __init__(self, input_dim=512, clinical_dim=128, num_classes=len(disease_labels)):
        super(DiseaseClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + clinical_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, clinical_data):
        x = torch.cat([x, clinical_data], dim=1)
        x = self.fc(x)
        return x

clinical_input_dim = len(clinical_cols)
clinical_output_dim = 128

classifier_model = DiseaseClassifier(input_dim=512, clinical_dim=clinical_output_dim).to(device)
clinical_encoder = ClinicalDataEncoder(input_dim=clinical_input_dim, output_dim=clinical_output_dim).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer_cls = optim.Adam(
    list(classifier_model.parameters()) + list(clinical_encoder.parameters()), 
    lr=1e-4
)
epochs_cls = 5

print("\nTraining Disease Classifier:")
classifier_model.train()
clinical_encoder.train()
for epoch in range(epochs_cls):
    total_loss = 0
    num_batches = 0
    progress_bar = tqdm(classification_dataloader, desc=f'Disease Classification Epoch [{epoch+1}/{epochs_cls}]')
    
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

# Step 6: Report Generation

def load_llm_model(device):
    try:
        print("Loading language model...")
        # Using TinyLlama as it's more accessible and lighter
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("Successfully loaded LLM model")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading LLM model: {str(e)}")
        raise  # Re-raise the exception to see the full error

# Load the LLM model
print("Loading LLM model...")
tokenizer, llm_model = load_llm_model(device)

llm_model.eval()

# Generate reports
print("\nGenerating reports...")
reports = []

for entry in unified_embeddings:
    # Get disease predictions
    classifier_model.eval()
    clinical_encoder.eval()
    with torch.no_grad():
        embedding = entry['unified_embedding'].squeeze(0).to(device)
        clinical_data = entry['clinical_data'].unsqueeze(0).to(device)
        clinical_embedding = clinical_encoder(clinical_data)
        logits = classifier_model(embedding.unsqueeze(0), clinical_embedding)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Map probabilities to disease labels
    disease_probs = dict(zip(disease_labels, probs))
    
    # Prepare prompt
    prompt = "<s>[INST] As a radiologist, generate a brief medical report based on the following information:\n\n"
    prompt += "Clinical Data:\n"
    clinical_data = entry['clinical_data'].numpy()
    for i, col in enumerate(clinical_cols):
        value = clinical_data[i]
        prompt += f"- {col}: {value:.2f}\n"
    
    prompt += "\nDisease Probabilities:\n"
    for disease, prob in disease_probs.items():
        prompt += f"- {disease}: {prob:.2f}\n"
    
    prompt += "\nPlease provide a concise radiology report including key findings and impressions. [/INST]"

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        # Generate report
        output_sequences = llm_model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        
        # Decode generated text
        generated_report = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # Clean up the report by extracting only the model's response
        if "[/INST]" in generated_report:
            generated_report = generated_report.split("[/INST]")[1].strip()
        
        reports.append({
            'id': entry['id'],
            'report': generated_report
        })

        print(f"\nGenerated report for ID {entry['id']}:")
        print("-" * 50)
        print(generated_report)
        print("-" * 50)
    
    except Exception as e:
        print(f"Error generating report for ID {entry['id']}: {str(e)}")
        continue

print(f"\nSuccessfully generated {len(reports)} reports")
