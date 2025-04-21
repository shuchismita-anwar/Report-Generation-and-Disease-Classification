import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Suppress symlink warning

import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import logging
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
def normalize_path(path):
    """Normalize path for consistent comparison"""
    return os.path.normpath(path).replace('\\', '/')

def normalize_csv_path(path):
    """Normalize path to match CSV format"""
    return os.path.normpath(path).replace('/', '\\')

XAMI_MIMIC_PATH = normalize_path(os.path.join(
    os.getcwd(),
    'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0',
    'mimic-eye'
))

# Conditions
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

# GPU and Memory Optimization
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_optimal_batch_size(model, image_size=(224, 224), max_memory_usage=4*1024*1024*1024):
    """Calculate optimal batch size based on GPU memory."""
    if not torch.cuda.is_available():
        return 4
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    test_batch = 4
    # CLIPModel expects same inputs: input_ids, attention_mask, pixel_values
    test_input_ids = torch.randint(0, model.clip.config.text_config.vocab_size, (test_batch, 77)).cuda()
    test_attention_mask = torch.ones((test_batch, 77)).cuda()
    test_images = torch.randn((test_batch, 3, *image_size)).cuda()

    with torch.no_grad():
        model(input_ids=test_input_ids, attention_mask=test_attention_mask, pixel_values=test_images)

    memory_per_batch = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    optimal_batch = max(1, int(max_memory_usage / (memory_per_batch / test_batch)))
    return min(optimal_batch, 32)

class XRayReportDataset(Dataset):
    def __init__(self, dataframe, processor, max_len=77, augment=False):
        self.data = dataframe
        self.processor = processor
        self.max_len = max_len
        self.augment = augment
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])
        self.labels = torch.tensor(self.data['labels'].tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = normalize_path(row['image_path'].replace('{XAMI_MIMIC_PATH}', XAMI_MIMIC_PATH))
        image = Image.open(image_path).convert('RGB')
        if self.augment:
            image = self.augment_transforms(image)

        # Load text
        if "EyeGaze" in row['transcript_path']:
            with open(row['transcript_path'], 'r', encoding='utf-8') as f:
                report_text = json.load(f)["full_text"]
        else:
            with open(row['transcript_path'], 'r', encoding='utf-8') as f:
                report_text = f.read()

        # Use one processor call for both modalities
        encoding = self.processor(
            text=report_text,
            images=image,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'labels': self.labels[index]
        }

class XRayReportModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # Load CLIP
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        # projection_dim is 512 by default
        hidden_size = self.clip.config.projection_dim
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
        self.last_hidden_state = None

    def forward(self, input_ids, attention_mask, pixel_values):
        # Get CLIP outputs with hidden states
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
            output_hidden_states=True  # <-- Enable hidden states
        )
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        # Extract vision hidden states for Grad-CAM
        # outputs.vision_model_output.hidden_states is a tuple: (layer1, ..., final_layer)
        if hasattr(outputs, 'vision_model_output') and outputs.vision_model_output is not None:
            # Get the last hidden state from vision model output
            vision_hidden_states = outputs.vision_model_output.hidden_states
            if vision_hidden_states is not None and len(vision_hidden_states) > 0:
                self.last_hidden_state = vision_hidden_states[-1]
                if self.last_hidden_state.requires_grad:
                    self.last_hidden_state.retain_grad()
            else:
                self.last_hidden_state = None
        else:
            self.last_hidden_state = None
        combined = torch.cat([image_features, text_features], dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits

def train_model(model, train_loader, val_loader, test_loader, device, pos_weights, num_epochs=100, patience=5):
    pos_weights = pos_weights.to(device)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_val_loss = float('inf')
    patience_counter = 0
    all_results = []

    for epoch in range(num_epochs):
        # Training loop as before...
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        prog = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in prog:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask, pixel_values)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            prog.set_postfix({'loss': loss.item()})

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask, pixel_values)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Metrics + logging (unchanged)...
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_metrics = calculate_metrics(np.array(train_labels), np.array(train_preds))
        val_metrics = calculate_metrics(np.array(val_labels), np.array(val_preds))
        train_pc = calculate_per_class_metrics(np.array(train_labels), np.array(train_preds))
        val_pc = calculate_per_class_metrics(np.array(val_labels), np.array(val_preds))

        logger.info(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}')
        logger.info(f'Train Metrics: {train_metrics}')
        logger.info(f'Val   Metrics: {val_metrics}')

        # Early stopping, saving, and cleanup as before...
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_xray_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break
        torch.cuda.empty_cache()

        all_results.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_per_class': train_pc,
            'val_per_class': val_pc
        })

    # Save results summary (unchanged)...
    with open('results.txt', 'w') as f:
        f.write('Training Results Summary\n=======================\n\n')
        for r in all_results:
            f.write(f"Epoch {r['epoch']}\n")
            f.write(f" Train Loss: {r['train_loss']:.4f}, Val Loss: {r['val_loss']:.4f}\n")
            f.write(f" Train Metrics: {r['train_metrics']}\n")
            f.write(f" Val   Metrics: {r['val_metrics']}\n\n")

    return model.state_dict()

def calculate_metrics(labels, preds, threshold=0.5):
    preds_b = (preds >= threshold).astype(int)
    try:
        roc = roc_auc_score(labels, preds, average='weighted')
    except ValueError:
        roc = float('nan')
    return {
        'accuracy': accuracy_score(labels, preds_b),
        'precision': precision_score(labels, preds_b, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds_b, average='weighted', zero_division=0),
        'f1': f1_score(labels, preds_b, average='weighted', zero_division=0),
        'roc_auc': roc
    }

def calculate_per_class_metrics(labels, preds, threshold=0.5):
    preds_b = (preds >= threshold).astype(int)
    metrics = {}
    for i, cond in enumerate(CONDITIONS):
        metrics[cond] = {
            'precision': precision_score(labels[:, i], preds_b[:, i], zero_division=0),
            'recall': recall_score(labels[:, i], preds_b[:, i], zero_division=0),
            'f1': f1_score(labels[:, i], preds_b[:, i], zero_division=0)
        }
    return metrics

def predict(model, image_path, device):
    model.eval()
    processor = CLIPProcessor.from_pretrained('flaviagiammarino/pubmed-clip-vit-base-patch32')
    image = Image.open(normalize_path(image_path)).convert('RGB')
    encoding = processor(
        text="inference",
        images=image,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device),
            pixel_values=encoding['pixel_values'].to(device)
        )
    probs = torch.sigmoid(outputs).cpu().numpy()[0]
    preds = {CONDITIONS[i]: float(probs[i]) for i in range(len(CONDITIONS))}
    return preds, int(np.argmax(probs))

def generate_gradcam(model, image_path, condition_index, device):
    model.eval()
    processor = CLIPProcessor.from_pretrained('flaviagiammarino/pubmed-clip-vit-base-patch32')
    image = Image.open(normalize_path(image_path)).convert('RGB')
    encoding = processor(images=image, return_tensors='pt')
    pixel_values = encoding['pixel_values'].to(device)
    # dummy text for full forward
    text_encoding = processor(text="inference", return_tensors='pt', padding=True, truncation=True)
    input_ids = text_encoding['input_ids'].to(device)
    attention_mask = text_encoding['attention_mask'].to(device)

    # Forward to get logits & save last vision hidden state
    logits = model(input_ids, attention_mask, pixel_values)
    if model.last_hidden_state is None:
        raise RuntimeError("Vision hidden states not available for Grad-CAM")
    # model.last_hidden_state.retain_grad()  # Now done in forward
    score = logits[0, condition_index]
    model.zero_grad()
    score.backward()

    grads = model.last_hidden_state.grad[:, 1:, :]  # skip class token
    weights = grads.mean(dim=2)
    n_patches = int(np.sqrt(weights.shape[1]))
    cam = nn.functional.relu(weights).view(1, n_patches, n_patches)
    cam = nn.functional.interpolate(cam.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def plot_gradcam(image_path, cam, title='Grad-CAM'):
    image = Image.open(normalize_path(image_path)).convert('RGB').resize((224, 224))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image); plt.title('Original'); plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image); plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(title); plt.axis('off')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load labels and build df (unchanged)...
    labels_df = pd.read_csv('condition_labels.csv')
    labels_df['conditions'] = labels_df['conditions'].fillna('')
    df = pd.DataFrame(columns=['image_path','transcript_path'])
    image_paths, transcript_paths = [], []

    # Walk through patient directories to collect image and transcript paths
    for patient_dir in os.listdir(XAMI_MIMIC_PATH):
        patient_path = os.path.join(XAMI_MIMIC_PATH, patient_dir)
        if os.path.isdir(patient_path) and patient_dir.startswith('patient_'):
            for root, dirs, files in os.walk(patient_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(root, file)
                        # Try to find a matching transcript for this image
                        # Assumes transcript has similar naming or is in the same folder
                        transcript = None
                        for tf in files:
                            if tf.lower().endswith('.txt') or tf.lower().endswith('.json'):
                                transcript = os.path.join(root, tf)
                                break
                        if transcript:
                            # Normalize transcript path to match CSV (relative, backslashes)
                            rel_transcript = os.path.relpath(transcript, os.getcwd())
                            rel_transcript = rel_transcript.replace('/', '\\')
                            image_paths.append(image_path)
                            transcript_paths.append(rel_transcript)
    logger.info(f"Found {len(image_paths)} images and {len(transcript_paths)} transcripts.")
    if transcript_paths:
        logger.info(f"Sample normalized transcript path: {transcript_paths[0]}")

    df['image_path'] = image_paths
    df['transcript_path'] = transcript_paths
    df = pd.merge(df, labels_df, on='transcript_path', how='inner')
    if df.empty:
        logger.error("No data found."); return

    # Create labels array
    labels_list = []
    for conds in df['conditions']:
        cl = conds.split('|') if conds else []
        labels_list.append([1 if c in cl else 0 for c in CONDITIONS])
    df['labels'] = labels_list

    pos_counts = np.sum(np.array(labels_list), axis=0)
    total = len(df)
    pos_weights = torch.tensor([total/(count+1e-5) for count in pos_counts], dtype=torch.float32)

    train_val, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df  = train_test_split(train_val, test_size=0.2, random_state=42)

    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model = XRayReportModel(num_labels=len(CONDITIONS)).to(device)

    batch_size = get_optimal_batch_size(model) if torch.cuda.is_available() else 4
    train_ds = XRayReportDataset(train_df, processor, augment=True)
    val_ds   = XRayReportDataset(val_df, processor, augment=False)
    test_ds  = XRayReportDataset(test_df, processor, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count()), pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=min(4, os.cpu_count()), pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, num_workers=min(4, os.cpu_count()), pin_memory=True)

    train_model(model, train_loader, val_loader, test_loader, device, pos_weights)

    # Grad‑CAM on first 3 test samples
    for i in range(min(3, len(test_df))):
        row = test_df.iloc[i]
        imgp = normalize_path(row['image_path'].replace('{XAMI_MIMIC_PATH}', XAMI_MIMIC_PATH))
        preds, idx = predict(model, imgp, device)
        cam = generate_gradcam(model, imgp, idx, device)
        plot_gradcam(imgp, cam, title=f'Grad-CAM: {CONDITIONS[idx]}')

    logger.info('Done! Model saved as best_xray_model.pt')

if __name__ == '__main__':
    main()
