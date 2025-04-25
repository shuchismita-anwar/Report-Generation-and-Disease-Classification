import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Suppress symlink warning

# Filter all warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('timm').setLevel(logging.ERROR)

import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
# from health_multimodal.image.transforms import create_chest_xray_transform_for_inference
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
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

# GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_optimal_batch_size(model, image_size=(224, 224)):
    """Simple default batch size for BioViL-T"""
    return 4

class XRayReportDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_transform, max_len=77, augment=False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.transform = image_transform
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
        img_path = normalize_path(row['image_path'].replace('{XAMI_MIMIC_PATH}', XAMI_MIMIC_PATH))
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        logger.debug(f"[DEBUG] Image mode: {image.mode}, size: {image.size}, path: {img_path}")
        if self.augment:
            image = self.augment_transforms(image)
        image_tensor = self.transform(image)
        logger.debug(f"[DEBUG] Image tensor shape after transform: {image_tensor.shape}")

        # load transcript
        tp = row['transcript_path']
        with open(tp, 'r', encoding='utf-8') as f:
            report_text = json.load(f).get("full_text", None) if tp.lower().endswith('.json') else f.read()

        text_enc = self.tokenizer(
            report_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': text_enc.input_ids.squeeze(0),
            'attention_mask': text_enc.attention_mask.squeeze(0),
            'pixel_values': image_tensor,
            'labels': self.labels[index]
        }

class XRayReportModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        model_name = "microsoft/BiomedVLP-BioViL-T"
        self.biol = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        # projection_dim is 512 by default
        hidden_size = self.biol.config.projection_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, pixel_values):
        # text embeddings
        text_feats = self.biol.get_projected_text_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # image embeddings
        if hasattr(self.biol, "get_image_features"):
            img_feats = self.biol.get_image_features(pixel_values=pixel_values)
        elif hasattr(self.biol, "get_projected_image_embeddings"):
            img_feats = self.biol.get_projected_image_embeddings(pixel_values=pixel_values)
        else:
            if not hasattr(self, "_warned_image_embed"):
                logger.warning("[WARNING] CXRBertModel has no image projection – using zeros.")
                self._warned_image_embed = True
            batch_size = pixel_values.shape[0]
            hidden_size = text_feats.shape[1]
            img_feats = torch.zeros((batch_size, hidden_size), device=pixel_values.device, dtype=text_feats.dtype)
        combined = torch.cat([img_feats, text_feats], dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits

def train_model(model, train_loader, val_loader, device, pos_weights, num_epochs=1000, patience=10):
    pos_weights = pos_weights.to(device)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    best_val_f1 = 0.0
    patience_counter = 0
    all_results = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        prog = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in prog:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            imgs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(ids, mask, imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            # Fix: detach before calling numpy to avoid RuntimeError
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            prog.set_postfix({'loss': loss.item()})

        # validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                imgs = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                with torch.cuda.amp.autocast():
                    logits = model(ids, mask, imgs)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                # Fix: detach before calling numpy to avoid RuntimeError
                val_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        train_metrics = calculate_metrics(np.array(train_labels), np.array(train_preds))
        val_metrics   = calculate_metrics(np.array(val_labels),   np.array(val_preds))

        logger.info(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val F1 {val_metrics["f1"]:.4f}')
        logger.info(f'Train Metrics: {train_metrics}')
        logger.info(f'Val   Metrics: {val_metrics}')

        scheduler.step(val_metrics['f1'])
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
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
            'val_metrics': val_metrics
        })

    # summary
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
    # per-label (hamming) accuracy
    acc = float((preds_b == labels).mean())
    return {
        'accuracy': acc,
        'precision': precision_score(labels, preds_b, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds_b, average='weighted', zero_division=0),
        'f1': f1_score(labels, preds_b, average='weighted', zero_division=0),
        'roc_auc': roc
    }

def predict(model, image_path, tokenizer, image_transform, device):
    model.eval()
    image = Image.open(normalize_path(image_path)).convert('L')  # Convert to grayscale
    img_tensor = image_transform(image).unsqueeze(0).to(device)
    text = "inference"
    text_enc = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        logits = model(text_enc.input_ids, text_enc.attention_mask, img_tensor)
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    return {CONDITIONS[i]: float(probs[i]) for i in range(len(CONDITIONS))}, int(np.argmax(probs))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # load labels & build df
    labels_df = pd.read_csv('condition_labels.csv')
    labels_df['conditions'] = labels_df['conditions'].fillna('')
    records = []
    for patient_dir in os.listdir(XAMI_MIMIC_PATH):
        patient_path = os.path.join(XAMI_MIMIC_PATH, patient_dir)
        if os.path.isdir(patient_path) and patient_dir.startswith('patient_'):
            for root, _, files in os.walk(patient_path):
                imgs = [f for f in files if f.lower().endswith(('.jpg','.jpeg','.png'))]
                txts = [f for f in files if f.lower().endswith(('.txt','.json'))]
                if imgs and txts:
                    image_path = os.path.join(root, imgs[0])
                    transcript_path = os.path.join(root, txts[0])
                    records.append({
                        'image_path': image_path,
                        'transcript_path': os.path.relpath(transcript_path, os.getcwd()).replace('/', '\\')
                    })
    logger.info(f"[DEBUG] Number of valid (image, transcript) pairs: {len(records)}")
    df = pd.DataFrame(records)
    logger.info(f"[DEBUG] Length of df after fix: {len(df)}")
    df = pd.merge(df, labels_df, on='transcript_path', how='inner')
    if df.empty:
        logger.warning("[WARNING] DataFrame is empty after merging with labels_df.")
        return

    # labels array
    labels_list = []
    for conds in df['conditions']:
        cl = conds.split('|') if conds else []
        labels_list.append([1 if c in cl else 0 for c in CONDITIONS])
    df['labels'] = labels_list

    pos_counts = np.sum(np.array(labels_list), axis=0)
    total = len(df)
    pos_weights = torch.tensor([total/(c+1e-5) for c in pos_counts], dtype=torch.float32)

    train_val, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df  = train_test_split(train_val, test_size=0.2, random_state=42)

    model_name = "microsoft/BiomedVLP-BioViL-T"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    resize = 256
    center_crop_size = 224
    logger.info(f"[DEBUG] Calling create_chest_xray_transform_for_inference with resize={resize}, center_crop_size={center_crop_size}")
    image_transform = create_chest_xray_transform_for_inference(resize, center_crop_size)

    model = XRayReportModel(num_labels=len(CONDITIONS)).to(device)
    batch_size = get_optimal_batch_size(model) if torch.cuda.is_available() else 4

    train_ds = XRayReportDataset(train_df, tokenizer, image_transform, augment=True)
    val_ds   = XRayReportDataset(val_df,   tokenizer, image_transform, augment=False)
    test_ds  = XRayReportDataset(test_df,  tokenizer, image_transform, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=min(4, os.cpu_count()), pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              num_workers=min(4, os.cpu_count()), pin_memory=True)

    train_model(model, train_loader, val_loader, device, pos_weights)

    # simple inference on first 3 test samples
    for i in range(min(3, len(test_df))):
        row = test_df.iloc[i]
        imgp = normalize_path(row['image_path'].replace('{XAMI_MIMIC_PATH}', XAMI_MIMIC_PATH))
        preds, idx = predict(model, imgp, tokenizer, image_transform, device)
        logger.info(f"Sample {i}: Predicted {CONDITIONS[idx]} with scores {preds}")

    logger.info('Done! Model saved as best_xray_model.pt')

if __name__ == '__main__':
    main()
