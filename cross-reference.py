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
from transformers import BertTokenizer, BertModel, ViTModel, ViTImageProcessor
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
MIMIC_EYE_PATH = XAMI_MIMIC_PATH

# Conditions
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

# GPU and Memory Optimization Settings
torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark mode
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True  # Enable TF32 on Ampere GPUs

def get_optimal_batch_size(model, image_size=(224, 224), max_memory_usage=4*1024*1024*1024):  # 4GB target
    """Calculate optimal batch size based on GPU memory."""
    if not torch.cuda.is_available():
        return 4  # Default batch size for CPU
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Test with a small batch
    test_batch = 4
    test_input_ids = torch.randint(0, 1000, (test_batch, 512)).cuda()
    test_attention_mask = torch.ones((test_batch, 512)).cuda()
    test_images = torch.randn((test_batch, 3, *image_size)).cuda()
    
    with torch.no_grad():
        model(test_input_ids, test_attention_mask, test_images)
    
    memory_per_batch = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    
    optimal_batch = max(1, int(max_memory_usage / (memory_per_batch / test_batch)))
    return min(optimal_batch, 32)  # Cap at 32 to avoid too large batches

# Dataset Class
class XRayReportDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_processor, max_len=512, augment=False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len
        self.augment = augment
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])
        
        # Pre-compute labels as tensors for efficiency
        self.labels = torch.tensor(self.data['labels'].tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = normalize_path(row['image_path'].replace('{XAMI_MIMIC_PATH}', XAMI_MIMIC_PATH))
        
        # Efficient image loading and preprocessing
        image = Image.open(image_path).convert('RGB')
        if self.augment:
            image = self.augment_transforms(image)
        
        # Process image and move to CPU tensor
        pixel_values = self.image_processor(image, return_tensors='pt')['pixel_values'].squeeze(0)
        
        # Efficient text processing
        if "EyeGaze" in row['transcript_path']:
            with open(row['transcript_path'], 'r', encoding='utf-8') as f:
                report_text = json.load(f)["full_text"]
        else:
            with open(row['transcript_path'], 'r', encoding='utf-8') as f:
                report_text = f.read()
        
        # Tokenize text
        encoding = self.tokenizer(
            report_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'pixel_values': pixel_values,
            'labels': self.labels[index]
        }

# Model Class
class XRayReportModel(nn.Module):
    def __init__(self, num_labels):
        super(XRayReportModel, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768 + 768, num_labels)
        self.last_hidden_state = None
    
    def forward(self, input_ids, attention_mask, pixel_values):
        vit_outputs = self.vit(pixel_values=pixel_values)
        self.last_hidden_state = vit_outputs.last_hidden_state
        image_features = vit_outputs.pooler_output
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_outputs.pooler_output
        combined_features = torch.cat((image_features, text_features), dim=1)
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)
        return logits

# Training Function with Class Weights
def train_model(model, train_loader, val_loader, test_loader, device, pos_weights, num_epochs=30, patience=5):
    # Move pos_weights to GPU
    pos_weights = pos_weights.to(device)
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    
    # Store all results for final saving
    all_results = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        # Use tqdm for better progress tracking
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            # Move batch to GPU efficiently
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Use automatic mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask, pixel_values)
                loss = criterion(outputs, labels)
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
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
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Calculate overall metrics
        train_metrics = calculate_metrics(np.array(train_labels), np.array(train_preds))
        val_metrics = calculate_metrics(np.array(val_labels), np.array(val_preds))
        
        # Calculate per-class metrics
        train_per_class = calculate_per_class_metrics(np.array(train_labels), np.array(train_preds))
        val_per_class = calculate_per_class_metrics(np.array(val_labels), np.array(val_preds))
        
        # Store epoch results
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_per_class': train_per_class,
            'val_per_class': val_per_class
        }
        all_results.append(epoch_results)
        
        logger.info(f'Epoch {epoch + 1}/{num_epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        logger.info(f'Train Metrics: {train_metrics}')
        logger.info(f'Val Metrics: {val_metrics}')
        logger.info('\nPer-Class Metrics:')
        logger.info('Train:')
        for condition, metrics in train_per_class.items():
            logger.info(f'{condition}: {metrics}')
        logger.info('\nValidation:')
        for condition, metrics in val_per_class.items():
            logger.info(f'{condition}: {metrics}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
            torch.save(best_model, 'best_xray_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    # Save final results to file
    with open('d:/Studies/Projects/Mimic/results.txt', 'w') as f:
        f.write('Training Results Summary\n')
        f.write('=======================\n\n')
        for result in all_results:
            f.write(f"Epoch {result['epoch']}\n")
            f.write(f"Train Loss: {result['train_loss']:.4f}, Val Loss: {result['val_loss']:.4f}\n")
            f.write(f"Train Metrics: {result['train_metrics']}\n")
            f.write(f"Val Metrics: {result['val_metrics']}\n")
            f.write('\nPer-Class Metrics:\n')
            f.write('Train:\n')
            for condition, metrics in result['train_per_class'].items():
                f.write(f'{condition}: {metrics}\n')
            f.write('\nValidation:\n')
            for condition, metrics in result['val_per_class'].items():
                f.write(f'{condition}: {metrics}\n')
            f.write('\n' + '='*50 + '\n\n')
    
    return best_model

# Metrics Functions
def calculate_metrics(labels, preds, threshold=0.5):
    preds_binary = (preds >= threshold).astype(int)
    try:
        roc_auc = roc_auc_score(labels, preds, average='weighted')
    except ValueError:
        roc_auc = float('nan')
    return {
        'accuracy': accuracy_score(labels, preds_binary),
        'precision': precision_score(labels, preds_binary, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds_binary, average='weighted', zero_division=0),
        'f1': f1_score(labels, preds_binary, average='weighted', zero_division=0),
        'roc_auc': roc_auc
    }

def calculate_per_class_metrics(labels, preds, threshold=0.5):
    preds_binary = (preds >= threshold).astype(int)
    metrics = {}
    for i, condition in enumerate(CONDITIONS):
        metrics[condition] = {
            'precision': precision_score(labels[:, i], preds_binary[:, i], zero_division=0),
            'recall': recall_score(labels[:, i], preds_binary[:, i], zero_division=0),
            'f1': f1_score(labels[:, i], preds_binary[:, i], zero_division=0)
        }
    return metrics

# Predict Function
def predict(model, image_path, device):
    model.eval()
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    image = Image.open(normalize_path(image_path)).convert('RGB')
    inputs = image_processor(images=image, return_tensors='pt')
    pixel_values = inputs['pixel_values'].to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dummy_text = "inference"
    encoding = tokenizer(dummy_text, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device),
            pixel_values=pixel_values
        )
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        predictions = {CONDITIONS[i]: float(probs[i]) for i in range(len(CONDITIONS))}
        max_prob_index = np.argmax(probs)
    return predictions, max_prob_index

# Grad-CAM Function
def generate_gradcam(model, image_path, condition_index, device):
    model.eval()
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    image = Image.open(normalize_path(image_path)).convert('RGB')
    inputs = image_processor(images=image, return_tensors='pt')
    pixel_values = inputs['pixel_values'].to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dummy_text = "inference"
    encoding = tokenizer(dummy_text, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    logits = model(input_ids, attention_mask, pixel_values)
    model.last_hidden_state.retain_grad()
    score = logits[0, condition_index]
    model.zero_grad()
    score.backward()
    
    grads = model.last_hidden_state.grad[:, 1:, :]
    weights = torch.mean(grads, dim=2)
    cam = nn.functional.relu(weights).view(1, 14, 14)
    cam = nn.functional.interpolate(cam.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

# Plot Grad-CAM Function
def plot_gradcam(image_path, cam, title='Grad-CAM'):
    image = Image.open(normalize_path(image_path)).convert('RGB').resize((224, 224))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Main Function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load pre-computed labels from CSV
    labels_df = pd.read_csv('condition_labels.csv')
    labels_df['conditions'] = labels_df['conditions'].fillna('')  # Handle any NaN values
    
    # Load image paths
    df = pd.DataFrame()
    image_paths = []
    transcript_paths = []
    
    # Walk through patient directories
    for patient_dir in os.listdir(XAMI_MIMIC_PATH):
        if not patient_dir.startswith('patient_'):
            continue
            
        patient_path = normalize_path(os.path.join(XAMI_MIMIC_PATH, patient_dir))
        
        # Look for REFLACX transcripts first
        reflacx_path = normalize_path(os.path.join(patient_path, 'REFLACX', 'main_data'))
        if os.path.exists(reflacx_path):
            for study_dir in os.listdir(reflacx_path):
                study_path = normalize_path(os.path.join(reflacx_path, study_dir))
                transcript_path = normalize_path(os.path.join(study_path, 'transcription.txt'))
                if os.path.exists(transcript_path):
                    # Look for corresponding images in CXR-JPG
                    cxr_jpg_path = normalize_path(os.path.join(patient_path, 'CXR-JPG'))
                    if os.path.exists(cxr_jpg_path):
                        for study_dir in os.listdir(cxr_jpg_path):
                            if not os.path.isdir(os.path.join(cxr_jpg_path, study_dir)):
                                continue
                            study_path = normalize_path(os.path.join(cxr_jpg_path, study_dir))
                            for file in os.listdir(study_path):
                                if file.endswith(('.jpg', '.png')):
                                    image_path = normalize_path(os.path.join(study_path, file))
                                    image_paths.append(image_path)
                                    csv_transcript_path = os.path.join(
                                        'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0',
                                        'mimic-eye',
                                        os.path.relpath(transcript_path, XAMI_MIMIC_PATH)
                                    )
                                    transcript_paths.append(normalize_csv_path(csv_transcript_path))
        
        # Then look for EyeGaze transcripts
        eyegaze_path = normalize_path(os.path.join(patient_path, 'EyeGaze'))
        if os.path.exists(eyegaze_path):
            transcript_dir = normalize_path(os.path.join(eyegaze_path, 'audio_segmentation_transcripts'))
            if os.path.exists(transcript_dir):
                for transcript_file in os.listdir(transcript_dir):
                    if transcript_file.endswith('.json'):
                        transcript_path = normalize_path(os.path.join(transcript_dir, transcript_file))
                        # Look for corresponding images in CXR-JPG
                        cxr_jpg_path = normalize_path(os.path.join(patient_path, 'CXR-JPG'))
                        if os.path.exists(cxr_jpg_path):
                            for study_dir in os.listdir(cxr_jpg_path):
                                if not os.path.isdir(os.path.join(cxr_jpg_path, study_dir)):
                                    continue
                                study_path = normalize_path(os.path.join(cxr_jpg_path, study_dir))
                                for file in os.listdir(study_path):
                                    if file.endswith(('.jpg', '.png')):
                                        image_path = normalize_path(os.path.join(study_path, file))
                                        image_paths.append(image_path)
                                        csv_transcript_path = os.path.join(
                                            'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0',
                                            'mimic-eye',
                                            os.path.relpath(transcript_path, XAMI_MIMIC_PATH)
                                        )
                                        transcript_paths.append(normalize_csv_path(csv_transcript_path))
    
    df['image_path'] = image_paths
    df['transcript_path'] = transcript_paths
    
    # Merge with labels dataframe
    df = pd.merge(df, labels_df, on='transcript_path', how='inner')
    
    if len(df) == 0:
        logger.error("No matching images and transcripts found. Cannot proceed.")
        return
    
    # Convert conditions string to binary labels
    labels_list = []
    for conditions in df['conditions']:
        condition_list = conditions.split('|') if conditions else []
        labels = [1 if condition in condition_list else 0 for condition in CONDITIONS]
        labels_list.append(labels)
    df['labels'] = labels_list
    
    # Calculate positive counts and class weights
    labels_array = np.array(labels_list)
    positive_counts = np.sum(labels_array, axis=0)
    
    total_samples = len(df)
    pos_weights = torch.tensor([total_samples / (count + 1e-5) for count in positive_counts], dtype=torch.float32)
    
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # Initialize model and move to GPU
    model = XRayReportModel(num_labels=len(CONDITIONS)).to(device)
    
    # Calculate optimal batch size
    batch_size = get_optimal_batch_size(model) if torch.cuda.is_available() else 4
    
    # Create datasets
    train_dataset = XRayReportDataset(train_df, tokenizer, image_processor, augment=True)
    val_dataset = XRayReportDataset(val_df, tokenizer, image_processor, augment=False)
    test_dataset = XRayReportDataset(test_df, tokenizer, image_processor, augment=False)
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )
    
    train_model(model, train_loader, val_loader, test_loader, device, pos_weights)
    
    # Generate Grad-CAM visualizations for sample test images
    for i in range(min(3, len(test_df))):
        test_sample = test_df.iloc[i]
        image_path = normalize_path(test_sample['image_path'].replace('{XAMI_MIMIC_PATH}', XAMI_MIMIC_PATH))
        predictions, max_prob_index = predict(model, image_path, device)
        condition = CONDITIONS[max_prob_index]
        cam = generate_gradcam(model, image_path, max_prob_index, device)
        plot_gradcam(image_path, cam, title=f'Grad-CAM for {condition}')
    
    logger.info('Training, testing, and visualization completed! Model saved as best_xray_model.pt')

if __name__ == '__main__':
    main()
