import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Suppress symlink warning

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
warnings.filterwarnings('ignore')  # Suppress all warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
XAMI_MIMIC_PATH = os.path.abspath(os.path.join(
    os.getcwd(),
    'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0',
    'mimic-eye'
)).replace('\\', '/')
MIMIC_EYE_PATH = XAMI_MIMIC_PATH

# Define conditions
CONDITIONS = [
    'cardiomegaly', 'pulmonary edema', 'atelectasis', 'pneumonia', 
    'pleural effusion', 'pneumothorax', 'lung nodule', 'normal'
]

# Custom Dataset Class with Augmentation
class XRayReportDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_processor, max_len=512, augment=False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len
        self.augment = augment
        
        self.augment_transforms = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Load report
        report_path = self.data.iloc[index]['report_path']
        with open(report_path, 'r', encoding='utf-8') as f:
            report_text = f.read()
        
        # Load and process image
        image_path = self.data.iloc[index]['image_path'].replace('{XAMI_MIMIC_PATH}', XAMI_MIMIC_PATH)
        image = Image.open(image_path).convert('RGB')
        
        if self.augment:
            image = self.augment_transforms(image)
        
        # Tokenize report
        encoding = self.tokenizer.encode_plus(
            report_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process image
        image_features = self.image_processor(images=image, return_tensors='pt')
        
        # Create labels
        labels = torch.zeros(len(CONDITIONS))
        report_lower = report_text.lower()
        for i, condition in enumerate(CONDITIONS):
            if condition in report_lower or (condition == 'normal' and 'no acute' in report_lower):
                labels[i] = 1
                
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': image_features['pixel_values'].squeeze(0),
            'labels': labels
        }

# Simplified Multimodal Model
class XRayReportModel(nn.Module):
    def __init__(self, num_labels):
        super(XRayReportModel, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Simplified fusion layer with higher dropout
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768 + 768, num_labels)
        
    def forward(self, input_ids, attention_mask, pixel_values):
        vit_outputs = self.vit(pixel_values=pixel_values)
        image_features = vit_outputs.pooler_output
        
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_outputs.pooler_output
        
        combined_features = torch.cat((image_features, text_features), dim=1)
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)
        
        return logits

# Training Function with Early Stopping
def train_model(model, train_loader, val_loader, test_loader, device, num_epochs=100, patience=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1} - Training'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, pixel_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1} - Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, pixel_values)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_metrics = calculate_metrics(np.array(train_labels), np.array(train_preds))
        val_metrics = calculate_metrics(np.array(val_labels), np.array(val_preds))
        
        logger.info(f'Epoch {epoch + 1}:')
        logger.info(f'Train Loss: {train_loss/len(train_loader):.4f}')
        logger.info(f'Val Loss: {val_loss/len(val_loader):.4f}')
        logger.info(f'Train Metrics: {train_metrics}')
        logger.info(f'Val Metrics: {val_metrics}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_xray_model.pt')
            logger.info('Model saved!')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    # Test phase after training
    model.load_state_dict(torch.load('best_xray_model.pt'))
    model.eval()
    test_loss = 0
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, pixel_values)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            preds = torch.sigmoid(outputs).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())
    
    test_metrics = calculate_metrics(np.array(test_labels), np.array(test_preds))
    logger.info(f'Test Loss: {test_loss/len(test_loader):.4f}')
    logger.info(f'Test Metrics: {test_metrics}')

# Metrics Calculation with Safe ROC AUC
def calculate_metrics(labels, preds, threshold=0.5):
    preds_binary = (preds >= threshold).astype(int)
    
    try:
        roc_auc = roc_auc_score(labels, preds, average='weighted')
    except ValueError:
        roc_auc = float('nan')
        
    metrics = {
        'accuracy': accuracy_score(labels, preds_binary),
        'precision': precision_score(labels, preds_binary, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds_binary, average='weighted', zero_division=0),
        'f1': f1_score(labels, preds_binary, average='weighted', zero_division=0),
        'roc_auc': roc_auc
    }
    return metrics

# Inference Function
def predict(model, image_path, device):
    model.eval()
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    image = Image.open(image_path).convert('RGB')
    inputs = image_processor(images=image, return_tensors='pt')
    
    with torch.no_grad():
        pixel_values = inputs['pixel_values'].to(device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dummy_text = "inference"
        encoding = tokenizer(dummy_text, return_tensors='pt', padding=True, truncation=True)
        
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device),
            pixel_values=pixel_values
        )
        
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        predictions = {CONDITIONS[i]: float(probs[i]) for i in range(len(CONDITIONS))}
        return predictions

# Main Execution
def main():
    # Load data and create splits
    df = pd.read_csv('crosstrain.csv')
    
    # First split: 80% train+val, 20% test
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # Second split: 80% of train_val (64% of total) for train, 20% (16% of total) for val
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
    
    # Resulting split: 64% train, 16% val, 20% test
    
    # Initialize processors
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # Create datasets and dataloaders
    train_dataset = XRayReportDataset(train_df, tokenizer, image_processor, augment=True)
    val_dataset = XRayReportDataset(val_df, tokenizer, image_processor, augment=False)
    test_dataset = XRayReportDataset(test_df, tokenizer, image_processor, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = XRayReportModel(num_labels=len(CONDITIONS)).to(device)
    
    # Train and test model
    train_model(model, train_loader, val_loader, test_loader, device)
    
    logger.info('Training and testing completed! Model saved as best_xray_model.pt')

if __name__ == '__main__':
    main()


# Output:
# INFO:__main__:Epoch 3:
# INFO:__main__:Train Loss: 0.0590
# INFO:__main__:Val Loss: 0.0514
# INFO:__main__:Train Metrics: {'accuracy': 0.9063586686537506, 'precision': 0.972879510374209, 'recall': 0.9943659603094517, 'f1': 0.9831646365318206, 'roc_auc': 0.9841565940627064}
# INFO:__main__:Val Metrics: {'accuracy': 0.9116186693147964, 'precision': 0.9724839647534225, 'recall': 0.9976423038059953, 'f1': 0.9844780390953844, 'roc_auc': 0.9820322203073384}
# INFO:__main__:Model saved!
# Epoch 4 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 1007/1007 [21:44<00:00,  1.30s/it]
# Epoch 4 - Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 252/252 [01:42<00:00,  2.45it/s] 
# INFO:__main__:Epoch 4:
# INFO:__main__:Train Loss: 0.0544
# INFO:__main__:Val Loss: 0.0442
# INFO:__main__:Train Metrics: {'accuracy': 0.8934426229508197, 'precision': 0.9717776057330064, 'recall': 0.9905819038008745, 'f1': 0.9807796724989619, 'roc_auc': 0.9860384754020884}
# INFO:__main__:Val Metrics: {'accuracy': 0.9116186693147964, 'precision': 0.9725190370786259, 'recall': 0.9976423038059953, 'f1': 0.9844871786053307, 'roc_auc': 0.9874991800226427}
# INFO:__main__:Model saved!
# Epoch 5 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 1007/1007 [22:11<00:00,  1.32s/it]
# Epoch 5 - Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 252/252 [01:55<00:00,  2.18it/s] 
# INFO:__main__:Epoch 5:
# INFO:__main__:Train Loss: 0.0440
# INFO:__main__:Val Loss: 0.0423
# INFO:__main__:Train Metrics: {'accuracy': 0.9123199205166418, 'precision': 0.9748976110622963, 'recall': 0.9939455095862765, 'f1': 0.9840741461941737, 'roc_auc': 0.9890226642321782}
# INFO:__main__:Val Metrics: {'accuracy': 0.9106256206554121, 'precision': 0.9729484571145887, 'recall': 0.9966318625799933, 'f1': 0.9842768156011413, 'roc_auc': 0.9883062472025459}
# INFO:__main__:Model saved!
# Epoch 6 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 1007/1007 [22:57<00:00,  1.37s/it]
# Epoch 6 - Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 252/252 [01:51<00:00,  2.26it/s] 
# INFO:__main__:Epoch 6:
# INFO:__main__:Train Loss: 0.0373
# INFO:__main__:Val Loss: 0.0382
# INFO:__main__:Train Metrics: {'accuracy': 0.9215101838052657, 'precision': 0.9779998939288711, 'recall': 0.9940295997309115, 'f1': 0.9857676242759383, 'roc_auc': 0.9922000147488402}
# INFO:__main__:Val Metrics: {'accuracy': 0.916583912611718, 'precision': 0.9738675590207262, 'recall': 0.9976423038059953, 'f1': 0.9852394252898178, 'roc_auc': 0.9910619883753885}
# INFO:__main__:Model saved!
# Epoch 7 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 1007/1007 [21:36<00:00,  1.29s/it]
# Epoch 7 - Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 252/252 [01:38<00:00,  2.55it/s] 
# INFO:__main__:Epoch 7:
# INFO:__main__:Train Loss: 0.0298
# INFO:__main__:Val Loss: 0.0341
# INFO:__main__:Train Metrics: {'accuracy': 0.9319423745653254, 'precision': 0.9833013458095414, 'recall': 0.9942818701648167, 'f1': 0.9878828963211773, 'roc_auc': 0.995059331929941}
# INFO:__main__:Val Metrics: {'accuracy': 0.9255213505461768, 'precision': 0.9814325402040024, 'recall': 0.9915796564499831, 'f1': 0.9863981680230306, 'roc_auc': 0.9923581732374581}
# INFO:__main__:Model saved!
# Epoch 8 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 1007/1007 [20:28<00:00,  1.22s/it]
# Epoch 8 - Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 252/252 [01:39<00:00,  2.54it/s] 
# INFO:__main__:Epoch 8:
# INFO:__main__:Train Loss: 0.0244
# INFO:__main__:Val Loss: 0.0391
# INFO:__main__:Train Metrics: {'accuracy': 0.9458519622454049, 'precision': 0.9867580738569548, 'recall': 0.9947023208879919, 'f1': 0.9903396958713274, 'roc_auc': 0.9970396845267354}
# INFO:__main__:Val Metrics: {'accuracy': 0.916583912611718, 'precision': 0.984526970363163, 'recall': 0.9848433816099696, 'f1': 0.9846263860699292, 'roc_auc': 0.9924188174071962}
# Epoch 9 - Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 1007/1007 [21:08<00:00,  1.26s/it] 
# Epoch 9 - Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 252/252 [01:42<00:00,  2.47it/s] 
# INFO:__main__:Epoch 9:
# INFO:__main__:Train Loss: 0.0186
# INFO:__main__:Val Loss: 0.0358
# 0.9931451442619088}
# Epoch 10 - Training: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1007/1007 [20:42<00:00,  1.23s/it]
# Epoch 10 - Validation: 100%|████████████████████████████████████████████████████████████████████████████████████████| 252/252 [01:39<00:00,  2.53it/s]
# INFO:__main__:Epoch 10:
# INFO:__main__:Train Loss: 0.0134
# INFO:__main__:Val Loss: 0.0377
# INFO:__main__:Train Metrics: {'accuracy': 0.9692001987083955, 'precision': 0.9928928462034415, 'recall': 0.996636394214598, 'f1': 0.9947166766974758, 'roc_auc': 0.9992352613673374}
# INFO:__main__:Val Metrics: {'accuracy': 0.9175769612711023, 'precision': 0.984551882091096, 'recall': 0.9851801953519703, 'f1': 0.9848315441616186, 'roc_auc': 0.9924047872328728}
# INFO:__main__:Early stopping triggered after 10 epochs
# Testing: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 315/315 [02:44<00:00,  1.92it/s]
# INFO:__main__:Test Loss: 0.0367
# INFO:__main__:Test Metrics: {'accuracy': 0.926131850675139, 'precision': 0.9799107532497918, 'recall': 0.9930238797960826, 'f1': 0.9863021439873686, 'roc_auc': 0.9915672059253596}
# INFO:__main__:Training and testing completed! Model saved as best_xray_model.pt