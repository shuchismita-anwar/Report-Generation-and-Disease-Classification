import os
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import logging
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------
# 1. Imports and Setup
# ---------------------------
# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a global clinical BERT tokenizer
clinical_bert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

# ---------------------------
# 2. Model Components
# ---------------------------
class ClinicalDataEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super(ClinicalDataEncoder, self).__init__()
        
        # Clinical BERT for text processing
        self.clinical_bert = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.tokenizer = clinical_bert_tokenizer  # Use the global tokenizer
        
        # Numerical features encoder
        self.numerical_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # First adjust to match input dimension
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(768 + 128, output_dim),  # 768 from BERT, 128 from numerical
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, numerical_data, text=None):
        # Process numerical features
        numerical_features = self.numerical_encoder(numerical_data)
        
        # If no text is provided, use a dummy text embedding
        if text is None:
            text_features = torch.zeros(numerical_data.size(0), 768, device=numerical_data.device)
        else:
            # Tokenize and process text with ClinicalBERT
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(numerical_data.device) for k, v in inputs.items()}
            with torch.no_grad():
                text_outputs = self.clinical_bert(**inputs)
            text_features = text_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Combine features
        combined = torch.cat([text_features, numerical_features], dim=1)
        return self.fusion(combined)

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_k):
        super().__init__()
        self.query = nn.Linear(dim_q, 256)  # Project to common dimension
        self.key = nn.Linear(dim_k, 256)
        self.value = nn.Linear(dim_k, 256)
        self.scale = 256 ** -0.5
        
        # Project back to original dimension
        self.project_out = nn.Linear(256, dim_q)
        
    def forward(self, x, context):
        # x: [batch_size, seq_len, dim_q]
        # context: [batch_size, seq_len, dim_k]
        q = self.query(x)  # [batch_size, seq_len, 256]
        k = self.key(context)  # [batch_size, seq_len, 256]
        v = self.value(context)  # [batch_size, seq_len, 256]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v  # [batch_size, seq_len, 256]
        out = self.project_out(out)  # [batch_size, seq_len, dim_q]
        return out

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=8, num_clinical_features=10):
        super(ImageClassifier, self).__init__()
        
        # Load pre-trained EfficientNet
        self.efficientnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        
        # Modify first convolution layer for grayscale images
        self.efficientnet.features[0][0] = nn.Conv2d(1, 48, kernel_size=3, stride=2, bias=False)
        
        # Get the number of features in the last layer
        num_features = self.efficientnet.classifier[1].in_features
        
        # Remove the original classifier
        self.efficientnet.classifier = nn.Identity()
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(num_features + num_clinical_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Create separate classifiers for each disease with residual connections
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 6)  # 6 classes per disease (0-5)
            ) for _ in range(num_classes)
        ])
        
        # Initialize weights
        self._initialize_weights()
        
        # Clinical feature encoder will be set later
        self.clinical_encoder = None
    
    def _initialize_weights(self):
        for m in self.shared_features.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for classifier in self.classifiers:
            for m in classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def set_clinical_encoder(self, encoder):
        self.clinical_encoder = encoder
    
    def forward(self, images, clinical_features=None):
        # Extract image features
        x = self.efficientnet(images)
        
        if clinical_features is not None:
            # Concatenate with clinical features
            x = torch.cat([x, clinical_features], dim=1)
        
        # Apply shared feature extractor
        shared_features = self.shared_features(x)
        
        # Apply each classifier
        outputs = [classifier(shared_features) for classifier in self.classifiers]
        outputs = torch.stack(outputs, dim=1)  # Shape: [batch_size, num_diseases, num_classes]
        
        return outputs, shared_features

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(DiseaseClassifier, self).__init__()
        
        # Vision backbone - EfficientNet-B4 (better performance than DenseNet)
        self.vision_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        # Modify first conv layer to accept 1-channel (grayscale) images
        self.vision_model.features[0][0] = nn.Conv2d(1, 48, kernel_size=3, stride=2, padding=1, bias=False)
        # Remove the classifier
        self.vision_model = nn.Sequential(*list(self.vision_model.children())[:-1])
        
        # Add adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Clinical data encoder is created externally
        self.clinical_encoder = None
        
        # Feature dimensions
        self.vision_dim = 1792  # EfficientNet-B4 output dimension
        self.clinical_dim = 256
        
        # Feature adaptation layers (used if input is not a raw image)
        self.vision_adapter = nn.Sequential(
            nn.Linear(1280, self.vision_dim),
            nn.LayerNorm(self.vision_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cross attention for multimodal fusion with improved architecture
        self.cross_attention_v2c = CrossAttention(dim_q=self.vision_dim, dim_k=self.clinical_dim)
        self.cross_attention_c2v = CrossAttention(dim_q=self.clinical_dim, dim_k=self.vision_dim)
        
        # Improved classifier with residual connections and batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(self.vision_dim + self.clinical_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Auxiliary classifiers for interpretability with improved architecture
        self.auxiliary_heads = nn.ModuleDict({
            'anatomical': nn.Sequential(
                nn.Linear(self.vision_dim + self.clinical_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 4)
            ),
            'severity': nn.Sequential(
                nn.Linear(self.vision_dim + self.clinical_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 3)
            ),
            'temporal': nn.Sequential(
                nn.Linear(self.vision_dim + self.clinical_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2)
            )
        })
        
    def set_clinical_encoder(self, encoder):
        self.clinical_encoder = encoder
        
    def forward(self, x, clinical_data, text_data=None):
        batch_size = x.size(0)
        
        # Handle pre-processed embeddings (from MedCLIP)
        if x.dim() == 4:  # x is a raw image tensor [batch_size, channels, height, width]
            visual_features = self.vision_model(x)
            visual_features = self.adaptive_pool(visual_features)
            visual_features = visual_features.view(batch_size, self.vision_dim)
        else:  # Handle precomputed embeddings
            visual_features = self.vision_adapter(x.view(batch_size, -1))
            
        # Extract clinical features using the external encoder
        if self.clinical_encoder is None:
            clinical_features = torch.zeros(batch_size, self.clinical_dim, device=x.device)
        else:
            clinical_features = clinical_data  # Already processed by external encoder
        
        # Add sequence dimension for attention
        visual_features = visual_features.unsqueeze(1)  # [batch_size, 1, vision_dim]
        clinical_features = clinical_features.unsqueeze(1)  # [batch_size, 1, clinical_dim]
        
        # Bidirectional cross-attention
        visual_attended = self.cross_attention_v2c(visual_features, clinical_features)  # [batch_size, 1, vision_dim]
        clinical_attended = self.cross_attention_c2v(clinical_features, visual_features)  # [batch_size, 1, clinical_dim]
        
        # Remove sequence dimension and concatenate
        visual_attended = visual_attended.squeeze(1)  # [batch_size, vision_dim]
        clinical_attended = clinical_attended.squeeze(1)  # [batch_size, clinical_dim]
        
        # Combine features
        fused_features = torch.cat([visual_attended, clinical_attended], dim=1)
        
        # Main disease classification
        main_output = self.classifier(fused_features)
        
        # Auxiliary outputs for interpretability
        auxiliary_outputs = {
            name: head(fused_features) 
            for name, head in self.auxiliary_heads.items()
        }
        
        return main_output, auxiliary_outputs

# ---------------------------
# 3. Dataset Handling
# ---------------------------
class MedicalDataset(Dataset):
    def __init__(self, df, scaler, transform=None, is_train=True):
        """
        df: DataFrame loaded from CSV.
        scaler: A fitted StandardScaler for the numeric clinical features.
        transform: torchvision transforms for the image.
        is_train: Flag indicating training or validation (if any train-only processing is needed).
        """
        self.df = df.reset_index(drop=True)
        self.scaler = scaler
        self.transform = transform
        self.is_train = is_train
        
        # Define the clinical numeric feature columns
        self.numeric_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'age']
        # Define a text column (for example, chiefcomplaint)
        self.text_col = 'chiefcomplaint'
        
        # Define disease label columns (based on available columns in CSV)
        self.label_cols = ['Airway wall thickening', 'Atelectasis', 'Consolidation', 'Enlarged cardiac silhouette', 
                          'Fibrosis', 'Fracture', 'Groundglass opacity', 'Pneumothorax']
        
        # Pre-scale numeric features for all rows (the scaler was fit on training data)
        self.numeric_features = self.scaler.transform(self.df[self.numeric_cols])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- Process the image ---
        image_path = row['image_path']
        image = Image.open(image_path).convert('L')  # Convert to grayscale as the model expects 1 channel
        if self.transform:
            image = self.transform(image)
        
        # --- Process the text ---
        text_data = row[self.text_col]
        if pd.isna(text_data):
            text_data = ""
        
        # --- Process the clinical numeric features ---
        numerical_data = torch.tensor(self.numeric_features[idx], dtype=torch.float)
        
        # --- Process the disease labels ---
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        
        return image, numerical_data, text_data, labels

# ---------------------------
# 4. Training and Evaluation
# ---------------------------
def main():
    # Replace the placeholder {XAMI_MIMIC_PATH} with the appropriate absolute path.
    XAMI_MIMIC_PATH = os.path.abspath(os.path.join(
        os.getcwd(),
        'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0',
        'mimic-eye'
    )).replace('\\', '/')
    MIMIC_EYE_PATH = XAMI_MIMIC_PATH

    logger.info("Starting training process...")
    logger.info(f"Using device: {device}")

    # Read the CSV file
    df = pd.read_csv("test.csv")
    logger.info(f"Loaded dataset with {len(df)} samples")
    
    # Replace the placeholder in the file path columns if they exist
    path_cols = ['image_path', 'report_path', 'reflacx_transcript_path', 'eyegaze_transcript_path']
    for col in path_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.replace("{XAMI_MIMIC_PATH}", XAMI_MIMIC_PATH) if isinstance(x, str) else x)
    
    # Preprocess numeric columns to handle non-numeric values
    numeric_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'age']
    for col in numeric_cols:
        if col in df.columns:
            # Convert fractions to float (e.g., '1/2' to 0.5)
            df[col] = df[col].apply(lambda x: float(eval(x)) if isinstance(x, str) and '/' in x else x)
            # Convert remaining strings to NaN and then fill with median
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # Split the DataFrame into training and validation sets (80% train, 20% val)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['split'])
    logger.info(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
    
    # Fit the StandardScaler on the training clinical numeric features
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    
    # Define image transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 1000
    EARLY_STOPPING_PATIENCE = 10
    
    # Create PyTorch Datasets and DataLoaders
    train_dataset = MedicalDataset(train_df, scaler, transform=train_transform, is_train=True)
    val_dataset = MedicalDataset(val_df, scaler, transform=val_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Instantiate the models
    clinical_input_dim = len(numeric_cols)
    clinical_encoder = ClinicalDataEncoder(input_dim=clinical_input_dim).to(device)
    model = ImageClassifier(num_classes=8, num_clinical_features=256).to(device)
    model.set_clinical_encoder(clinical_encoder)
    
    # Loss function with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not any(p is pp for pp in clinical_encoder.parameters())], 'lr': LEARNING_RATE},
        {'params': clinical_encoder.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training metrics tracking
    best_val_loss = float('inf')
    best_val_accuracy = 0
    patience_counter = 0
    
    def evaluate_model(model, loader, criterion, phase="val"):
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Get label columns from the dataset
        label_cols = loader.dataset.label_cols
        logger.info(f"Label columns: {label_cols}")
        
        with torch.no_grad():
            for images, numerical_data, text_data, labels in tqdm(loader, desc=f"{phase} evaluation"):
                images = images.to(device)
                numerical_data = numerical_data.to(device)
                labels = labels.to(device)
                
                clinical_features = clinical_encoder(numerical_data, text_data)
                outputs, _ = model(images, clinical_features)  # Shape: [batch_size, num_diseases, num_classes]
                
                # Calculate loss for each disease
                loss = 0
                for i in range(outputs.size(1)):
                    loss += criterion(outputs[:, i], labels[:, i].long())
                loss = loss / outputs.size(1)  # Average loss across diseases
                
                total_loss += loss.item() * images.size(0)
                
                # Get predictions for each disease
                preds = torch.argmax(outputs, dim=2)  # Shape: [batch_size, num_diseases]
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        logger.info(f"Predictions shape: {all_preds.shape}")
        logger.info(f"Labels shape: {all_labels.shape}")
        logger.info(f"First few predictions:\n{all_preds[:2]}")
        logger.info(f"First few labels:\n{all_labels[:2]}")
        
        # Calculate overall metrics using weighted averaging
        try:
            overall_accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
            overall_precision = precision_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
            overall_recall = recall_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
            overall_f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {str(e)}")
            overall_accuracy = overall_precision = overall_recall = overall_f1 = 0.0
        
        # Calculate per-class metrics
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        
        for i in range(len(label_cols)):
            try:
                # Get predictions and labels for this disease
                class_labels = all_labels[:, i]
                class_preds = all_preds[:, i]
                
                # Calculate metrics for each disease
                class_precision = precision_score(class_labels, class_preds, average='weighted', zero_division=0)
                class_recall = recall_score(class_labels, class_preds, average='weighted', zero_division=0)
                class_f1 = f1_score(class_labels, class_preds, average='weighted', zero_division=0)
                
            except Exception as e:
                logger.error(f"Error calculating metrics for disease {i} ({label_cols[i]}): {str(e)}")
                logger.error(f"Disease {i} unique values - Labels: {np.unique(class_labels)}, Preds: {np.unique(class_preds)}")
                class_precision = class_recall = class_f1 = 0.0
            
            precision_per_class.append(class_precision)
            recall_per_class.append(class_recall)
            f1_per_class.append(class_f1)
        
        avg_loss = total_loss / len(loader.dataset)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': overall_accuracy * 100,
            'precision': overall_precision * 100,
            'recall': overall_recall * 100,
            'f1': overall_f1 * 100
        }
        
        # Log per-class metrics
        logger.info(f"\n{phase.capitalize()} Per-class metrics:")
        for i, label in enumerate(label_cols):
            logger.info(f"{label}:")
            logger.info(f"  Precision: {precision_per_class[i]*100:.2f}%")
            logger.info(f"  Recall: {recall_per_class[i]*100:.2f}%")
            logger.info(f"  F1: {f1_per_class[i]*100:.2f}%")
            logger.info(f"  Unique labels: {np.unique(all_labels[:, i])}")
            logger.info(f"  Unique predictions: {np.unique(all_preds[:, i])}")
        
        return metrics
    
    logger.info("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        clinical_encoder.train()
        running_loss = 0.0
        
        # Training phase
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, numerical_data, text_data, labels in pbar:
            images = images.to(device)
            numerical_data = numerical_data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            clinical_features = clinical_encoder(numerical_data, text_data)
            outputs, _ = model(images, clinical_features)
            loss = 0
            for i in range(outputs.size(1)):
                loss += criterion(outputs[:, i], labels[:, i].long())
            loss = loss / outputs.size(1)  # Average loss across diseases
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'training_loss': loss.item()})
        
        # Calculate metrics for both training and validation
        train_metrics = evaluate_model(model, train_loader, criterion, "train")
        val_metrics = evaluate_model(model, val_loader, criterion, "val")
        
        # Log metrics
        logger.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        logger.info("Training Metrics:")
        for k, v in train_metrics.items():
            logger.info(f"{k}: {v:.2f}")
        
        logger.info("\nValidation Metrics:")
        for k, v in val_metrics.items():
            logger.info(f"{k}: {v:.2f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'clinical_encoder_state_dict': clinical_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info("Training complete.")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")

if __name__ == '__main__':
    main()
