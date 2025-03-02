import os
import pandas as pd
import torch
import torch.nn as nn
import nltk
import logging
import sys
import random
from datetime import datetime
import warnings
import numpy as np
import re
import traceback
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import torch.nn.functional as F
from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPTextModel

# Set TensorFlow warning fix
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging to both file and console
log_filename = 'results.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),  # Use 'w' mode to clear previous content
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Initialize tokenizer globally
clinical_bert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

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
            if len(file_path) >= 260 and os.name == 'nt' and not file_path.startswith('\\\\?\\'):
                file_path = '\\\\?\\' + os.path.abspath(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return text
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ''

def load_preprocess_image(path, device):
    try:
        if len(path) >= 260 and os.name == 'nt' and not path.startswith('\\\\?\\'):
            path = '\\\\?\\' + os.path.abspath(path)
        img = Image.open(path).convert('RGB')
    except Exception as e:
        logger.error(f"Error processing image {path}: {e}")
        img = Image.new('RGB', (224, 224))
    img = image_transform(img)
    return img.to(device)

# -------------------------------------------------------------------------
# NEW: Extract key findings/impression from the reference for alignment
# -------------------------------------------------------------------------

def extract_key_findings_and_impression(reference_text: str) -> str:
    """
    Naive approach: search for 'findings' and 'impression' sections in the reference.
    If not found, return the entire reference text. This helps ensure the final report
    aligns with the original diagnoses or descriptions.
    """
    # Normalize case for searching
    ref_lower = reference_text.lower()
    findings_start = ref_lower.find('findings')
    impression_start = ref_lower.find('impression')

    # If neither "findings" nor "impression" appear, just return the full text.
    if findings_start == -1 and impression_start == -1:
        return reference_text

    # We'll try to slice from "Findings" through "Impression" or end of text.
    extracted = []
    if findings_start != -1:
        # from "Findings" to the end or until "Impression"
        end_of_findings = len(reference_text)
        if impression_start != -1 and impression_start > findings_start:
            end_of_findings = impression_start
        extracted.append(reference_text[findings_start:end_of_findings].strip())

    if impression_start != -1:
        # from "Impression" to the end
        extracted.append(reference_text[impression_start:].strip())

    # Combine them
    return "\n".join(extracted).strip()

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
        max_size = 0
        for item in data:
            emb = item['unified_embedding']
            if emb.dim() == 3:
                emb = emb.squeeze(0)
            emb_size = emb.size(-1)
            max_size = max(max_size, emb_size)
        self.max_size = max_size

    def __len__(self):
        return len(self.data)

    def pad_embedding(self, embedding, target_size):
        if embedding.dim() == 3:
            embedding = embedding.squeeze(0)
        elif embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        current_size = embedding.size(1)
        if current_size < target_size:
            padding = torch.zeros(embedding.size(0), target_size - current_size, device=embedding.device)
            return torch.cat([embedding, padding], dim=1)
        return embedding[:, :target_size]

    def __getitem__(self, idx):
        embedding = self.data[idx]['unified_embedding']
        if embedding.dim() == 3:
            embedding = embedding.squeeze(0)
        elif embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        if embedding.size(0) > 1:
            embedding = embedding.mean(dim=0, keepdim=True)
        
        embedding = self.pad_embedding(embedding, self.max_size)
        clinical_data = self.data[idx]['clinical_data'].float()
        labels = self.data[idx]['disease_labels']
        
        return embedding.squeeze(0), clinical_data, labels

# -------------------------------------------------------------------------
# Step 3.5: Additional Knowledge-Based Embeddings (Optional)
# -------------------------------------------------------------------------

class KnowledgeBasedEmbedder:
    def __init__(self, model_name='allenai/scibert_scivocab_uncased'):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            logger.error(f"Error initializing KnowledgeBasedEmbedder with {model_name}: {e}")
            raise e

    def generate_text_embedding(self, text: str, device='cpu'):
        self.model = self.model.to(device)
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding

def combine_embeddings(medclip_embedding: torch.Tensor, kb_embedding: torch.Tensor):
    if medclip_embedding.device != kb_embedding.device:
        kb_embedding = kb_embedding.to(medclip_embedding.device)
    batch_size = max(medclip_embedding.size(0), kb_embedding.size(0))
    if medclip_embedding.dim() == 2:
        medclip_embedding = medclip_embedding.unsqueeze(0)
    if kb_embedding.dim() == 2:
        kb_embedding = kb_embedding.unsqueeze(0)
    if medclip_embedding.size(0) == 1 and kb_embedding.size(0) > 1:
        medclip_embedding = medclip_embedding.expand(batch_size, -1, -1)
    elif kb_embedding.size(0) == 1 and medclip_embedding.size(0) > 1:
        kb_embedding = kb_embedding.expand(batch_size, -1, -1)
    if medclip_embedding.size(1) != kb_embedding.size(1):
        if medclip_embedding.size(1) == 1:
            medclip_embedding = medclip_embedding.expand(-1, kb_embedding.size(1), -1)
        elif kb_embedding.size(1) == 1:
            kb_embedding = kb_embedding.expand(-1, medclip_embedding.size(1), -1)
    combined = torch.cat([medclip_embedding, kb_embedding], dim=-1)
    combined = combined / (combined.norm(dim=-1, keepdim=True) + 1e-8)
    return combined

# -------------------------------------------------------------------------
# Step 4: Model Definitions
# -------------------------------------------------------------------------

class ClinicalDataEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super(ClinicalDataEncoder, self).__init__()
        
        # Clinical BERT for text processing
        self.clinical_bert = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
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
        ).to(device)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(768 + 128, output_dim),  # 768 from BERT, 128 from numerical
            nn.LayerNorm(output_dim),
            nn.ReLU()
        ).to(device)
        
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
        self.efficientnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT).to(device)
        
        # Modify first convolution layer for grayscale images
        self.efficientnet.features[0][0] = nn.Conv2d(1, 48, kernel_size=3, stride=2, bias=False).to(device)
        
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
        ).to(device)
        
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
            ).to(device) for _ in range(num_classes)
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
        self.vision_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT).to(device)
        # Modify first conv layer to accept 1-channel (grayscale) images
        self.vision_model.features[0][0] = nn.Conv2d(1, 48, kernel_size=3, stride=2, padding=1, bias=False).to(device)
        # Remove the classifier
        self.vision_model = nn.Sequential(*list(self.vision_model.children())[:-1]).to(device)
        
        # Add adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        
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
        ).to(device)
        
        # Cross attention for multimodal fusion with improved architecture
        self.cross_attention_v2c = CrossAttention(dim_q=self.vision_dim, dim_k=self.clinical_dim).to(device)
        self.cross_attention_c2v = CrossAttention(dim_q=self.clinical_dim, dim_k=self.vision_dim).to(device)
        
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
        ).to(device)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def set_clinical_encoder(self, encoder):
        self.clinical_encoder = encoder
    
    def forward(self, x, clinical_features=None):
        batch_size = x.size(0)
        
        # Process image through vision backbone
        if x.dim() == 4:  # Raw image input
            x = self.vision_model(x)
            x = self.adaptive_pool(x)
            visual_features = x.view(batch_size, -1)
        else:  # Pre-computed embeddings
            visual_features = self.vision_adapter(x.view(batch_size, -1))

        # Process clinical features if provided
        if clinical_features is not None:
            # Add sequence dimension for attention
            visual_features = visual_features.unsqueeze(1)  # [batch_size, 1, vision_dim]
            clinical_features = clinical_features.unsqueeze(1)  # [batch_size, 1, clinical_dim]
            
            # Cross attention between modalities
            visual_attended = self.cross_attention_v2c(visual_features, clinical_features)
            clinical_attended = self.cross_attention_c2v(clinical_features, visual_features)
            
            # Remove sequence dimension
            visual_features = visual_attended.squeeze(1)
            clinical_features = clinical_attended.squeeze(1)
            
            # Concatenate features
            features = torch.cat([visual_features, clinical_features], dim=1)
        else:
            # If no clinical features, use only visual features
            features = visual_features
        
        # Final classification
        output = self.classifier(features)
        
        return output, features

# -------------------------------------------------------------------------
# Step 5: Load LLM Model
# -------------------------------------------------------------------------

def load_llm_model(device):
    try:
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        hf_token = "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"
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=hf_token,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with minimal configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=hf_token,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        logger.info("LLM model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading LLM model: {e}")
        return None, None

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
        # Skip empty pairs
        if not gen_report.strip() or not ref_report.strip():
            continue

        metrics['avg_length'].append(len(gen_report.split()))
        
        reference = nltk.word_tokenize(ref_report.lower())
        hypothesis = nltk.word_tokenize(gen_report.lower())
        
        try:
            metrics['bleu_1'].append(sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0),
                                                   smoothing_function=smooth_fn))
            metrics['bleu_2'].append(sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0),
                                                   smoothing_function=smooth_fn))
            metrics['bleu_3'].append(sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0),
                                                   smoothing_function=smooth_fn))
            metrics['bleu_4'].append(sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25),
                                                   smoothing_function=smooth_fn))
        except Exception as e:
            logger.error(f"Error calculating BLEU scores: {str(e)}")

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
            logger.error(f"Error calculating ROUGE scores: {str(e)}")
    
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
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    g = torch.Generator()
    g.manual_seed(42)
    
    # Counter for limiting report generation to 5 patients
    reports_generated = 0
    max_reports = 5

    # Paths
    XAMI_MIMIC_PATH = os.path.abspath(os.path.join(
        os.getcwd(),
        'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0',
        'mimic-eye'
    )).replace('\\', '/')
    MIMIC_EYE_PATH = XAMI_MIMIC_PATH

    # Data Loading
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
            df[path_col] = df[path_col].apply(lambda x: os.path.abspath(x) if pd.notna(x) else x)

    for col in ['report_path', 'reflacx_transcript_path', 'eyegaze_transcript_path']:
        if col in df.columns:
            df[col] = df[col].str.replace(
                'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye',
                MIMIC_EYE_PATH,
                regex=False
            )
            df[col] = df[col].apply(lambda x: os.path.abspath(x) if pd.notna(x) else x)

    clinical_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'age']
    df[clinical_cols] = df[clinical_cols].astype(float)
    df[clinical_cols] = df[clinical_cols].fillna(df[clinical_cols].mean())
    scaler = StandardScaler()
    df[clinical_cols] = scaler.fit_transform(df[clinical_cols])

    # Initialize MedCLIP
    try:
        vision_model = MedCLIPVisionModel()
        text_model = MedCLIPTextModel()
        medclip_model = MedCLIPModel().to(device)
        logger.info("Successfully initialized MedCLIP models.")
    except Exception as e:
        logger.error(f"Error initializing MedCLIP models: {e}")
        return

    disease_labels = [
        'Atelectasis', 'Pneumothorax', 'Consolidation', 'Pulmonary edema',
        'Enlarged cardiac silhouette', 'Fibrosis', 'Fracture', 'Support devices'
    ]

    # Prepare Embeddings
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
    
    # Optional: Additional Knowledge-Based Embeddings
    kb_embedder = KnowledgeBasedEmbedder(model_name='allenai/scibert_scivocab_uncased')

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

            medclip_embedding = (image_features + text_features) / 2
            medclip_embedding = medclip_embedding / medclip_embedding.norm(dim=-1, keepdim=True)

            kb_embeddings_list = []
            start_idx = batch_idx_local * feature_dataloader.batch_size

            for i in range(images.size(0)):
                idx = start_idx + i
                if idx >= len(df):
                    break
                row_data = df.iloc[idx]
                report_text = read_text_file(row_data['report_path'])
                reflacx_text = read_text_file(row_data['reflacx_transcript_path'])
                eyegaze_text = read_text_file(row_data['eyegaze_transcript_path'])
                combined_text = ' '.join([report_text, reflacx_text, eyegaze_text])

                kb_embedding = kb_embedder.generate_text_embedding(combined_text, device=device)
                kb_embeddings_list.append(kb_embedding)

            kb_embeddings_tensor = torch.stack(kb_embeddings_list).to(device)
            combined_batch_embedding = combine_embeddings(medclip_embedding, kb_embeddings_tensor)
            
            for i in range(combined_batch_embedding.size(0)):
                idx = start_idx + i
                if idx >= len(df):
                    break
                row_data = df.iloc[idx]
                unified_embeddings.append({
                    'id': row_data['id'],
                    'unified_embedding': combined_batch_embedding[i].unsqueeze(0),
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

    # Train/Val Split
    total_length = len(unified_embeddings)
    train_length = int(0.8 * total_length)
    val_length = total_length - train_length
    train_data, val_data = random_split(unified_embeddings, [train_length, val_length],
                                        generator=torch.Generator().manual_seed(42))

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

    # Initialize Classifier
    clinical_encoder = ClinicalDataEncoder(input_dim=len(clinical_cols)).to(device)
    classifier_model = DiseaseClassifier(num_classes=len(disease_labels)).to(device)
    
    # Set the clinical encoder
    classifier_model.set_clinical_encoder(clinical_encoder)

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
        
        for batch_idx, (embeddings, clinical_data_batch, labels) in enumerate(train_dataloader_cls):
            embeddings = embeddings.to(device)
            clinical_data_batch = clinical_data_batch.to(device)
            labels = labels.to(device)

            optimizer_cls.zero_grad()
            clinical_embedding = clinical_encoder(clinical_data_batch)
            outputs, aux_outputs = classifier_model(embeddings, clinical_embedding)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_cls.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Validation
        classifier_model.eval()
        clinical_encoder.eval()
        val_loss = 0
        num_val_batches = 0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for embeddings, clinical_data_batch, labels in val_dataloader_cls:
                embeddings = embeddings.to(device)
                clinical_data_batch = clinical_data_batch.to(device)
                labels = labels.to(device)

                clinical_embedding = clinical_encoder(clinical_data_batch)
                outputs, aux_outputs = classifier_model(embeddings, clinical_embedding)
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
                # Some metrics won't compute if all labels are 0 or 1.
                if len(np.unique(all_labels[:, i])) > 1:
                    aurocs.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
                else:
                    aurocs.append(0.5)

                disease_preds = (all_probs[:, i] >= 0.5).astype(int)
                disease_labels_true = all_labels[:, i]

                f1s.append(f1_score(disease_labels_true, disease_preds,
                                    average='weighted', zero_division=0))
                precisions.append(precision_score(disease_labels_true, disease_preds,
                                                  average='weighted', zero_division=0))
                recalls.append(recall_score(disease_labels_true, disease_preds,
                                            average='weighted', zero_division=0))

            except Exception:
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
    logger.info("\nLoading LLM model...")
    try:
        model_llm, tokenizer_llm = load_llm_model(device)
        if model_llm is None or tokenizer_llm is None:
            logger.error("Failed to load LLM model. Exiting.")
            return
        model_llm.eval()  
    except Exception as e:
        logger.error(f"Failed to load LLM model: {e}")
        return
    model_llm.model.eval()

    # -------------------------------------------------------------------------
    # Step 10: Generate Reports with Alignment to Reference
    # -------------------------------------------------------------------------
    logger.info("\nGenerating reports...")
    reports = []
    reference_reports = []

    def generate_report(entry, model_llm, tokenizer, device, classifier_model, clinical_encoder):
        try:
            logger.info(f"\nProcessing report for ID {entry['id']}...")
            logger.info("Extracting features and predictions...")
            
            # Get classifier predictions
            with torch.no_grad():
                embedding = entry['unified_embedding']
                if embedding.dim() == 3:
                    embedding = embedding.squeeze(0)
                elif embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                
                clinical_data_tensor = entry['clinical_data']
                if clinical_data_tensor.dim() == 1:
                    clinical_data_tensor = clinical_data_tensor.unsqueeze(0)
                
                embedding = embedding.to(device)
                clinical_data_tensor = clinical_data_tensor.to(device)
                
                clinical_embedding = clinical_encoder(clinical_data_tensor)
                logits, aux_outputs = classifier_model(embedding, clinical_embedding)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                
                # Get top predictions
                top_k = 3
                top_indices = np.argsort(probs)[-top_k:][::-1]
                top_probs = probs[top_indices]
                
                logger.info("\nTop predictions:")
                for idx, prob in zip(top_indices, top_probs):
                    logger.info(f"Class {idx}: {prob:.3f}")

            # Map class indices to conditions
            conditions = {
                0: "No acute cardiopulmonary abnormality",
                1: "Cardiomegaly",
                2: "Pulmonary edema",
                3: "Pneumonia",
                4: "Pleural effusion",
                5: "Atelectasis",
                6: "Pneumothorax",
                7: "Consolidation"
            }
            
            # Get the predicted conditions and their probabilities
            findings = []
            for idx, prob in zip(top_indices, top_probs):
                if prob > 0.5:  # Only include if probability > 50%
                    findings.append(f"{conditions[idx]} (confidence: {prob:.2f})")
            
            findings_str = "\n".join([f"- {f}" for f in findings]) if findings else "- No significant findings detected"
            
            prompt = f"""You are a professional radiologist. Generate a detailed radiology report based on the following information:

Clinical Context:
- Patient ID: {entry['id']}
- Model Predictions and Confidence:
{findings_str}

Guidelines for Report Generation:
1. Structure the report with clear sections: INDICATION, COMPARISON, FINDINGS, and IMPRESSION
2. Be precise in terminology but avoid over-interpretation
3. Focus on the detected conditions and their significance
4. Maintain professional radiological reporting style

Reference Report Structure:
INDICATION:
[Describe the reason for examination]

COMPARISON:
[State availability of prior imaging]

FINDINGS:
- Start with PA and lateral views description
- Describe lung fields systematically
- Comment on pleural spaces
- Note cardiac silhouette
- Mention bony structures
- Detail any abnormalities detected by the model
- Describe the extent and characteristics of identified conditions

IMPRESSION:
- Summarize the key findings based on model predictions
- List detected conditions in order of significance
- Keep interpretation focused and concise

Generate a professional radiology report following these guidelines:
"""
            
            logger.info("\nGenerating report with model predictions...")
            inputs = tokenizer_llm(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model_llm.generate(
                **inputs,
                max_length=1024,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True,
                pad_token_id=tokenizer_llm.pad_token_id,
                eos_token_id=tokenizer_llm.eos_token_id,
                do_sample=True
            )
            
            report = tokenizer_llm.decode(outputs[0], skip_special_tokens=True)
            logger.info("\nReport generated successfully")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    for entry in unified_embeddings:
        # Check if we've generated the maximum number of reports
        if reports_generated >= max_reports:
            logger.info("\nReached maximum number of reports (5). Stopping generation.")
            break
            
        logger.info(f"\nGenerating report for ID {entry['id']}:")
        logger.info("-" * 50)

        # -------------------------------------------------------------
        # 1) Get reference text and extract key findings/impression
        # -------------------------------------------------------------
        reference_report_raw = read_text_file(entry['labels']['report_path'])
        # This ensures we parse important details for direct alignment
        parsed_reference_content = extract_key_findings_and_impression(reference_report_raw)

        # -------------------------------------------------------------
        # 2) Classifier logic (optional for further AI usage)
        # -------------------------------------------------------------
        try:
            classifier_model.eval()
            clinical_encoder.eval()
            with torch.no_grad():
                embedding = entry['unified_embedding']
                if embedding.dim() == 3:
                    embedding = embedding.squeeze(0)  # [batch_size, channels]
                elif embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)  # [1, channels]
                
                clinical_data_tensor = entry['clinical_data']
                if clinical_data_tensor.dim() == 1:
                    clinical_data_tensor = clinical_data_tensor.unsqueeze(0)
                
                embedding = embedding.to(device)
                clinical_data_tensor = clinical_data_tensor.to(device)
                
                clinical_embedding = clinical_encoder(clinical_data_tensor)
                logits, aux_outputs = classifier_model(embedding, clinical_embedding)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Classifier model error: {str(e)}")

        # -------------------------------------------------------------
        # 3) Build an improved prompt using reference data
        # -------------------------------------------------------------
        clinical_history = entry['labels'].get('chiefcomplaint', 'No clear indication provided.')
        comparison_text = "No prior imaging available for comparison."

        # We feed the extracted reference findings/impressions to help the LLM accurately align
        improved_prompt = f"""Generate a professional chest X-ray report following standard radiological format. Focus on objective findings and avoid speculation.

CLINICAL INFORMATION:
- Chief complaint: {clinical_history}

REPORT FORMAT:
INDICATION: State the reason for examination
COMPARISON: {comparison_text}
FINDINGS: Describe all radiographic findings systematically
IMPRESSION: Summarize key findings and their clinical significance

GUIDELINES:
1. Use precise radiological terminology
2. Include both normal and abnormal findings
3. Be concise but thorough
4. Maintain professional tone
5. Focus on imaging findings only
6. Avoid clinical speculation

Begin report:"""

        # -------------------------------------------------------------
        # 4) Final cleaning function for the LLM output
        # -------------------------------------------------------------
        def clean_report(text):
            # Remove any text before the first section header
            section_headers = ["INDICATION:", "COMPARISON:", "FINDINGS:", "IMPRESSION:"]
            
            # Find the first occurrence of any header
            first_header_pos = float('inf')
            for header in section_headers:
                pos = text.find(header)
                if pos != -1 and pos < first_header_pos:
                    first_header_pos = pos
            
            if first_header_pos != float('inf'):
                text = text[first_header_pos:]
            
            # Remove any template/instruction text
            text = re.sub(r'\[.*?\]', '', text)  # Remove any [instructions]
            text = re.sub(r'Guidelines:.*?report:', '', text, flags=re.DOTALL)  # Remove guidelines
            text = re.sub(r'You are a radiologist.*?report:', '', text, flags=re.DOTALL)  # Remove prompt
            text = re.sub(r'Begin report:.*?:', '', text, flags=re.DOTALL)  # Remove begin text
            text = re.sub(r'Remember to:.*?findings\.?', '', text, flags=re.DOTALL)  # Remove guidelines
            text = re.sub(r'CLINICAL INFORMATION:.*?REPORT FORMAT:', '', text, flags=re.DOTALL)  # Remove clinical info section
            text = re.sub(r'REPORT FORMAT:.*?GUIDELINES:', '', text, flags=re.DOTALL)  # Remove format section
            text = re.sub(r'GUIDELINES:.*?Begin report:', '', text, flags=re.DOTALL)  # Remove guidelines
            text = re.sub(r'_+', '', text)  # Remove underscores
            text = re.sub(r'Dr\..*?\n', '', text)  # Remove doctor references
            text = re.sub(r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)', '', text)  # Remove timestamps
            text = re.sub(r'End report:.*$', '', text, flags=re.DOTALL)  # Remove end report text
            text = re.sub(r'End of Report.*$', '', text, flags=re.DOTALL)  # Remove end of report text
            text = re.sub(r'Please call.*$', '', text, flags=re.DOTALL)  # Remove contact instructions
            text = re.sub(r'Radiologist:.*$', '', text, flags=re.DOTALL)  # Remove radiologist info
            text = re.sub(r'Radiology Resident:.*$', '', text, flags=re.DOTALL)  # Remove resident info
            text = re.sub(r'by Dr\..*$', '', text, flags=re.DOTALL)  # Remove doctor attribution
            text = re.sub(r'Thank you.*$', '', text, flags=re.DOTALL)  # Remove thank you notes
            text = re.sub(r'Sincerely,.*$', '', text, flags=re.DOTALL)  # Remove signature
            
            # Extract content for each section
            sections = {}
            for i, header in enumerate(section_headers):
                start = text.find(header)
                if start != -1:
                    next_header_pos = float('inf')
                    for next_header in section_headers[i+1:]:
                        pos = text.find(next_header, start)
                        if pos != -1:
                            next_header_pos = min(next_header_pos, pos)
                    
                    if next_header_pos == float('inf'):
                        content = text[start + len(header):].strip()
                    else:
                        content = text[start + len(header):next_header_pos].strip()
                    
                    # Clean the content
                    content = content.strip()
                    if not content or content.lower() in ['none', 'none.', 'no findings', 'no specific findings.']:
                        if header == "INDICATION:":
                            content = clinical_history if clinical_history else "Not provided"
                        elif header == "COMPARISON:":
                            content = comparison_text
                        elif header == "FINDINGS:":
                            content = "No acute cardiopulmonary abnormalities identified."
                        elif header == "IMPRESSION:":
                            content = "Normal chest radiograph."
                    sections[header] = content
            
            # Build the final report with sections in correct order
            final_report = []
            for header in section_headers:
                if header in sections:
                    final_report.append(f"{header}")
                    final_report.append(sections[header])
                else:
                    if header == "INDICATION:":
                        final_report.append(f"{header}")
                        final_report.append(clinical_history if clinical_history else "Not provided")
                    elif header == "COMPARISON:":
                        final_report.append(f"{header}")
                        final_report.append(comparison_text)
                    elif header == "FINDINGS:":
                        final_report.append(f"{header}")
                        final_report.append("No acute cardiopulmonary abnormalities identified.")
                    elif header == "IMPRESSION:":
                        final_report.append(f"{header}")
                        final_report.append("Normal chest radiograph.")
                final_report.append("")  # Add blank line between sections
            
            # Clean up whitespace
            text = "\n".join(final_report)
            text = re.sub(r'\n\s*\n+', '\n\n', text)  # Standardize line spacing
            text = re.sub(r'[ \t]+', ' ', text)  # Clean up horizontal whitespace
            text = text.strip()
            
            return text

        # -------------------------------------------------------------
        # 5) Generate the new radiology report from the LLM
        # -------------------------------------------------------------
        try:
            report = generate_report(entry, model_llm, tokenizer_llm, device, classifier_model, clinical_encoder)
            if report is None:
                continue

            logger.info("\nGenerated Report:")
            logger.info(report)

            reports.append({
                'id': entry['id'],
                'report': report
            })
            
            # Increment the counter
            reports_generated += 1
            logger.info(f"Report {reports_generated} of {max_reports} generated.")

            # Show reference for comparison
            if reference_report_raw:
                logger.info("\nReference Report (Original Full):")
                logger.info("-" * 50)
                logger.info(reference_report_raw)
                logger.info("-" * 50)
            reference_reports.append(reference_report_raw)

        except Exception as e:
            logger.error(f"Error generating final report: {str(e)}")
            continue

    # -------------------------------------------------------------------------
    # Step 11: Calculate Text Generation Metrics
    # -------------------------------------------------------------------------
    generated_texts = [r['report'] for r in reports]
    reference_texts = [r for r in reference_reports]
    text_metrics = calculate_text_generation_metrics(generated_texts, reference_texts)
    # (Optional) Could print or store metrics here.

if __name__ == '__main__':
    main()
