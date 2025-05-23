import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModel
from datasets import load_dataset
import logging
import random
from safetensors.torch import load_file  # Add this import for safetensors

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('chexpert_inference')

# Constants
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Edema',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pneumonia', 'Support Devices'
]

# Hardcoded paths and settings
MODEL_PATH = "output_image_only_chexpert_finetuned_raddino_from_hf/finetune_google_vit-base-patch16-224-in21k/checkpoint-1833/model.safetensors"
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"  # Extracting from the checkpoint path
HF_DATASET_NAME = "danjacobellis/chexpert"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Model definition (must match the training model architecture)
class ImageOnlyClassifier(nn.Module):
    def __init__(self, num_conditions: int, vit_model_name: str):
        super().__init__()
        logger.info(f"Initializing ImageOnlyClassifier for {num_conditions} conditions using HF ViT: {vit_model_name}")

        # --- Image encoder (Hugging Face ViT) ---
        try:
            self.vit = AutoModel.from_pretrained(vit_model_name)
            img_feature_dim = self.vit.config.hidden_size
            logger.info(f"Loaded HF ViT {vit_model_name}. Feature dimension: {img_feature_dim}")
        except Exception as e:
            logger.error(f"Failed to load HF ViT {vit_model_name}: {e}")
            raise e

        # Projection layer
        self.img_proj = nn.Linear(img_feature_dim, 768)  # IMG_PROJ_DIM = 768

        # Classifier Head: Outputs 4 logits for each of the num_conditions
        self.classifier = nn.Linear(768, num_conditions * 4)

    def forward(self, img, labels=None):
        # --- Image encoding (Hugging Face ViT) ---
        img_outputs = self.vit(pixel_values=img)
        
        # Use pooler_output if available, otherwise CLS token's last_hidden_state
        if hasattr(img_outputs, 'pooler_output') and img_outputs.pooler_output is not None:
            img_feat = img_outputs.pooler_output
        else:
            img_feat = img_outputs.last_hidden_state[:, 0, :]  # CLS token

        img_proj = self.img_proj(img_feat)  # [B, IMG_PROJ_DIM]

        # --- Classification ---
        logits = self.classifier(img_proj)
        
        return {"logits": logits}

def load_model():
    """
    Load the trained model from the hardcoded checkpoint path
    """
    model = ImageOnlyClassifier(len(CONDITIONS), vit_model_name=VIT_MODEL_NAME)
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        # Load using safetensors for .safetensors files
        if MODEL_PATH.endswith('.safetensors'):
            logger.info("Detected SafeTensors format, using specialized loader")
            state_dict = load_file(MODEL_PATH)
        else:
            # Fallback to regular torch.load for .pt or .bin files
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e
        
    model.to(DEVICE)
    model.eval()
    return model

def get_sample_from_dataset():
    """
    Get a sample image from the danjacobellis/chexpert dataset test set
    """
    logger.info(f"Loading {HF_DATASET_NAME} dataset from Hugging Face Hub...")
    try:
        # Load the dataset
        dataset = load_dataset(HF_DATASET_NAME, split="train")
        
        # Split into train (70%), val (15%), test (15%)
        train_val_test = dataset.train_test_split(test_size=0.30)
        train_dataset = train_val_test['train']
        val_test_dataset = train_val_test['test']
        
        # Split val_test into val and test
        val_test_split = val_test_dataset.train_test_split(test_size=0.50)
        val_dataset = val_test_split['train']
        test_dataset = val_test_split['test']
        
        logger.info(f"Dataset loaded. Test set contains {len(test_dataset)} samples.")
        
        # Get a random sample from the test set
        sample_idx = random.randint(0, len(test_dataset) - 1)
        sample = test_dataset[sample_idx]
        
        logger.info(f"Selected sample index {sample_idx} from test set")
        
        # Extract and prepare the image
        image = sample['image'].convert("RGB")
        
        # Collect ground truth labels if available
        true_labels = []
        for condition in CONDITIONS:
            if condition in sample:
                true_labels.append(int(sample[condition]) if sample[condition] is not None else 0)
            else:
                true_labels.append(0)  # Default if not present
        
        return image, true_labels, sample_idx
        
    except Exception as e:
        logger.error(f"Error getting sample from dataset: {e}")
        raise e

def preprocess_image(image):
    """
    Preprocess an image for the model
    """
    try:
        # Display the original image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title("Input X-ray Image")
        plt.axis('off')
        plt.show()
        
        # Apply the same transformations as in training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
        
        return transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise e

def run_inference(model, image_tensor):
    """
    Run inference on a preprocessed image
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        logits = outputs["logits"]
        
        # Reshape logits to [num_conditions, 4]
        logits_reshaped = logits.reshape(len(CONDITIONS), 4)
        
        # Get the predicted class (0-3) for each condition
        predictions = torch.argmax(logits_reshaped, axis=1).cpu().numpy()
        
        # Create binary prediction vector (1 if condition present)
        binary_pred_vector = [1 if pred > 0 else 0 for pred in predictions]
        
        return predictions, binary_pred_vector, logits

def visualize_predictions(predictions, binary_pred_vector, true_labels=None):
    """
    Visualize the model predictions
    """
    class_meaning = {
        0: "Not Mentioned",
        1: "Negative",
        2: "Uncertain",
        3: "Positive"
    }
    
    # Print the binary vector
    logger.info(f"Binary prediction vector (1 if condition present): {binary_pred_vector}")
    
    # Create a table for detailed predictions
    headers = ["Condition", "Prediction", "Status"]
    if true_labels is not None:
        headers.append("True Label")
    
    rows = []
    
    found_conditions = []
    for cond_idx, condition in enumerate(CONDITIONS):
        pred_class = predictions[cond_idx]
        pred_status = class_meaning.get(pred_class, "Unknown")
        
        if true_labels is not None:
            true_class = true_labels[cond_idx]
            true_status = class_meaning.get(true_class, "Unknown")
            rows.append([condition, pred_class, pred_status, f"{true_class} ({true_status})"])
        else:
            rows.append([condition, pred_class, pred_status])
        
        # Track conditions with positive or uncertain findings
        if pred_class == 2 or pred_class == 3:
            found_conditions.append(f"{condition} ({pred_status})")
    
    # Print the table
    print("\n--- CheXpert Predictions ---")
    if true_labels is not None:
        print(f"{'Condition':<15} {'Prediction':<10} {'Status':<15} {'True Label':<20}")
        print("-" * 65)
    else:
        print(f"{'Condition':<15} {'Prediction':<10} {'Status':<15}")
        print("-" * 45)
    
    for row in rows:
        if true_labels is not None:
            print(f"{row[0]:<15} {row[1]:<10} {row[2]:<15} {row[3]:<20}")
        else:
            print(f"{row[0]:<15} {row[1]:<10} {row[2]:<15}")
    
    # Summarize findings
    print("\n--- Summary ---")
    if found_conditions:
        print("Found conditions:")
        for condition in found_conditions:
            print(f"- {condition}")
    else:
        print("No significant findings detected.")

def plot_condition_probabilities(logits):
    """
    Plot the probability distribution for each condition
    """
    # Apply softmax to get probabilities
    logits_reshaped = logits.reshape(len(CONDITIONS), 4)
    probabilities = torch.softmax(logits_reshaped, dim=1).cpu().numpy()
    
    # Create a figure with subplots for each condition
    fig, axes = plt.subplots(len(CONDITIONS), 1, figsize=(10, 2*len(CONDITIONS)))
    
    class_names = ["Not Mentioned", "Negative", "Uncertain", "Positive"]
    
    for i, (condition, ax) in enumerate(zip(CONDITIONS, axes)):
        ax.bar(class_names, probabilities[i], color='skyblue')
        ax.set_title(f"{condition}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Load model
        logger.info("Loading model...")
        model = load_model()
        
        # Get a sample from the dataset
        logger.info("Getting sample from dataset...")
        image, true_labels, sample_idx = get_sample_from_dataset()
        logger.info(f"Using test sample #{sample_idx}")
        
        # Preprocess image
        logger.info("Preprocessing image...")
        image_tensor = preprocess_image(image)
        
        # Run inference
        logger.info("Running inference...")
        predictions, binary_pred_vector, logits = run_inference(model, image_tensor)
        
        # Visualize predictions with true labels
        logger.info("Visualizing predictions...")
        visualize_predictions(predictions, binary_pred_vector, true_labels)
        
        # Plot condition probabilities
        logger.info("Plotting probability distributions...")
        plot_condition_probabilities(logits)
            
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 