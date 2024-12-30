import os
import torch
import pandas as pd
from PIL import Image  # Make sure PIL is imported once here at the top
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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

def load_preprocess_image(path, device):
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        # Removed: "from PIL import Image" here
        img = Image.new('RGB', (224, 224))
    img = image_transform(img)
    return img.to(device)
