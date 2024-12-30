import torch
import pandas as pd
from torch.utils.data import Dataset
from data_utils import read_text_file, load_preprocess_image

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = self.data[idx]['unified_embedding'].squeeze(0)
        labels = self.data[idx]['disease_labels']
        clinical_data = self.data[idx]['clinical_data'].float()
        return embedding, clinical_data, labels
