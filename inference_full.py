import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
import timm
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Constants
MODEL_PATH = "output_full/model_full_vit_base_patch16_224"
IMG_DIR = os.path.join("output", "img_png") # For example images
OUTPUT_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training data CSVs for corpus creation
TRAIN_CSV = "dataset_splits/train.csv"
VAL_CSV = "dataset_splits/val.csv"
TEST_CSV = "dataset_splits/test.csv" # For evaluating retrieval on test images

# Define conditions from training (used for loading model and context)
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Edema',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pneumonia', 'Support Devices'
]

# Stop words for sub-word tokens from BioClinicalBERT tokenizer
# These are examples; might need refinement based on observed common/uninformative tokens
# Many common English words will be sub-words. Punctuation is often its own token.
SUBWORD_STOP_TOKENS = set([
    ".", ",", ":", ";", "(", ")", "[", "]", "{", "}", "-", "_", "'", "\"",
    "##s", "##d", "##ing", "##ed", "##er", "##ly", 
    "is", "are", "was", "were", "be", "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "patient", "report", "study", "image", "images", "comparison", "impression", "findings", 
    "history", "exam", "examination", "technique", "date", "time", "md", "dr",
    "shows", "seen", "present", "noted", "consistent", "likely", "suggests", "appears"
    # Add more specific sub-word tokens if they prove uninformative
])

class MultiModalModel(nn.Module):
    def __init__(self, num_conditions:int):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True, num_classes=0)
        self.img_proj = nn.Linear(768, 768)
        self.txt_enc = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.txt_proj = nn.Linear(768, 768)
        self.bbox_encoder = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2), nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2), nn.AdaptiveAvgPool2d((1,4)))
        self.fix_emb = nn.Linear(4, 64)
        self.fix_gru = nn.GRU(64, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.fix_proj = nn.Linear(1024, 768)
        self.fusion = nn.Sequential(nn.Linear(768+768+768+128, 768), nn.LayerNorm(768), nn.Dropout(0.1), nn.ReLU())
        self.classifier = nn.Linear(768, num_conditions)

    def get_image_embedding(self, img):
        with torch.no_grad():
            img_feat = self.vit(img)
            img_proj = self.img_proj(img_feat)
        return img_proj
    
    def get_text_embedding(self, input_ids, attention_mask):
        with torch.no_grad():
            txt_outputs = self.txt_enc(input_ids=input_ids, attention_mask=attention_mask)
            # Using CLS token for sentence/phrase level, but for single tokens, its own embedding is fine.
            # The output of txt_enc is last_hidden_state: (batch_size, sequence_length, hidden_size)
            # For single tokens passed to get_text_embedding, seq_length will be small (related to the token itself + special tokens)
            # We typically take the embedding of the first actual token (after [CLS]) or average if multiple sub-words form one word.
            # Here, since we embed *individual sub-words from the tokenizer*, we expect each to be a single unit for embedding.
            # The model's projection layer is key.
            txt_feat = txt_outputs.last_hidden_state[:, 0, :] # CLS token is a common choice for sentence/phrase tasks
                                                            # For individual token meaning, need to be careful. Let's assume the projection handles this context.
            txt_proj = self.txt_proj(txt_feat)
        return txt_proj

class ImageToTokenRetriever:
    def __init__(self, model_path, train_csv_path, val_csv_path, device="cuda"):
        self.device = device
        self.model = self._load_trained_model(model_path)
        self.model.eval()
        self.model.to(device)
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
        self.unique_corpus_subwords = []
        self.subword_embeddings_tensor = None
        self._build_subword_corpus_and_embeddings(train_csv_path, val_csv_path)
        
    def _load_trained_model(self, model_path):
        model = MultiModalModel(num_conditions=len(CONDITIONS))
        if os.path.exists(os.path.join(model_path, "model.safetensors")):
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(model_path, "model.safetensors"), device="cpu")
        else:
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        if missing_keys: print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys: print(f"Warning: Unexpected keys: {unexpected_keys}")
        print(f"Loaded {len(filtered_state_dict)} parameters into model.")
        return model

    def _build_subword_corpus_and_embeddings(self, train_csv_path, val_csv_path):
        print("Building sub-word token corpus from training and validation data...")
        df_train = pd.read_csv(train_csv_path)
        df_val = pd.read_csv(val_csv_path)
        corpus_df = pd.concat([df_train, df_val], ignore_index=True)
        
        all_subwords_set = set()
        for idx, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc="Extracting sub-word tokens"):
            transcript_path = row.get("transcript_path", "")
            if not transcript_path or not os.path.exists(transcript_path):
                continue
            try:
                with open(transcript_path, 'r') as f: transcript_data = json.load(f)
                full_text = transcript_data.get("full_text", "")
                if not full_text: continue
                subword_tokens_in_transcript = self.tokenizer.tokenize(full_text.lower())
                # Normalize by stripping whitespace from each sub-word token
                normalized_subwords = [sw.strip() for sw in subword_tokens_in_transcript]
                # Filter based on normalized, stripped sub-words
                filtered_subwords = [sw for sw in normalized_subwords if sw and sw not in SUBWORD_STOP_TOKENS and len(sw) > 1]
                all_subwords_set.update(filtered_subwords)
            except Exception as e: print(f"Error tokenizing transcript {transcript_path} for row {idx}: {e}")
        
        self.unique_corpus_subwords = sorted(list(all_subwords_set))
        print(f"Found {len(self.unique_corpus_subwords)} unique sub-word strings in corpus.")
        # Verification Step 1: Check for true uniqueness in the list itself
        if len(self.unique_corpus_subwords) != len(set(self.unique_corpus_subwords)):
            print("ERROR: self.unique_corpus_subwords is NOT truly unique after sorting set!")
            # Further debugging: Find duplicates
            from collections import Counter
            counts = Counter(self.unique_corpus_subwords)
            duplicates = {item: count for item, count in counts.items() if count > 1}
            print(f"Duplicates found: {duplicates}")

        if not self.unique_corpus_subwords:
            print("Warning: No unique sub-word tokens found. Cannot build embeddings.")
            self.subword_embeddings_tensor = torch.empty((0,768))
            return

        temp_embeddings_list = []
        batch_size = 256
        for i in tqdm(range(0, len(self.unique_corpus_subwords), batch_size), desc="Embedding unique sub-word tokens"):
            batch_subwords = self.unique_corpus_subwords[i:i+batch_size]
            try:
                inputs = self.tokenizer(batch_subwords, return_tensors="pt", padding=True, truncation=True, max_length=8).to(self.device)
                subword_batch_embeddings = self.model.get_text_embedding(inputs["input_ids"], inputs["attention_mask"])
                temp_embeddings_list.append(subword_batch_embeddings.cpu())
            except Exception as e:
                print(f"Error embedding batch of sub-words starting with '{batch_subwords[0] if batch_subwords else ''}': {e}")
        
        if temp_embeddings_list:
            self.subword_embeddings_tensor = torch.cat(temp_embeddings_list, dim=0)
            print(f"Embedded {self.subword_embeddings_tensor.shape[0]} unique sub-word tokens.")
            # Verification Step 2: Check if number of embeddings matches number of unique words
            if self.subword_embeddings_tensor.shape[0] != len(self.unique_corpus_subwords):
                print(f"ERROR: Mismatch! Number of embeddings ({self.subword_embeddings_tensor.shape[0]}) != Number of unique sub-words ({len(self.unique_corpus_subwords)})")
        else:
            print("Warning: No sub-word token embeddings were generated.")
            self.subword_embeddings_tensor = torch.empty((0,768))

    def _preprocess_image(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            return self.img_transform(img).unsqueeze(0).to(self.device)
        except Exception as e: print(f"Error processing image {image_path}: {e}"); return None

    def retrieve_similar_subwords_for_image(self, image_path, top_k=15):
        img_tensor = self._preprocess_image(image_path)
        if img_tensor is None or self.subword_embeddings_tensor is None or self.subword_embeddings_tensor.nelement() == 0:
            return []
        with torch.no_grad(): image_embedding = self.model.get_image_embedding(img_tensor).cpu()
        if self.subword_embeddings_tensor.shape[0] == 0: return [] 

        similarities = cosine_similarity(image_embedding.numpy(), self.subword_embeddings_tensor.numpy())[0]
        actual_top_k = min(top_k, len(similarities))
        if actual_top_k == 0: return []
        top_indices = np.argsort(similarities)[-actual_top_k:][::-1]
        
        retrieved_subwords = [(self.unique_corpus_subwords[i], float(similarities[i])) for i in top_indices]
        return retrieved_subwords

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    retriever = ImageToTokenRetriever(MODEL_PATH, TRAIN_CSV, VAL_CSV, device=DEVICE)

    def process_single_image(image_path):
        print(f"\nProcessing image: {image_path}")
        subwords = retriever.retrieve_similar_subwords_for_image(image_path, top_k=20) # Increased top_k for subwords
        if subwords:
            print("\nMost Similar Sub-word Tokens to Image:")
            for subword, score in subwords:
                print(f"  - {subword} (Similarity: {score:.4f})")
        else:
            print("Could not retrieve similar sub-word tokens.")

    def process_directory(directory, output_file="subword_retrieval_results.json"):
        results = {}
        image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in tqdm(image_files, desc="Processing directory for sub-word retrieval"):
            image_path = os.path.join(directory, img_file)
            subwords = retriever.retrieve_similar_subwords_for_image(image_path, top_k=20)
            results[img_file] = subwords
        output_path = os.path.join(OUTPUT_DIR, output_file)
        with open(output_path, "w") as f: json.dump(results, f, indent=2)
        print(f"Sub-word token retrieval results saved to {output_path}")

    def process_test_set(output_file="test_set_subword_retrieval.json"):
        try:
            test_df = pd.read_csv(TEST_CSV)
            results = {}
            for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing test set for sub-word retrieval"):
                image_name = Path(row["image_path"]).name; image_path_to_process = None
                paths_to_check = [row["image_path"], os.path.join(IMG_DIR, image_name), os.path.join(IMG_DIR, Path(image_name).stem + ".png")]
                for p in paths_to_check: 
                    if os.path.exists(p): image_path_to_process = p; break
                if not image_path_to_process:
                    print(f"Img not found: {image_name} (searched {paths_to_check})")
                    results[image_name] = {"error": "Image not found", "true_condition": row.get("condition", "Unknown")}; continue
                subwords = retriever.retrieve_similar_subwords_for_image(image_path_to_process, top_k=20)
                results[image_name] = {"retrieved_subwords": subwords, "true_condition": row.get("condition", "Unknown")}
            output_path = os.path.join(OUTPUT_DIR, output_file); 
            with open(output_path, "w") as f: json.dump(results, f, indent=2)
            print(f"Test set sub-word token retrieval results saved to {output_path}")
        except Exception as e: print(f"Error on test set sub-word retrieval: {e}")

    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test": process_test_set()
        else: 
            path = sys.argv[1]
            if os.path.isfile(path): process_single_image(path)
            elif os.path.isdir(path): process_directory(path)
            else: print(f"Error: Invalid path {path}")
    else:
        example_images = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith('.png')][:1]
        if example_images: 
            for img_file in example_images: process_single_image(img_file)
        else: 
            print(f"No example PNG images in {IMG_DIR}. Use --test or provide an image/directory path.")
            print("Usage: python inference_full.py [path] | --test")

if __name__ == "__main__":
    main() 