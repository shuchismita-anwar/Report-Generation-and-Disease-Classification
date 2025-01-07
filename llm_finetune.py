import os
import warnings
import json
import tempfile
from importlib.metadata import version
from typing import Dict, List, Optional, Union
import random
from pathlib import Path
from collections import defaultdict
import gc
import numpy as np
from datasets import load_dataset
import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    default_data_collator,
    set_seed,
    logging as hf_logging,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
import pandas as pd
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

hf_logging.set_verbosity_error()

def read_text_file(file_path: str) -> str:
    """Read text content from a file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {str(e)}")
        return ""

def normalize_clinical_value(value: Union[str, float, int], field: str) -> str:
    """Normalize clinical values with field-specific ranges."""
    if pd.isna(value):
        return ""
    
    try:
        float_value = float(value)
        
        # Define normal ranges for clinical measurements with expanded ranges
        ranges = {
            'temperature': (85.0, 106.0),  # Fahrenheit (includes mild hypo/hyperthermia)
            'heartrate': (35, 220),        # BPM (includes brady/tachycardia)
            'o2sat': (0, 100),             # Percentage
            'sbp': (60, 240),              # mmHg (includes hypo/hypertension)
            'dbp': (20, 280),              # mmHg (includes hypo/hypertension)
            'pain': (0, 20)                # Pain scale (max value 20)
        }
        
        if field in ranges:
            min_val, max_val = ranges[field]
            
            # Special handling for temperature to convert Celsius to Fahrenheit
            if field == 'temperature':
                # If value seems to be in Celsius (between 35-42), convert to Fahrenheit
                if 35.0 <= float_value <= 42.0:
                    float_value = (float_value * 9/5) + 32
                    print(f"Note: Converting {value}°C to {float_value:.1f}°F")
            
            if min_val <= float_value <= max_val:
                if field == 'temperature':
                    return f"{float_value:.1f}°F"
                elif field == 'o2sat':
                    return f"{int(float_value)}%"
                elif field == 'pain':
                    return f"{int(float_value)}/20"  # Use /20 scale
                elif field in ['sbp', 'dbp']:
                    return f"{int(float_value)} mmHg"
                elif field == 'heartrate':
                    return f"{int(float_value)} bpm"
                else:
                    return f"{float_value:.0f}"
            else:
                # Handle out-of-range values
                if field == 'pain':
                    if float_value > max_val:
                        return f"{int(max_val)}/20"  # Cap at 20 and use /20 scale
                    else:
                        return f"{int(float_value)}/20"  # Use /20 scale for all pain values
                elif float_value < min_val:
                    print(f"Warning: Low {field} ({float_value}) below normal range (min: {min_val})")
                else:
                    print(f"Warning: High {field} ({float_value}) above normal range (max: {max_val})")
                # Still return the value with units for out-of-range cases
                if field == 'temperature':
                    return f"{float_value:.1f}°F"
                elif field == 'o2sat':
                    return f"{int(float_value)}%"
                elif field in ['sbp', 'dbp']:
                    return f"{int(float_value)} mmHg"
                elif field == 'heartrate':
                    return f"{int(float_value)} bpm"
                
        return str(value)
    except ValueError:
        return str(value)

def augment_text(text: str) -> List[str]:
    """Perform text augmentation for radiology reports."""
    augmented_texts = [text]  # Original text is always included
    
    # 1. Synonym replacement for common medical terms
    medical_synonyms = {
        'normal': ['unremarkable', 'regular', 'typical'],
        'abnormal': ['irregular', 'unusual', 'atypical'],
        'mild': ['slight', 'minimal', 'minor'],
        'moderate': ['medium', 'intermediate'],
        'severe': ['significant', 'marked', 'extensive'],
        'acute': ['sudden', 'recent'],
        'chronic': ['long-standing', 'persistent'],
    }
    
    for word, synonyms in medical_synonyms.items():
        if word in text.lower():
            for synonym in synonyms:
                augmented_text = text.lower().replace(word, synonym)
                augmented_texts.append(augmented_text)
    
    # 2. Number format variations
    def convert_numbers(match):
        num = float(match.group())
        # Add variations like rounding, different decimal places
        variations = [
            f"{num:.0f}",
            f"{num:.1f}",
            f"{num:.2f}"
        ]
        return random.choice(variations)
    
    import re
    number_pattern = r'\d+\.?\d*'
    augmented_text = re.sub(number_pattern, convert_numbers, text)
    if augmented_text != text:
        augmented_texts.append(augmented_text)
    
    # 3. Word order variation for measurements
    measurement_pattern = r'(\d+\.?\d*)\s*(cm|mm|in)'
    def swap_measurement_order(match):
        number, unit = match.groups()
        return f"{unit} {number}"
    
    augmented_text = re.sub(measurement_pattern, swap_measurement_order, text)
    if augmented_text != text:
        augmented_texts.append(augmented_text)
    
    # 4. Add/remove optional punctuation
    punctuation_variations = [
        ('. ', ' '),
        (', ', ' '),
        (' - ', ' '),
        ('(', ''),
        (')', '')
    ]
    
    for punct, repl in punctuation_variations:
        if punct in text:
            augmented_text = text.replace(punct, repl)
            augmented_texts.append(augmented_text)
    
    # 5. Medical abbreviation expansion/contraction
    medical_abbrev = {
        'pt': 'patient',
        'dx': 'diagnosis',
        'hx': 'history',
        'w/': 'with',
        'w/o': 'without',
        'b/l': 'bilateral',
    }
    
    for abbrev, full in medical_abbrev.items():
        if abbrev in text.lower():
            augmented_text = text.lower().replace(abbrev, full)
            augmented_texts.append(augmented_text)
        if full in text.lower():
            augmented_text = text.lower().replace(full, abbrev)
            augmented_texts.append(augmented_text)
    
    # Remove duplicates and limit the number of augmentations
    augmented_texts = list(set(augmented_texts))
    max_augmentations = 5
    return augmented_texts[:max_augmentations]

def check_bitsandbytes_compatibility():
    try:
        import bitsandbytes as bnb
        import pkg_resources
        try:
            return True, pkg_resources.get_distribution("bitsandbytes").version
        except:
            return True, "0.41.1"
    except ImportError:
        return False, None

def prepare_model_for_training(model):
    """Helper function to prepare model for training"""
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model

class CustomTrainer(Trainer):
    def __init__(self, tokenizer=None, original_dataset=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        print(f"Tokenizer initialized: {self.tokenizer is not None}")
        self.original_dataset = original_dataset
        print(f"Original dataset initialized: {self.original_dataset is not None}")
        self.best_metric = float('inf')
        self.metrics_history = []
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Log gradient norms if in evaluation
        if not self.is_in_train:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Gradient norm: {total_norm}")
        
        return (loss, outputs) if return_outputs else loss
        
    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        
        # Check if tokenizer and dataset are available
        if self.tokenizer is None:
            print("Warning: Tokenizer not available for text generation evaluation")
            return metrics
            
        if self.original_dataset is None:
            print("Warning: Original dataset not available for text generation evaluation")
            return metrics
            
        sample_size = min(5, len(self.original_dataset))
        samples = random.sample(range(len(self.original_dataset)), sample_size)
        
        predictions = []
        references = []
        
        for idx in samples:
            # Get original text
            full_text = self.original_dataset[idx]['text']
            
            # Split the text into input and target
            try:
                input_text = full_text.split("Write a thorough radiology interpretation:")[0].strip()
                target_text = full_text.split("Write a thorough radiology interpretation:")[1].strip()
            except IndexError:
                print(f"Warning: Could not split text for sample {idx}")
                continue
            
            inputs = self.tokenizer(input_text, 
                                  return_tensors='pt', 
                                  truncation=True, 
                                  max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predictions.append(pred_text)
            references.append(target_text)
            
            # Calculate BLEU and ROUGE scores
            bleu1 = sentence_bleu([target_text.split()], pred_text.split(), weights=(1, 0, 0, 0))
            bleu4 = sentence_bleu([target_text.split()], pred_text.split(), weights=(0.25, 0.25, 0.25, 0.25))
            rouge_scores = self.rouge_scorer.score(target_text, pred_text)
            
            print(f"\nSample {idx}:")
            print(f"Input: {input_text[:100]}...")
            print(f"Prediction: {pred_text[:100]}...")
            print(f"Reference: {target_text[:100]}...")
            print(f"BLEU-1: {bleu1:.4f}")
            print(f"BLEU-4: {bleu4:.4f}")
            print(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}")
            print(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")
        
        # Save predictions and metrics
        output_dir = Path("evaluation_outputs")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        with open(output_dir / f"eval_samples_{timestamp}.json", "w") as f:
            json.dump({
                "predictions": predictions,
                "references": references,
                "metrics": metrics
            }, f, indent=2)
        
        # Update best metric and save checkpoint if improved
        if metrics["eval_loss"] < self.best_metric:
            self.best_metric = metrics["eval_loss"]
            self.save_model("best_model")
            print(f"\nNew best model saved! Validation loss: {metrics['eval_loss']:.4f}")
        
        self.metrics_history.append(metrics)
        return metrics

def main():
    # -------------------------------------------------
    # 1. Basic Setup
    # -------------------------------------------------
    set_seed(42)
    
    # Set CUDA memory allocation strategy
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Clear GPU cache
        torch.cuda.empty_cache()
        # Set memory efficient flags
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("CUDA is not available. Using CPU instead.")

    HF_AUTH_TOKEN = "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"

    model_name = "meta-llama/Llama-3.2-3b"

    # -------------------------------------------------
    # 2. Paths (Windows Absolute Paths)
    # -------------------------------------------------
    data_csv_path = r"./test22222.csv"

    XAMI_MIMIC_PATH = os.path.abspath(
        os.path.join(
            os.getcwd(),
            'mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0',
            'mimic-eye'
        )
    ).replace('\\', '/')
    MIMIC_EYE_PATH = XAMI_MIMIC_PATH

    print(f"XAMI_MIMIC_PATH: {XAMI_MIMIC_PATH}")
    print(f"MIMIC_EYE_PATH: {MIMIC_EYE_PATH}")

    # -------------------------------------------------
    # 3. Prepare Dataset
    # -------------------------------------------------
    df = pd.read_csv(data_csv_path)
    df = df.fillna("")
    
    def make_example(row):
        # Normalize clinical values
        def normalize_value(value, field):
            return normalize_clinical_value(value, field)

        # Extract and format patient details
        age = normalize_value(row.get('age', ''), 'age')
        gender = row.get('gender', '')
        chief_complaint = row.get('chiefcomplaint', '')
        
        # Get clinical measurements if available
        clinical_data = {
            'temperature': row.get('temperature', ''),
            'heartrate': row.get('heartrate', ''),
            'o2sat': row.get('o2sat', ''),
            'sbp': row.get('sbp', ''),
            'dbp': row.get('dbp', ''),
            'pain': row.get('pain', '')
        }
        
        # Format clinical data string
        clinical_str = "\n".join([
            f" - {k.title()}: {normalize_value(v, k)}"
            for k, v in clinical_data.items()
            if v and normalize_value(v, k)  # Only include valid normalized values
        ])

        # Construct the prompt
        prompt = (
            f"You are a radiologist. Given the following patient data:\n"
            f" - Subject ID: {row['subject_id']}\n"
            f" - Age: {age}\n"
            f" - Gender: {gender}\n"
            f" - Chief Complaint: {chief_complaint}\n"
        )
        
        if clinical_str:
            prompt += f"Clinical Measurements:\n{clinical_str}\n"
        
        prompt += "\nWrite a thorough radiology interpretation:\n\nInterpretation:"

        # Read and process the report from file
        report_text = ""
        if 'report_path' in row and row['report_path']:
            report_text = read_text_file(row['report_path'])
        
        examples = []
        if report_text:
            # Generate augmented versions of the report
            augmented_reports = augment_text(report_text)
            for aug_report in augmented_reports:
                examples.append({"text": prompt + " " + aug_report})
        else:
            # If no report is available, still include the prompt
            examples.append({"text": prompt})
        
        return examples

    # Create examples with augmentation
    data_dicts = []
    for _, row in df.iterrows():
        examples = make_example(row)
        data_dicts.extend(examples)

    print(f"Created {len(data_dicts)} examples (including augmentations)")

    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        for item in data_dicts:
            f.write(json.dumps(item) + '\n')
        temp_json_path = f.name

    try:
        # Load dataset
        dataset = load_dataset('json', data_files={'train': temp_json_path})
        raw_dataset = dataset["train"]

        # Split data into train and validation sets (80-20 split)
        total_size = len(raw_dataset)
        val_size = int(0.2 * total_size)  # 20% for validation
        train_size = total_size - val_size

        train_dataset = raw_dataset.select(range(train_size))
        val_dataset = raw_dataset.select(range(train_size, total_size))

        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")

        # Print dataset structure for debugging
        print("\nDataset structure:")
        print("Train dataset features:", train_dataset.features)
        print("Sample train example:", train_dataset[0])
        print("\nValidation dataset features:", val_dataset.features)
        print("Sample validation example:", val_dataset[0])

        # Store original validation dataset before tokenization
        original_val_dataset = val_dataset

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=HF_AUTH_TOKEN,
            padding_side="right"
        )
        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(example):
            result = tokenizer(
                example["text"],
                padding="max_length",
                max_length=1024,  # Increased to 1024 for longer inputs
                truncation=True,
                return_tensors="pt"
            )
            result["labels"] = result["input_ids"].clone()
            return result

        print("Tokenizing dataset...")
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )

        print("Loading model...")
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. This script requires a GPU.")

            torch.cuda.empty_cache()

            # More aggressive memory optimization
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Configure 4-bit quantization for more memory savings
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                load_in_8bit_fp32_cpu_offload=True  # Offload fp32 weights to CPU
            )

            # Load base model with 4-bit quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                use_auth_token=HF_AUTH_TOKEN,
                trust_remote_code=True,
                # Memory optimizations
                max_memory={0: "6GB"},  # Reserve 6GB for model
                offload_folder="offload",
                offload_state_dict=True,
            )

            # Prepare model for training
            model = prepare_model_for_training(model)

            # LoRA configuration with more memory efficient settings
            lora_config = LoraConfig(
                r=4,  # Reduced from 8
                lora_alpha=16,  # Reduced from 32
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                ],  # Reduced target modules
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=False
            )

            print("Applying LoRA adapters...")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # Training arguments with early stopping and reduced steps
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,  # Reduced number of epochs
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=4,  # Reduced from 8
                evaluation_strategy="steps",
                eval_steps=100,  # More frequent evaluation
                logging_steps=50,
                learning_rate=2e-4,
                weight_decay=0.01,
                fp16=True,
                bf16=False,
                max_grad_norm=0.3,
                max_steps=1000,  # Limit total training steps
                warmup_ratio=0.03,
                group_by_length=True,
                save_strategy="steps",
                save_steps=100,
                save_total_limit=3,  # Keep only the last 3 checkpoints
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="none"
            )

            # Add early stopping
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
            
            # Initialize the Trainer with memory-optimized compute_metrics
            trainer = CustomTrainer(
                tokenizer=tokenizer,  # Pass tokenizer first
                original_dataset=original_val_dataset,  # Pass original dataset second
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=default_data_collator,
                callbacks=[early_stopping]
            )

            print("Starting training...")
            try:
                trainer.train()
                
                # Save the final model without evaluation
                trainer.save_model("./results/final_model")
                print("Training completed successfully!")
                
            except Exception as e:
                print(f"Error during model setup or training: {str(e)}")
                raise

        except Exception as e:
            print(f"Error during model setup or training: {str(e)}")
            raise

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)

if __name__ == "__main__":
    main()
