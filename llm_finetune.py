import os
import warnings
import json
import tempfile
from importlib.metadata import version
from typing import Dict, List, Optional, Union
import random
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    default_data_collator,
    set_seed,
    logging as hf_logging,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import pandas as pd
import numpy as np
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
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
            'pain': (0, 20)                # Pain scale
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
                    return f"{int(float_value)}/10"
                elif field in ['sbp', 'dbp']:
                    return f"{int(float_value)} mmHg"
                elif field == 'heartrate':
                    return f"{int(float_value)} bpm"
                else:
                    return f"{float_value:.0f}"
            else:
                # Handle out-of-range values
                if field == 'pain' and float_value > max_val:
                    # print(f"Note: Pain value {float_value} capped at maximum {max_val}")
                    return f"{int(max_val)}/10"  # Cap pain at maximum
                elif float_value < min_val:
                    print(f"Warning: Low {field} ({float_value}) below normal range (min: {min_val})")
                else:
                    print(f"Warning: High {field} ({float_value}) above normal range (max: {max_val})")
                # Still return the value with units for out-of-range cases
                if field == 'temperature':
                    return f"{float_value:.1f}°F"
                elif field == 'o2sat':
                    return f"{int(float_value)}%"
                elif field == 'pain':
                    return f"{int(max_val)}/10"  # Cap pain values
                elif field in ['sbp', 'dbp']:
                    return f"{int(float_value)} mmHg"
                elif field == 'heartrate':
                    return f"{int(float_value)} bpm"
                
        return str(value)
    except ValueError:
        return str(value)

def augment_text(text: str) -> List[str]:
    """Perform simple text augmentation for radiology reports."""
    augmented = []
    
    # Medical terminology replacements
    replacements = {
        'opacity': ['shadow', 'density', 'haziness'],
        'normal': ['unremarkable', 'within normal limits', 'unchanged'],
        'abnormal': ['pathologic', 'unusual', 'irregular'],
        'evident': ['visible', 'apparent', 'noted'],
        'bilateral': ['both sides', 'on both sides'],
        'mild': ['slight', 'minimal', 'minor'],
        'moderate': ['intermediate', 'medium'],
        'severe': ['marked', 'significant', 'extensive']
    }
    
    # Create variations using replacements
    augmented.append(text)  # Original text
    
    # Create one variation with random replacements
    new_text = text
    for original, alternatives in replacements.items():
        if original.lower() in new_text.lower():
            replacement = random.choice(alternatives)
            new_text = new_text.replace(original, replacement)
    
    if new_text != text:
        augmented.append(new_text)
    
    return augmented

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
    
    # Enable gradient computation for all parameters
    for param in model.parameters():
        param.requires_grad = True
    return model

def main():
    # -------------------------------------------------
    # 1. Basic Setup
    # -------------------------------------------------
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
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

        # ---------------------
        # Split into train & validation
        # ---------------------

        # 0.1 for 90/10 split
        split_dataset = raw_dataset.train_test_split(
            test_size=0.1,
            seed=42,
            shuffle=True
        )
        train_dataset = split_dataset["train"]
        val_dataset   = split_dataset["test"]

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
                max_length=1024,  # Increased from 512 to 1024
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

            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_auth_token=HF_AUTH_TOKEN,
                use_cache=False,  # Disable KV-cache for training
                trust_remote_code=True
            )

            # Prepare model for training
            model = prepare_model_for_training(model)

            # LoRA configuration with expanded target modules
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "lm_head"
                ],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=False
            )

            print("Applying LoRA adapters...")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                
                try:
                    # Convert logits to predictions
                    if isinstance(predictions, torch.Tensor):
                        predictions = predictions.cpu().numpy()
                    predictions = np.argmax(predictions, axis=-1)
                    
                    # Decode predictions and labels
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        use_auth_token=HF_AUTH_TOKEN
                    )
                    
                    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    
                    # Calculate BLEU scores with error handling
                    smooth = SmoothingFunction().method1
                    bleu_scores = {}
                    for i in range(1, 5):
                        try:
                            scores = [
                                sentence_bleu([l.split()], p.split(), 
                                             weights=[1.0/i]*i, 
                                             smoothing_function=smooth)
                                for p, l in zip(pred_str, label_str)
                                if l.strip() and p.strip()  # Skip empty strings
                            ]
                            bleu_scores[f'bleu-{i}'] = np.mean(scores) if scores else 0.0
                        except Exception as e:
                            print(f"Warning: Error calculating BLEU-{i}: {str(e)}")
                            bleu_scores[f'bleu-{i}'] = 0.0
                    
                    # Calculate ROUGE scores with error handling
                    rouge_scores = {}
                    try:
                        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                        rouge_scores = {
                            'rouge1': np.mean([
                                scorer.score(p, l)['rouge1'].fmeasure 
                                for p, l in zip(pred_str, label_str)
                                if l.strip() and p.strip()
                            ]),
                            'rouge2': np.mean([
                                scorer.score(p, l)['rouge2'].fmeasure 
                                for p, l in zip(pred_str, label_str)
                                if l.strip() and p.strip()
                            ]),
                            'rougeL': np.mean([
                                scorer.score(p, l)['rougeL'].fmeasure 
                                for p, l in zip(pred_str, label_str)
                                if l.strip() and p.strip()
                            ])
                        }
                    except Exception as e:
                        print(f"Warning: Error calculating ROUGE scores: {str(e)}")
                        rouge_scores = {
                            'rouge1': 0.0,
                            'rouge2': 0.0,
                            'rougeL': 0.0
                        }
                    
                    # Calculate perplexity with error handling
                    try:
                        perplexity = evaluate.load("perplexity").compute(
                            predictions=pred_str, 
                            model_id='gpt2'
                        )['mean_perplexity']
                    except Exception as e:
                        print(f"Warning: Error calculating perplexity: {str(e)}")
                        perplexity = float('inf')
                    
                    metrics = {
                        'perplexity': perplexity,
                        **bleu_scores,
                        **rouge_scores
                    }
                    
                    return metrics
                except Exception as e:
                    print(f"Warning: Error in compute_metrics: {str(e)}")
                    return {
                        'perplexity': float('inf'),
                        'bleu-1': 0.0,
                        'bleu-2': 0.0,
                        'bleu-3': 0.0,
                        'bleu-4': 0.0,
                        'rouge1': 0.0,
                        'rouge2': 0.0,
                        'rougeL': 0.0
                    }

            # Training arguments
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                logging_steps=10,
                save_steps=50,
                evaluation_strategy="steps",
                eval_steps=50,
                learning_rate=2e-4,
                fp16=True,
                load_best_model_at_end=True,
                metric_for_best_model="bleu-4",
                greater_is_better=True,
            )

            # Initialize the Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=default_data_collator,
                compute_metrics=compute_metrics,
            )

            print("Starting training...")
            trainer.train()

            # Optional: Evaluate at the end of training
            final_metrics = trainer.evaluate(eval_dataset=val_dataset)
            print("Final Evaluation:", final_metrics)

        except Exception as e:
            print(f"Error during model setup or training: {str(e)}")
            raise

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)

if __name__ == "__main__":
    main()
