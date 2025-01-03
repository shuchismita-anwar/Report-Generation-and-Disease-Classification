import os
import warnings
import json
import tempfile
from importlib.metadata import version

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

hf_logging.set_verbosity_error()

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
    torch.cuda.set_device(0)

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
        prompt = (
            f"You are a radiologist. Given the following patient data:\n"
            f" - Subject ID: {row['subject_id']}\n"
            f" - Chief Complaint: {row['chiefcomplaint']}\n"
            f"Write a thorough radiology interpretation:\n\n"
            "Interpretation:"
        )
        # The label/response is what you want the model to learn to generate
        label = " [Your real radiology reading or interpretative text here]"
        return {"text": prompt + label}

    data_dicts = [make_example(r) for _, r in df.iterrows()]

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
                max_length=512,
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

            print(f"Using GPU: {torch.cuda.get_device_name()}")

            torch.cuda.empty_cache()

            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                use_auth_token=HF_AUTH_TOKEN,
                use_cache=False  # Disable KV-cache for training
            )

            # Prepare model for training
            model = prepare_model_for_training(model)

            # LoRA configuration
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=False
            )

            print("Applying LoRA adapters...")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # -------------------------------------------------
            # Set up metrics
            # -------------------------------------------------
            # For language models, common metrics include perplexity.
            metric = evaluate.load("perplexity")

            def compute_metrics(eval_pred):
                # The HF Trainer passes (logits, labels)
                logits, labels = eval_pred
                # Convert logits to float32 for perplexity
                if isinstance(logits, torch.Tensor):
                    logits = logits.detach().cpu().numpy().astype(np.float32)
                if isinstance(labels, torch.Tensor):
                    labels = labels.detach().cpu().numpy()
                
                # Because we used causal language modeling,
                # we can pass the model outputs directly to perplexity.
                # The perplexity metric from 'evaluate' expects a dict:
                result = metric.compute(predictions=logits, references=labels)
                return {"perplexity": result["perplexity"]}

            # -------------------------------------------------
            # Training Arguments
            # -------------------------------------------------
            # Increase max_steps or specify num_train_epochs for extended training.
            training_args = TrainingArguments(
                output_dir="./output-llama-3.2-3b-qlora",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                max_steps=200,
                logging_steps=10,
                save_steps=50,
                evaluation_strategy="steps",  # Evaluate every X steps
                eval_steps=50,
                optim="adamw_torch",
                fp16=True,
                gradient_checkpointing=True,
                remove_unused_columns=False,
                torch_compile=False,
            )

            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,      # validation dataset
                data_collator=default_data_collator,
                compute_metrics=compute_metrics # call our perplexity function
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
