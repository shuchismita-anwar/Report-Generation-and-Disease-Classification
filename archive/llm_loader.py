import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_llm_model(device):
    try:
        print("Loading language model...")
        model_name = "meta-llama/Llama-3.2-3B"
        hf_token = "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"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
        model.to(device)
        
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            model.resize_token_embeddings(len(tokenizer))
        
        model.config.pad_token_id = tokenizer.pad_token_id
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device=0 if torch.cuda.is_available() else -1,
        )
        
        print("Successfully loaded LLM model")
        return tokenizer, pipe
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        raise
