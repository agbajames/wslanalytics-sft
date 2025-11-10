import os
from datasets import load_dataset
from transformers import AutoTokenizer

def get_env(name, default=None):
    """Safely get environment variable with fallback"""
    return os.getenv(name, default)

def tokenizer_for(model_name):
    """Load and configure tokenizer for the model"""
    tok = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True, 
        token=os.getenv("HF_TOKEN") or None
    )
    tok.pad_token = tok.eos_token
    return tok

def load_jsonl(path):
    """Load JSONL file as Hugging Face dataset"""
    return load_dataset("json", data_files=path, split="train")

def format_pair_fn(tok, max_in=1024, max_out=384):
    """Create function that formats instruction-output pairs for training"""
    def fn(x):
        instruction = x["instruction"]
        target = x["output"]
        
        # Build the full prompt
        prompt = f"{instruction}\n\n### Response:\n"
        
        # Tokenize instruction and target separately
        enc_in = tok(prompt, truncation=True, max_length=max_in)
        enc_out = tok(target, truncation=True, max_length=max_out)
        
        # Combine them
        input_ids = enc_in["input_ids"] + enc_out["input_ids"]
        attention_mask = enc_in["attention_mask"] + enc_out["attention_mask"]
        
        # Labels: -100 for instruction (don't learn), actual tokens for output (learn this!)
        labels = [-100] * len(enc_in["input_ids"]) + enc_out["input_ids"]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    return fn
