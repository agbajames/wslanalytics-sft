import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER = os.getenv("OUT_DIR", "artifacts/sft")

print("="*60)
print("LOADING MODEL...")
print("="*60)

# Load tokenizer
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.pad_token = tok.eos_token

# Load base model
base = AutoModelForCausalLM.from_pretrained(
    BASE, 
    torch_dtype=torch.float32, 
    device_map="cpu"
)

# Load your trained adapter
model = PeftModel.from_pretrained(base, ADAPTER)

print("✅ Model loaded with your trained adapter!")

print("\n" + "="*60)
print("TEST PROMPT:")
print("="*60)

# Test prompt (similar to training format)
prompt = (
    "<s>You are WSLAnalytics: a British-English football analyst. "
    "Write concise, data-led analysis with light emoji and appropriate hashtags.</s>\n"
    "<TITLE>Arsenal vs Chelsea — Matchweek 7</TITLE>\n"
    "<CONTEXT>Arsenal increased shot volume; Chelsea's GA steady. "
    "Foord progressive carries. Expect tight margins decided by pressing and set pieces.</CONTEXT>\n\n"
    "Turn this into a numbered X thread (6 bullets) with a short verdict."
)

print(prompt[:200] + "...")

print("\n" + "="*60)
print("MODEL OUTPUT:")
print("="*60)

# Generate
ids = tok(prompt, return_tensors="pt")
output = model.generate(
    **ids, 
    max_new_tokens=220,
    do_sample=True,
    temperature=0.3,
    top_p=0.9
)

result = tok.decode(output[0], skip_special_tokens=True)

# Extract just the generated part (after the prompt)
generated = result[len(prompt):].strip()

print(generated)
print("\n" + "="*60)
