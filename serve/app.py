import os, torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER = os.getenv("OUT_DIR", "artifacts/sft")

print("="*60)
print("LOADING MODEL FOR API...")
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

print("✅ Model loaded and ready!")
print("="*60)

# Create FastAPI app
app = FastAPI(title="WSLAnalytics API", version="1.0")

# Request schema
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 220
    temperature: float = 0.3
    top_p: float = 0.9

# Health check endpoint
@app.get("/")
def root():
    return {
        "service": "WSLAnalytics API",
        "status": "running",
        "model": BASE,
        "adapter": ADAPTER
    }

# Generation endpoint
@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate text from your fine-tuned model"""
    
    # Tokenize input
    ids = tok(req.prompt, return_tensors="pt")
    
    # Generate
    output = model.generate(
        **ids,
        max_new_tokens=req.max_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p
    )
    
    # Decode
    text = tok.decode(output[0], skip_special_tokens=True)
    
    # Extract just generated part (after prompt)
    generated = text[len(req.prompt):].strip()
    
    return {
        "text": generated,
        "prompt_length": len(req.prompt),
        "generated_length": len(generated)
    }

# Model info endpoint
@app.get("/model-info")
def model_info():
    """Get information about the loaded model"""
    return {
        "base_model": BASE,
        "adapter_path": ADAPTER,
        "vocab_size": len(tok),
        "total_parameters": model.num_parameters(),
        "device": "cpu"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
