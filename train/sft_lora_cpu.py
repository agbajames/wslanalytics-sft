import os, sys, torch

# Add parent directory to path so we can import train.utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from train.utils import tokenizer_for, load_jsonl, format_pair_fn, get_env

# Load configuration from .env
BASE = get_env("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
OUT  = get_env("OUT_DIR", "artifacts/sft")
MAX_IN  = int(get_env("MAX_INPUT_TOKENS", "1024"))
MAX_OUT = int(get_env("MAX_TARGET_TOKENS", "384"))

print("="*60)
print("LOADING DATA...")
print("="*60)

# Load training and validation datasets
ds_tr = load_jsonl("data/processed/sft_train.jsonl")
ds_va = load_jsonl("data/processed/sft_val.jsonl")
print(f"Train examples: {len(ds_tr)}")
print(f"Val examples: {len(ds_va)}")

print("\n" + "="*60)
print("LOADING TOKENIZER...")
print("="*60)

# Load tokenizer
tok = tokenizer_for(BASE)
print(f"Vocab size: {len(tok)}")

print("\n" + "="*60)
print("FORMATTING DATA...")
print("="*60)

# Format data for training
fmt = format_pair_fn(tok, MAX_IN, MAX_OUT)
ds_tr = ds_tr.map(fmt, remove_columns=ds_tr.column_names)
ds_va = ds_va.map(fmt, remove_columns=ds_va.column_names)
print("Data formatted!")

print("\n" + "="*60)
print("LOADING BASE MODEL...")
print("="*60)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE, 
    torch_dtype=torch.float32,  # CPU requires float32
    device_map="cpu"
)
print(f"Model loaded: {BASE}")
print(f"Parameters: {model.num_parameters():,}")

print("\n" + "="*60)
print("ADDING LORA ADAPTERS...")
print("="*60)

# Configure LoRA
peft_cfg = LoraConfig(
    r=8,                    # LoRA rank
    lora_alpha=16,          # LoRA scaling factor
    lora_dropout=0.05,      # Dropout for regularization
    bias="none",            # Don't adapt bias terms
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Add LoRA to model
model = get_peft_model(model, peft_cfg)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
print(f"Percentage trainable: {100 * trainable_params / model.num_parameters():.2f}%")

print("\n" + "="*60)
print("CONFIGURING TRAINING...")
print("="*60)

# Training configuration
args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    eval_strategy="steps",              # ← FIXED: was evaluation_strategy
    eval_steps=100,
    save_steps=100,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=20,
    bf16=False,
    fp16=False,
    save_total_limit=2,
    report_to="none"
)

print("Training config:")
print(f"  Epochs: {args.num_train_epochs}")
print(f"  Batch size: {args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
print(f"  Learning rate: {args.learning_rate}")

# Data collator
collator = DataCollatorForLanguageModeling(tok, mlm=False)

print("\n" + "="*60)
print("STARTING TRAINING...")
print("="*60)

# Create trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tr,
    eval_dataset=ds_va,
    data_collator=collator
)

# Train!
trainer.train()

print("\n" + "="*60)
print("SAVING MODEL...")
print("="*60)

# Save the fine-tuned adapter
trainer.save_model(OUT)
tok.save_pretrained(OUT)

print(f"✅ SFT adapter saved to: {OUT}")
print("="*60)
