# WSLAnalytics LLM Fine-Tuning Project 

> **Status**: Active Development | **Current Phase**: MVP → Production-Ready System

A complete end-to-end machine learning system demonstrating parameter-efficient fine-tuning of language models for domain-specific content generation. Full-stack ML engineering from data preparation through deployment and evaluation.

## Project Vision

Transform a simple fine-tuning proof-of-concept into a production-ready ML system with:
- Docker containerization
- CI/CD pipelines
- Comprehensive evaluation framework
- RL task environment wrappers
- Professional documentation

**Current Status**: MVP complete with working training pipeline, API deployment, and automated evaluation. Actively expanding training data and building RL integration.

---

## What's Built (MVP Phase)

### Core Pipeline
```
Raw Data → Normalize → Split → Train (LoRA) → API → Evaluate
                                                   ↓
                                            83% Quality Score
```

### Key Features
- ✅ **Parameter-Efficient Training**: LoRA reduces trainable params to 0.2% (2.3M of 1.1B)
- ✅ **CPU-Optimized**: <3 minutes training time on laptop hardware
- ✅ **REST API**: FastAPI deployment with configurable generation parameters
- ✅ **Automated Evaluation**: Multi-metric quality assessment framework
- ✅ **Complete Data Pipeline**: Normalization, validation, train/val splitting

### Technologies
- **ML**: PyTorch, Transformers (Hugging Face), PEFT/LoRA
- **API**: FastAPI, Uvicorn, Pydantic
- **Data**: Datasets library, custom JSONL processing
- **Evaluation**: Custom metrics framework
- **Python**: 3.9 - 3.13

---

## 📊 Current Results

| Metric | Score | Notes |
|--------|-------|-------|
| Task Completion | 80% | Format adherence, structure |
| Fact Accuracy | 90% | Number/stat preservation |
| Safety | 80% | Refusal of harmful requests |
| **Overall** | **83%** | Baseline with 3 training examples |

**Training Performance:**
- Training time: 178 seconds (CPU-only)
- Loss reduction: 3.5 → 2.5
- Trainable parameters: 2,252,800 (0.20% of base model)
- Base model: TinyLlama-1.1B-Chat-v1.0

---

## 🏗️ Project Structure

```
wslanalytics-sft/
├── data/
│   ├── raw/                 # Original training data
│   └── processed/           # Cleaned, split datasets
├── train/
│   ├── utils.py             # Training utilities
│   └── sft_lora_cpu.py      # Main training script
├── eval/
│   ├── metrics.py           # Quality metrics
│   ├── run_eval.py          # Evaluation runner
│   └── suites/              # Test case collections
├── serve/
│   └── app.py               # FastAPI deployment
├── scripts/
│   ├── 01_normalize.py      # Data cleaning
│   ├── 02_split.py          # Train/val split
│   ├── 03_report.py         # Data quality reports
│   └── smoke_test.py        # Quick model validation
└── artifacts/               # Generated models (gitignored)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space

### Setup
```bash
# Clone and setup
git clone https://github.com/yourusername/wslanalytics-sft
cd wslanalytics-sft
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create directory structure
mkdir -p data/{raw,processed} artifacts
```

### Run the Pipeline
```bash
# 1. Prepare data
python scripts/01_normalize.py
python scripts/02_split.py
python scripts/03_report.py

# 2. Train model (~3 minutes)
python train/sft_lora_cpu.py

# 3. Quick test
python scripts/smoke_test.py

# 4. Start API (Terminal 1)
python serve/app.py

# 5. Run evaluation (Terminal 2)
python eval/run_eval.py --suite all
```

---

## 🔬 How It Works

### LoRA Fine-Tuning
Instead of training all 1.1B parameters, LoRA adds small adapter matrices to attention layers:
- **Full fine-tuning**: Update 1.1B params → requires GPU, hours of training
- **LoRA approach**: Update 2.3M params → CPU-friendly, minutes of training
- **Trade-off**: 0.2% parameters, ~80% of full fine-tuning performance

### Training Process
1. **Data formatting**: Convert instruction-output pairs into model-compatible format
2. **LoRA injection**: Add trainable adapters to query, key, value, output projection layers
3. **Supervised training**: Teach model to predict outputs given instructions
4. **Adapter saving**: Store only the 2.3M trained parameters (not the full 1.1B model)

### Evaluation Framework
Automated metrics assess:
- **Structure**: Numbered bullets, hashtags, format compliance
- **Accuracy**: Preservation of numbers and statistics from context
- **Safety**: Appropriate refusal of harmful or hallucination requests

---

## 📈 Active Development Roadmap

### Phase 1: Scale Training Data ⏳ *In Progress*
- [x] MVP with 3 training examples
- [ ] Scale to 15 examples (diverse formats and topics)
- [ ] Scale to 50+ examples (production-quality dataset)
- [ ] Implement data augmentation strategies

### Phase 2: RL Integration 🎯 *Next*
- [ ] Wrap evaluation in Gymnasium environment
- [ ] Implement reward shaping based on quality metrics
- [ ] Create training loops for RL algorithms
- [ ] Document RL task specifications

### Phase 3: Production Infrastructure 🚢 *Planned*
- [ ] Docker containerization
- [ ] GitHub Actions CI/CD
- [ ] Comprehensive test suite
- [ ] Monitoring and logging
- [ ] Model versioning

### Phase 4: Advanced ML 🔬 *Future*
- [ ] DPO (Direct Preference Optimization) implementation
- [ ] Multi-model comparison framework
- [ ] A/B testing infrastructure
- [ ] Deployment options (Lambda, HF Spaces)

---

## 🎓 Key Learnings

### What Worked
1. **LoRA enables practical experimentation**: CPU training in minutes vs. hours on GPU
2. **Automated evaluation drives iteration**: Quantitative metrics reveal specific weaknesses
3. **Small datasets teach system design**: 3 examples proved the pipeline works
4. **Modular architecture enables extension**: Easy to add new metrics, data, features

### Current Limitations
1. **Data quantity**: 3 examples cause repetitive outputs and overfitting
2. **Evaluation coverage**: Need more diverse test cases for robust quality assessment
3. **Deployment**: Local-only, not production-grade infrastructure

### Next Steps
1. **Scale data**: 3 → 50+ examples with emphasis on diversity
2. **Add RL wrapper**: Enable reinforcement learning experiments
3. **Production deploy**: Containerise and add monitoring

---

## 🔧 Technical Decisions

### Why LoRA over Full Fine-Tuning?
- **Speed**: 3 minutes vs. several hours
- **Hardware**: Laptop CPU vs. expensive GPU
- **Iteration**: Rapid experimentation and learning
- **Trade-off**: Acceptable for domain adaptation tasks

### Why TinyLlama?
- **Size**: 1.1B params fits in CPU memory
- **Quality**: Sufficient for domain-specific tasks
- **Speed**: Fast inference for API deployment
- **Open**: Fully accessible for learning and modification

### Why CPU Training?
- **Accessibility**: Anyone can reproduce without GPU access
- **Educational**: Demonstrates parameter-efficient techniques
- **Practical**: Real constraint that drove technical decisions

---
