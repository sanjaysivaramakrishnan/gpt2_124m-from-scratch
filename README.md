# GPT-2 From Scratch

A clean, from-scratch implementation of **GPT-2 (124M)** in PyTorch — trained on 10 billion tokens from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

## ✨ Features

| Feature | Details |
|---|---|
| **Architecture** | Full GPT-2 transformer (CausalAttention → MLP → Block → GPT) |
| **Flash Attention** | via `F.scaled_dot_product_attention(is_causal=True)` |
| **Mixed Precision** | `bfloat16` autocast for speed & memory savings |
| **torch.compile** | Kernel fusion across the model graph |
| **Multi-GPU (DDP)** | Distributed Data Parallel with NCCL backend |
| **Gradient Accumulation** | Simulates 524K-token batch size on any hardware |
| **Cosine LR + Warmup** | Standard GPT-2 learning-rate schedule |
| **Checkpoint Resumption** | Saves & restores optimizer + dataloader state |
| **HellaSwag Eval** | Real benchmark evaluation (completion-style) |
| **Weight Tying** | Shared weights between token embedding & output head |

## 📁 Project Structure

```
gpt-2/
├── model.py           # GPT-2 architecture (CausalAttention, MLP, Block, GPT)
├── train.py           # Training script with DDP, grad accum, checkpointing
├── generate.py        # Standalone text generation with CLI
├── hellaswag.py       # HellaSwag dataset download & preprocessing
├── hellaswag_eval.py  # HellaSwag evaluation + sample generation
├── config.py          # Centralized hyperparameters
├── requirements.txt   # Python dependencies
└── .gitignore
```

## 🏗️ Architecture

```
Input Tokens
     │
     ▼
┌─────────────────┐
│  Token Embedding │ (wte)
│  + Position Emb  │ (wpe)
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│   Transformer Block ×12 │
│  ┌───────────────────┐  │
│  │ LayerNorm → Attn  │  │  ← Multi-head causal self-attention
│  │ (+ residual)      │  │    with Flash Attention
│  ├───────────────────┤  │
│  │ LayerNorm → MLP   │  │  ← GELU feed-forward (4× expansion)
│  │ (+ residual)      │  │
│  └───────────────────┘  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│   LayerNorm     │ (ln_f)
├─────────────────┤
│   Linear Head   │ (lm_head, weight-tied with wte)
└────────┬────────┘
         │
         ▼
     Logits (vocab_size)
```

**Model Size:** ~124M parameters  
**Config:** 12 layers, 12 heads, 768 embedding dim, 1024 context length

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Text

```bash
python generate.py --checkpoint log_model_19072.pt --prompt "Hello, I'm a language model,"

# With custom sampling parameters
python generate.py --checkpoint log_model_19072.pt \
    --prompt "The future of AI is" \
    --temperature 0.9 --top_k 40 --max_length 150
```

### 3. Evaluate on HellaSwag

```bash
python hellaswag_eval.py --checkpoint log_model_19072.pt
```

### 4. Train from Scratch

```bash
# Single GPU
python train.py

# Multi-GPU with DDP
torchrun --nproc_per_node=4 train.py

# Custom hyperparameters
python train.py --lr 3e-4 --batch_size 32 --max_steps 10000

# Resume from checkpoint
python train.py --resume log/model_10000.pt
```

## 📊 Training Details

| Parameter | Value |
|---|---|
| Dataset | FineWeb-Edu (10B tokens) |
| Tokenizer | GPT-2 BPE (`tiktoken`) |
| Batch Size | 524,288 tokens (via gradient accumulation) |
| Learning Rate | 6e-4 (cosine decay to 6e-5) |
| Warmup | 500 steps |
| Total Steps | 19,073 (~1 epoch) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, ε=1e-8) |
| Weight Decay | 0.1 (on 2D+ params only) |
| Precision | bfloat16 mixed precision |

## 📚 References

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2 Paper)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — inspiration for this implementation
- [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830)
- [FineWeb-Edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

## 📜 License

This project is for educational purposes.
