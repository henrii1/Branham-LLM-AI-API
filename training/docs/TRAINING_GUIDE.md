# Training Guide

## Overview

This guide covers training LoRA/QLoRA adapters for the Branham Model API (Section 8).

## Training Stages

### 1. Continued Pretraining Adapter (Section 8.1)

**Purpose**: Internalize sermon corpus patterns

**Goals**:
- Learn sermon tone and cadence
- Internalize recurring motifs and phrasing
- Familiarize with multilingual sermon titles (if applicable)

**Command**:
```bash
uv run accelerate launch training/continued_pretrain/train_lora.py \
    --config config/training/continued_pretrain.yaml
```

**Data Requirements**:
- Full sermon corpus in JSONL format
- Deterministic sharding for reproducibility
- See `datasets/docs/DATA_FORMAT.md` for format

### 2. Q/A Instruction Tuning (Section 8.3)

**Purpose**: Teach cite-or-refuse behavior

**Goals**:
- Cite-or-refuse behavior (only answer with evidence)
- Quote-intent recognition
- Multilingual output formatting
- Proper reference formatting

**Command**:
```bash
uv run accelerate launch training/instruction_tune/train_qa_lora.py \
    --config config/training/instruction_tune.yaml
```

**Data Requirements**:
- Synthetic Q/A pairs generated from sermon corpus
- Must include proper references and grounding
- See `training/instruction_tune/build_qa.py` for generation

## Multi-GPU Training (Section 9.3)

Using `accelerate` for multi-GPU support:

```bash
# Configure accelerate
uv run accelerate config

# Launch training
uv run accelerate launch --multi_gpu --num_processes=2 \
    training/continued_pretrain/train_lora.py
```

## Device Support (Section 9)

### Apple Silicon (MPS)
```bash
export DEVICE_PREFERENCE=mps
uv run python training/continued_pretrain/train_lora.py
```

### NVIDIA GPU (CUDA)
```bash
export DEVICE_PREFERENCE=cuda
export USE_BNB_QUANT=true  # Enable QLoRA quantization
uv run python training/continued_pretrain/train_lora.py
```

### CPU (Fallback)
```bash
export DEVICE_PREFERENCE=cpu
uv run python training/continued_pretrain/train_lora.py
```

## Experiment Tracking

### WandB
```bash
export WANDB_PROJECT=branham-model-api
export WANDB_ENTITY=your-entity
uv run python training/continued_pretrain/train_lora.py
```

### TensorBoard
```bash
# Training will log to ./runs by default
tensorboard --logdir runs
```

## Evaluation

### Retrieval Evaluation
```bash
uv run python training/eval/retrieval_eval.py \
    --test_queries data/eval/test_queries.jsonl
```

### Generation Evaluation
```bash
uv run python training/eval/generation_eval.py \
    --test_set data/eval/test_qa.jsonl
```

## Best Practices

1. **Start with continued pretraining** before instruction tuning
2. **Use QLoRA** for memory efficiency on consumer GPUs
3. **Log everything** with WandB or TensorBoard
4. **Validate on held-out sermons** to avoid memorization
5. **Test refusal behavior** explicitly - model should refuse generic questions
6. **Verify reference format** in generated outputs

## Configuration

All training configs are in `config/training/`:
- `continued_pretrain.yaml` - Continued pretraining settings
- `instruction_tune.yaml` - Instruction tuning settings
- `lora.yaml` - LoRA/QLoRA hyperparameters

## Troubleshooting

### Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use QLoRA (4-bit quantization)
- Reduce sequence length

### Slow Training
- Enable `torch.compile()` (if supported)
- Use `flash-attn` if available
- Increase batch size (if memory allows)
- Use mixed precision (fp16/bf16)

### Poor Convergence
- Adjust learning rate
- Increase warmup steps
- Check data quality
- Verify preprocessing

