# Windows Instructions for Unsloth Fine-tuning

This document outlines the Windows-specific fixes and configurations needed to run Unsloth fine-tuning successfully on Windows systems.

## Prerequisites
- Windows 10/11
- NVIDIA GPU with CUDA support
- Python 3.8+
- Virtual environment (recommended)

## Windows-Specific Issues and Fixes

### 1. Torch Dynamo Compilation Error Fix
**Issue**: InductorError with Triton compilation on Windows
**Solution**: Disable torch compile before any training operations

```python
import torch
torch._dynamo.config.disable = True  # Must be set early, before unsloth imports
torch.cuda.is_available()
```

**When to apply**: Right after importing torch, before any unsloth operations

### 2. Unicode Encoding Fix
**Issue**: UnicodeDecodeError with 'charmap' codec on Windows
**Solution**: Monkey patch the built-in open function to use UTF-8 by default

```python
# Windows-specific UTF-8 encoding fix
import builtins
original_open = builtins.open
def utf8_open(*args, **kwargs):
    # Skip binary mode files
    if 'mode' in kwargs and 'b' in kwargs['mode']:
        return original_open(*args, **kwargs)
    if len(args) >= 2 and 'b' in args[1]:
        return original_open(*args, **kwargs)
    # Add UTF-8 encoding for text files
    if 'encoding' not in kwargs and len(args) < 3:
        kwargs['encoding'] = 'utf-8'
    return original_open(*args, **kwargs)
builtins.open = utf8_open
```

**When to apply**: Before importing unsloth (not needed on Linux/Colab)

### 3. Multiprocessing Fix
**Issue**: "One of the subprocesses has abruptly died during map operation"
**Solution**: Disable multiprocessing in dataset operations and trainer config

```python
# In SFTConfig, add:
dataset_num_proc = 1  # Disable multiprocessing for Windows
```

## Complete Working Code Structure

### Cell 1: Environment Setup
```python
%load_ext autoreload
%autoreload 2
```

### Cell 2: Torch Configuration (Windows Fix #1)
```python
import torch
torch._dynamo.config.disable = True  # Windows fix for Triton compilation
torch.cuda.is_available()
```

### Cell 3: UTF-8 Encoding Fix (Windows Fix #2)
```python
# Windows-specific UTF-8 encoding fix
import builtins
original_open = builtins.open
def utf8_open(*args, **kwargs):
    if 'mode' in kwargs and 'b' in kwargs['mode']:
        return original_open(*args, **kwargs)
    if len(args) >= 2 and 'b' in args[1]:
        return original_open(*args, **kwargs)
    if 'encoding' not in kwargs and len(args) < 3:
        kwargs['encoding'] = 'utf-8'
    return original_open(*args, **kwargs)
builtins.open = utf8_open
```

### Cell 4: Model Loading
```python
from unsloth import FastLanguageModel

# Model configuration parameters
max_seq_length = 2048  # Maximum sequence length (512-8192, affects memory usage)
dtype = None           # Data type: None (auto), torch.float16, torch.bfloat16
load_in_4bit = True    # 4-bit quantization (saves ~75% memory, slight quality loss)

# Load pre-trained model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",  # HuggingFace model ID
    max_seq_length=max_seq_length,    # Context window size
    dtype=dtype,                      # Precision (None = auto-detect best)
    load_in_4bit=load_in_4bit        # Memory optimization
)
```

#### FastLanguageModel.from_pretrained() Parameters:
- **model_name**: HuggingFace model identifier or local path
- **max_seq_length**: Context window (512-8192). Higher = more memory, longer context
- **dtype**: Precision level
  - `None`: Auto-detect (recommended)
  - `torch.float16`: Half precision (faster, less memory)
  - `torch.bfloat16`: Better numerical stability
- **load_in_4bit**: Quantization for memory efficiency
  - `True`: ~75% memory reduction, minimal quality loss
  - `False`: Full precision, more memory usage

### Cell 5: LoRA Adapter Setup
```python
# Add LoRA adapters for fine-tuning quantized models
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                                    # LoRA rank (1-256)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,                          # LoRA scaling factor
    lora_dropout = 0,                         # Dropout rate (0-0.3)
    bias = "none",                            # Bias training strategy
    use_gradient_checkpointing = "unsloth",   # Memory optimization
    random_state = 3407,                      # Reproducibility seed
    use_rslora = False,                       # Rank-Stabilized LoRA
    loftq_config = None,                      # LoftQ quantization
)
```

#### FastLanguageModel.get_peft_model() Parameters:
- **r (rank)**: LoRA adapter size (1-256)
  - Lower (4-8): Faster, less memory, may underfit
  - Medium (16-32): Balanced performance
  - Higher (64-256): Better quality, more memory
- **target_modules**: Which layers to adapt
  - `["q_proj", "k_proj", "v_proj", "o_proj"]`: Attention layers only
  - Add `["gate_proj", "up_proj", "down_proj"]`: Include MLP layers (recommended)
- **lora_alpha**: Scaling factor (typically equals rank)
  - Higher values = stronger adaptation
  - Formula: effective_lr = learning_rate * lora_alpha / r
- **lora_dropout**: Regularization (0-0.3)
  - 0: No dropout (faster training)
  - 0.1-0.3: Prevents overfitting on small datasets
- **bias**: Bias parameter training
  - `"none"`: Don't train bias (recommended, faster)
  - `"all"`: Train all bias parameters
  - `"lora_only"`: Only LoRA bias
- **use_gradient_checkpointing**: Memory optimization
  - `"unsloth"`: Unsloth's optimized checkpointing
  - `True`: Standard checkpointing
  - `False`: No checkpointing (more memory)
- **use_rslora**: Rank-Stabilized LoRA
  - `True`: Better stability for high ranks
  - `False`: Standard LoRA (recommended for most cases)
- **loftq_config**: LoftQ quantization-aware initialization
  - `None`: Standard initialization
  - Custom config for better quantized model adaptation

### Cell 6: Chat Template and Dataset Setup
```python
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset

# Configure chat template for Llama format
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

# Dataset formatting function
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# Load training dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
```

### Cell 7: Dataset Processing
```python
from unsloth.chat_templates import standardize_sharegpt

# Standardize dataset format and apply chat template
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True)
```

### Cell 8: Training Setup
```python
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq

# Create trainer with Windows-specific fixes
trainer = SFTTrainer(
    model = model,                                    # Fine-tuning model
    tokenizer = tokenizer,                           # Tokenizer
    train_dataset = dataset,                         # Training data
    dataset_text_field = "text",                     # Text column name
    max_seq_length = max_seq_length,                 # Sequence length limit
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),  # Batching
    packing = False,                                 # Sequence packing
    args = SFTConfig(
        # Batch size and memory management
        per_device_train_batch_size = 2,             # Batch size per GPU
        gradient_accumulation_steps = 4,             # Gradient accumulation
        
        # Training schedule
        warmup_steps = 5,                            # Learning rate warmup
        max_steps = 60,                              # Total training steps
        # num_train_epochs = 1,                      # Alternative to max_steps
        
        # Optimization
        learning_rate = 2e-4,                        # Learning rate
        optim = "adamw_8bit",                        # Optimizer type
        weight_decay = 0.01,                         # L2 regularization
        lr_scheduler_type = "linear",                # LR schedule
        
        # Logging and output
        logging_steps = 1,                           # Log frequency
        output_dir = "outputs",                      # Save directory
        report_to = "none",                          # Experiment tracking
        
        # Reproducibility and Windows fixes
        seed = 3407,                                 # Random seed
        dataset_num_proc = 1,                        # Windows fix: disable multiprocessing
    ),
)
```

#### SFTTrainer Parameters:
- **model**: The LoRA-adapted model to train
- **tokenizer**: Tokenizer for text processing
- **train_dataset**: Dataset with "text" field containing formatted conversations
- **dataset_text_field**: Column name containing training text
- **max_seq_length**: Maximum sequence length (truncates longer sequences)
- **data_collator**: Handles batching and padding
- **packing**: Concatenate multiple short examples
  - `True`: Better GPU utilization, may affect learning
  - `False`: Each example separate (recommended for chat)

#### SFTConfig Parameters:

**Memory and Batch Management:**
- **per_device_train_batch_size**: Samples per GPU (1-8)
  - Lower: Less memory, slower training
  - Higher: More memory, faster training
- **gradient_accumulation_steps**: Accumulate gradients (1-32)
  - Effective batch size = per_device_batch_size × accumulation_steps × num_gpus
  - Use to simulate larger batches with limited memory

**Training Schedule:**
- **max_steps**: Total training steps (overrides num_train_epochs)
- **num_train_epochs**: Complete dataset passes (alternative to max_steps)
- **warmup_steps**: Steps for learning rate warmup
  - Typically 5-10% of total steps
  - Helps training stability

**Optimization:**
- **learning_rate**: Step size (1e-5 to 5e-4)
  - LoRA typically uses higher LR than full fine-tuning
  - Start with 2e-4, adjust based on loss curves
- **optim**: Optimizer choice
  - `"adamw_8bit"`: Memory-efficient (recommended)
  - `"adamw"`: Standard AdamW
  - `"sgd"`: Simple SGD
- **weight_decay**: L2 regularization (0.01-0.1)
  - Prevents overfitting
  - Higher values for smaller datasets
- **lr_scheduler_type**: Learning rate schedule
  - `"linear"`: Linear decay (recommended)
  - `"cosine"`: Cosine annealing
  - `"constant"`: No decay

**Monitoring:**
- **logging_steps**: How often to log metrics
- **save_steps**: Checkpoint frequency (optional)
- **eval_steps**: Evaluation frequency (if eval_dataset provided)
- **report_to**: Experiment tracking
  - `"none"`: No tracking
  - `"wandb"`: Weights & Biases
  - `"tensorboard"`: TensorBoard

#### Effective Batch Size Calculation:
```
Effective Batch Size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus
Example: 2 × 4 × 1 = 8 samples per optimization step
```

#### Memory vs Quality Trade-offs:
- **Higher batch size**: Better gradient estimates, more memory
- **Higher learning rate**: Faster training, risk of instability
- **More steps**: Better convergence, longer training time
- **Higher LoRA rank**: Better adaptation, more parameters to train

### Cell 9: Training
```python
# Start training
trainer_stats = trainer.train()
```

## Inference Setup

### Basic Inference
```python
from unsloth.chat_templates import get_chat_template

# Configure for inference
tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")
FastLanguageModel.for_inference(model)  # Enable 2x faster inference

# Prepare input
messages = [
    {"role": "user", "content": "Your question here"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

# Generate response (method 1: extract new tokens only)
input_length = inputs.shape[1]
outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
                         temperature = 1.5, min_p = 0.1)
response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
print(response)
```

### Streaming Inference
```python
from transformers import TextStreamer

# Streaming output (method 2: real-time streaming)
text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128,
                   use_cache=True, temperature=1.5, min_p=0.1)
```

## Parameter Tuning Guidelines

### For Different Use Cases:

**Quick Experimentation:**
```python
# Fast, minimal resource usage
r = 8, lora_alpha = 8
per_device_train_batch_size = 1
max_steps = 50
learning_rate = 5e-4
```

**Balanced Training:**
```python
# Good quality-speed balance
r = 16, lora_alpha = 16
per_device_train_batch_size = 2
max_steps = 100-500
learning_rate = 2e-4
```

**High Quality:**
```python
# Best results, more resources
r = 32, lora_alpha = 32
per_device_train_batch_size = 4
max_steps = 1000+
learning_rate = 1e-4
```

### Memory Optimization Tips:
- Reduce `per_device_train_batch_size` if OOM
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use `load_in_4bit=True` for quantization
- Enable `use_gradient_checkpointing="unsloth"`
- Lower `max_seq_length` if possible

### Training Stability:
- Start with lower learning rates (1e-4) for stability
- Use warmup_steps (5-10% of max_steps)
- Monitor loss curves - should decrease smoothly
- Add `weight_decay=0.01` to prevent overfitting

## Key Differences from Linux/Colab

1. **Torch Dynamo**: Must be disabled on Windows due to Triton compilation issues
2. **File Encoding**: Windows uses cp1252 by default, causing Unicode errors with unsloth files
3. **Multiprocessing**: Windows has issues with subprocess handling in dataset operations
4. **No Additional Dependencies**: Linux/Colab environments work without these fixes

## Troubleshooting

### Common Errors and Solutions

1. **InductorError with CompiledKernel**: Add `torch._dynamo.config.disable = True`
2. **UnicodeDecodeError with charmap**: Apply the UTF-8 encoding fix
3. **Subprocess died during map operation**: Set `dataset_num_proc = 1`
4. **Quantized model fine-tuning error**: Add LoRA adapters with `get_peft_model()`

### Performance Notes
- The Windows fixes may slightly reduce performance compared to Linux
- Disabling multiprocessing makes dataset processing slower but more stable
- All fixes are Windows-specific and not needed on Linux systems

### Recommended Starting Configuration:
```python
# Model loading
max_seq_length = 2048
load_in_4bit = True

# LoRA configuration
r = 16
lora_alpha = 16
lora_dropout = 0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training configuration
per_device_train_batch_size = 2
gradient_accumulation_steps = 4  # Effective batch size = 8
learning_rate = 2e-4
max_steps = 100  # Adjust based on dataset size
warmup_steps = 10
weight_decay = 0.01
```

This configuration provides a good balance of quality, speed, and memory usage for most fine-tuning tasks.