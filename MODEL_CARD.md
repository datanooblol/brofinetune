# Bro Chatbot - Fine-tuned Llama 3.2 3B

A conversational AI chatbot with a distinctive "bro" personality - casual, chilled, and supportive. Fine-tuned using Unsloth for efficient training.

## Model Details

- **Base Model**: unsloth/Llama-3.2-3B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) with Unsloth
- **Model Type**: Conversational AI Chatbot
- **Language**: English
- **License**: Same as base model

## Training Configuration

### Model Loading Parameters
```python
max_seq_length = 2048    # Maximum context window
dtype = None             # Auto-detect optimal precision
load_in_4bit = True      # 4-bit quantization for memory efficiency
```

### LoRA Configuration
```python
r = 16                   # LoRA rank
lora_alpha = 16          # LoRA scaling factor
lora_dropout = 0         # No dropout
target_modules = [       # Modules to adapt
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### Training Parameters
```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-4
max_steps = 60
warmup_steps = 5
weight_decay = 0.01
optimizer = "adamw_8bit"
```

## Usage

### Installation
```bash
pip install unsloth transformers torch
```

### Loading the Model
```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Load model with exact training parameters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "your-username/bro-chatbot",
    max_seq_length = 2048,    # IMPORTANT: Use same as training
    dtype = None,
    load_in_4bit = True,      # IMPORTANT: Use same as training
)

# Setup chat template
tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

# Enable fast inference
FastLanguageModel.for_inference(model)
```

### Basic Usage
```python
def chat_with_bro(message):
    messages = [{"role": "user", "content": message}]
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to("cuda")
    
    # Generate response
    outputs = model.generate(
        input_ids=inputs, 
        max_new_tokens=128,
        use_cache=True, 
        temperature=0.7, 
        min_p=0.1
    )
    
    # Extract only the new response
    input_length = inputs.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response

# Example usage
response = chat_with_bro("How do I learn Python?")
print(response)
```

### Streaming Usage
```python
from transformers import TextStreamer

def chat_with_bro_streaming(message):
    messages = [{"role": "user", "content": message}]
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to("cuda")
    
    # Stream response in real-time
    text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    model.generate(
        input_ids=inputs, 
        streamer=text_streamer, 
        max_new_tokens=128,
        use_cache=True, 
        temperature=0.7, 
        min_p=0.1
    )

# Example usage
chat_with_bro_streaming("What's the meaning of life?")
```

## Personality Traits

The model exhibits these characteristics:
- **Casual**: Uses relaxed language, slang, and informal expressions
- **Chilled**: Maintains a calm, easy-going attitude in all responses  
- **Supportive**: Always encouraging and positive, offering help and motivation
- **Friendly**: Uses terms like "bro", "dude", "man" in conversations
- **Helpful**: Provides useful information while maintaining the casual tone

## Training Data

- **Dataset Size**: 57 conversation pairs
- **Data Format**: User-assistant conversations in JSON format
- **Topics Covered**: 
  - General knowledge questions
  - Personal advice and motivation
  - Technical explanations
  - Casual conversations
  - Problem-solving scenarios

## Example Conversations

**User**: "What is the capital of France?"
**Bro Bot**: "Yo bro! The capital of France is Paris, dude. Beautiful city with the Eiffel Tower and all that good stuff. You thinking of visiting or just curious? Either way, you got this!"

**User**: "I'm feeling really stressed about my job interview tomorrow."
**Bro Bot**: "Hey man, I totally get that stress - interviews can be nerve-wracking! But listen bro, you wouldn't have gotten the interview if they didn't see something awesome in you already. Just be yourself, take some deep breaths, and remember you're gonna crush it. You got the skills, now just show 'em that chill confidence!"

## Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (RTX 3070, RTX 4060 Ti, or equivalent)
- **RAM**: 16GB system RAM
- **Storage**: 10GB free space

### Recommended Requirements  
- **GPU**: 16GB+ VRAM (RTX 4080, RTX 4090, or equivalent)
- **RAM**: 32GB system RAM
- **Storage**: 20GB+ free space

## Performance Notes

- **Inference Speed**: ~2x faster with Unsloth optimizations
- **Memory Usage**: ~75% reduction with 4-bit quantization
- **Context Window**: 2048 tokens maximum
- **Response Quality**: Maintains personality while being informative

## Limitations

- Trained on a relatively small dataset (57 examples)
- May occasionally break character in complex technical discussions
- Limited to English language conversations
- Context window limited to 2048 tokens
- Requires GPU for optimal performance

## Technical Details

### Model Architecture
- **Base**: Llama 3.2 3B parameters
- **Adaptation**: LoRA with rank 16
- **Quantization**: 4-bit using bitsandbytes
- **Chat Template**: Llama 3.1 format

### Training Infrastructure
- **Framework**: Unsloth + TRL (Transformers Reinforcement Learning)
- **Optimization**: AdamW 8-bit optimizer
- **Memory**: Gradient checkpointing enabled
- **Platform**: Tested on Windows and Linux

## Citation

If you use this model, please cite:

```bibtex
@misc{bro-chatbot-2024,
  title={Bro Chatbot: A Casual Conversational AI},
  author={Your Name},
  year={2024},
  howpublished={HuggingFace Model Hub},
  url={https://huggingface.co/your-username/bro-chatbot}
}
```

## Acknowledgments

- **Unsloth**: For efficient fine-tuning framework
- **Meta**: For the Llama 3.2 base model
- **HuggingFace**: For model hosting and transformers library

## Contact

For questions or issues, please open an issue on the model repository or contact [your-email@example.com].

---

**Note**: This model is for research and educational purposes. Please use responsibly and be aware of potential biases in AI-generated content.