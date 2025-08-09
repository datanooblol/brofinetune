# brofinetune
A repo for fine-tuning a bro-like conversational chatbot using Supervised Fine-Tuning (SFT) with Unsloth.

## Why I Made This

Honestly, I had a rough day today. I think I blew up something important and I'm not feeling great about it. When you're down and need someone to talk to, but there's no one around... that's when I realized I needed to create something.

So I made this bro-chatbot - not just for me, but for anyone who's having a tough time and needs a supportive voice. Sometimes we all need that chill friend who's got our back, you know?

If you're going through something difficult, I hope this little AI bro can help you feel a bit better. We all deserve someone who believes in us, even when we don't believe in ourselves.

**🤖 Try the model**: [doublebank/bro-chatbot](https://huggingface.co/doublebank/bro-chatbot) on HuggingFace

## Project Overview
This project creates a conversational AI chatbot with a distinctive "bro" personality - casual, chilled, and supportive. The bot responds to various topics while maintaining a laid-back, friendly, and encouraging tone.

## Personality Traits
- **Casual**: Uses relaxed language, slang, and informal expressions
- **Chilled**: Maintains a calm, easy-going attitude in all responses
- **Supportive**: Always encouraging and positive, offering help and motivation

## Tech Stack
- **Framework**: Unsloth for efficient LLM fine-tuning
- **Method**: Supervised Fine-Tuning (SFT)
- **Language**: Python
- **Data Format**: Conversational pairs in JSON format

## Quick Start

### Use the Pre-trained Model
```python
from hg_chat import HGChat

# Load the bro-chatbot from HuggingFace
bro_chat = HGChat("doublebank/bro-chatbot")

# Chat with your AI bro
response = bro_chat.chat("I'm having a rough day, bro")
print(response)

# Or stream the response
bro_chat.chat_stream("Tell me everything will be okay")
```

### Train Your Own
1. Install dependencies: `pip install unsloth transformers torch`
2. Review training data in `training_data.json`
3. Follow the detailed plan in `PLAN.md`
4. Run the fine-tuning script

## Training Data
The training dataset contains 57 conversation pairs covering topics like:
- General knowledge questions
- Personal advice and motivation
- Technical explanations
- Casual conversations
- Problem-solving scenarios
- Emotional support and encouragement

All responses maintain the bro-like personality while being helpful and informative.

## For Anyone Who Needs This

If you're reading this and you're going through a tough time - you're not alone. We all have days where everything seems to go wrong. This little AI bro might not solve your problems, but maybe it can remind you that you're stronger than you think.

Take care of yourself, and remember: tomorrow is a new day, bro. 💪

---

*Made with ❤️ for everyone who needs a supportive friend*
