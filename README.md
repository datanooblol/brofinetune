# brofinetune
A repo for fine-tuning a bro-like conversational chatbot using Supervised Fine-Tuning (SFT) with Unsloth.

## Project Overview
This project aims to create a conversational AI chatbot with a distinctive "bro" personality - casual, chilled, and supportive. The bot should respond to various topics while maintaining a laid-back, friendly, and encouraging tone.

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
1. Install dependencies: `pip install unsloth`
2. Review training data in `training_data.json`
3. Follow the detailed plan in `PLAN.md`
4. Run the fine-tuning script

## Training Data
The training dataset contains 100+ diverse conversation examples covering topics like:
- General knowledge questions
- Personal advice and motivation
- Technical explanations
- Casual conversations
- Problem-solving scenarios

All responses maintain the bro-like personality while being helpful and informative.
