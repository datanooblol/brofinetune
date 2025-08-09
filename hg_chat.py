from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
import torch

class BaseLLM(ABC):
    @abstractmethod
    def run(self, system_prompt: str, messages: list) -> str:
        pass

class HGChat(BaseLLM):
    def __init__(self, model_name: str, max_seq_length: int = 2048, load_in_4bit: bool = True):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model, self.tokenizer = self.load_model(model_name)
        
    def load_model(self, model_name: str):
        """Load model and tokenizer from HuggingFace"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        
        # Setup chat template and inference mode
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
        FastLanguageModel.for_inference(model)
        
        return model, tokenizer
    
    def UserMessage(self, text: str, **kwargs) -> Dict[str, Any]:
        """Create user message"""
        return {"role": "user", "content": text}
    
    def AIMessage(self, text: str) -> Dict[str, Any]:
        """Create assistant message"""
        return {"role": "assistant", "content": text}
    
    def SystemMessage(self, text: str) -> Dict[str, Any]:
        """Create system message"""
        return {"role": "system", "content": text}
    
    def run(self, system_prompt: Optional[str] = None, messages: List[Dict[str, str]] = None, 
            max_new_tokens: int = 256, temperature: float = 0.7, min_p: float = 0.1) -> str:
        """Generate response from messages"""
        if messages is None:
            messages = []
            
        # Add system prompt if provided
        if system_prompt:
            messages = [self.SystemMessage(system_prompt)] + messages
            
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                min_p=min_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Extract only the new response
        input_length = inputs.shape[1]
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return response.strip()
    
    def stream(self, system_prompt: Optional[str] = None, messages: List[Dict[str, str]] = None,
               max_new_tokens: int = 256, temperature: float = 0.7, min_p: float = 0.1,
               return_full_text: bool = False):
        """Stream response in real-time, optionally return complete text"""
        if messages is None:
            messages = []
            
        # Add system prompt if provided
        if system_prompt:
            messages = [self.SystemMessage(system_prompt)] + messages
            
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        if return_full_text:
            # Custom streamer that captures text
            class CaptureStreamer(TextStreamer):
                def __init__(self, tokenizer, **kwargs):
                    super().__init__(tokenizer, **kwargs)
                    self.captured_text = ""
                    
                def put(self, value):
                    super().put(value)
                    if len(value.shape) > 1 and value.shape[0] == 1:
                        text = self.tokenizer.decode(value[0], skip_special_tokens=True)
                        if hasattr(self, 'previous_text'):
                            new_text = text[len(self.previous_text):]
                            self.captured_text += new_text
                        self.previous_text = text
            
            capture_streamer = CaptureStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Generate with capture
            outputs = self.model.generate(
                input_ids=inputs,
                streamer=capture_streamer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                min_p=min_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            # Extract full response
            input_length = inputs.shape[1]
            full_response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            return full_response.strip()
        else:
            # Regular streaming without return
            text_streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            self.model.generate(
                input_ids=inputs,
                streamer=text_streamer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                min_p=min_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
    
    def chat(self, message: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Simple chat interface"""
        messages = [self.UserMessage(message)]
        return self.run(system_prompt=system_prompt, messages=messages, **kwargs)
    
    def chat_stream(self, message: str, system_prompt: Optional[str] = None, 
                   return_full_text: bool = False, **kwargs):
        """Simple streaming chat interface"""
        messages = [self.UserMessage(message)]
        return self.stream(system_prompt=system_prompt, messages=messages, 
                          return_full_text=return_full_text, **kwargs)