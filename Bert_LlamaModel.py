# model.py
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from typing import Optional, Dict
import re

class ScriptLLaMA(nn.Module):
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        super(ScriptLLaMA, self).__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Add special tokens for script formatting
        special_tokens = {
            "additional_special_tokens": [
                "[SCENE]", "[/SCENE]",
                "[ACTION]", "[/ACTION]",
                "[CHARACTER]", "[/CHARACTER]",
                "[DIALOGUE]", "[/DIALOGUE]",
                "[GENRE]", "[/GENRE]"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load model with 8-bit quantization
        self.llama = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Prepare model for k-bit training
        self.llama = prepare_model_for_kbit_training(self.llama)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.llama = get_peft_model(self.llama, lora_config)
        
        # Resize token embeddings
        self.llama.resize_token_embeddings(len(self.tokenizer))
        
        self.system_prompt = (
            "You are a professional screenplay writer. "
            "Write engaging and creative movie scripts in the specified genre. "
            "Follow standard screenplay formatting using scene headings (INT./EXT.), "
            "action descriptions, character names, and dialogue."
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs
                
    def format_prompt(self, genre: str) -> str:
        """Format the input prompt with genre and system instructions"""
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        
        system_prompt = f"{B_SYS}{self.system_prompt}{E_SYS}"
        user_prompt = (
            f"Write a movie script in the {genre} genre. "
            f"Use [SCENE] for scene headings, [ACTION] for action descriptions, "
            f"[CHARACTER] for character names, and [DIALOGUE] for dialogue."
        )
        
        return f"{B_INST} {system_prompt}{user_prompt} {E_INST}"

    def generate_script(
        self,
        genre: str,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> str:
        """Generate a movie script based on genre"""
        prompt = self.format_prompt(genre)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llama.device)
        
        outputs = self.llama.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        generated_script = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Clean up the generated script
        generated_script = self.clean_script(generated_script)
        
        return generated_script
    
    def clean_script(self, script: str) -> str:
        """Clean up the generated script and ensure proper formatting"""
        # Remove the prompt
        script = script.split("[/INST]")[-1].strip()
        
        # Ensure proper spacing around special tokens
        script = re.sub(r'\[SCENE\]', '\n[SCENE]', script)
        script = re.sub(r'\[ACTION\]', '\n[ACTION]', script)
        script = re.sub(r'\[CHARACTER\]', '\n[CHARACTER]', script)
        script = re.sub(r'\[DIALOGUE\]', '\n[DIALOGUE]', script)
        
        # Remove extra newlines
        script = re.sub(r'\n{3,}', '\n\n', script)
        
        return script.strip()

def format_script_for_display(script: str) -> str:
    """Format the script for display by removing special tokens"""
    script = re.sub(r'\[SCENE\](.+?)\[/SCENE\]', r'SCENE: \1', script)
    script = re.sub(r'\[ACTION\](.+?)\[/ACTION\]', r'ACTION: \1', script)
    script = re.sub(r'\[CHARACTER\](.+?)\[/CHARACTER\]', r'CHARACTER: \1', script)
    script = re.sub(r'\[DIALOGUE\](.+?)\[/DIALOGUE\]', r'DIALOGUE: \1', script)
    script = re.sub(r'\[GENRE\](.+?)\[/GENRE\]', r'GENRE: \1', script)
    return script