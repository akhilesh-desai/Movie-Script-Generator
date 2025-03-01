import sys
sys.path.append("/content/drive/MyDrive/Bert_Llama_Model")
import argparse
import torch
from transformers import GenerationConfig
from peft import PeftModel, PeftConfig
from Bert_LlamaModel import ScriptLLaMA
import os
import warnings
warnings.filterwarnings("ignore")

def generate_script(
    genre: str,
    checkpoint_path: str,
    output_path: str,
    max_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
):
    # Initialize base model
    model = ScriptLLaMA()
    
    # Load the PEFT adapter
    model.llama = PeftModel.from_pretrained(
        model.llama,
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Set up generation config
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        max_length=max_length,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
    )
    
    # Prepare prompt
    prompt = (
        f"[GENRE]{genre}[/GENRE]\n"
        "[SCENE]"
    )
    
    # Tokenize input
    inputs = model.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(model.llama.device)
    
    # Generate
    with torch.inference_mode():
        outputs = model.llama.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    # Decode generated text
    generated_text = model.tokenizer.decode(
        outputs.sequences[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True
    )
    
    # Format the script
    formatted_script = format_script_for_display(generated_text)
    
    # Save script
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_script)
    
    return formatted_script

def format_script_for_display(script: str) -> str:
    """Format the script by cleaning up special tokens and formatting"""
    # Remove the initial prompt/genre specification if present
    script = script.split("[/GENRE]")[-1].strip()
    
    # Basic formatting
    script = script.replace("[SCENE]", "\nSCENE: ")
    script = script.replace("[/SCENE]", "\n")
    script = script.replace("[ACTION]", "\nACTION: ")
    script = script.replace("[/ACTION]", "\n")
    script = script.replace("[CHARACTER]", "\nCHARACTER: ")
    script = script.replace("[/CHARACTER]", "\n")
    script = script.replace("[DIALOGUE]", "\nDIALOGUE: ")
    script = script.replace("[/DIALOGUE]", "\n")
    
    # Remove any remaining special tokens
    script = script.replace("<s>", "").replace("</s>", "")
    
    # Clean up multiple newlines
    script = "\n".join(line.strip() for line in script.split("\n") if line.strip())
    
    return script

def main():
    parser = argparse.ArgumentParser(description="Generate movie scripts")
    parser.add_argument("--genre", type=str, required=True, help="Movie genre")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length of generated script")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate script
    generated_script = generate_script(
        genre=args.genre,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    print(f"\nScript generated and saved to {args.output}")
    print("\nFirst few lines of the generated script:")
    print("-" * 50)
    print("\n".join(generated_script.split("\n")[:10]))
    print("-" * 50)

if __name__ == "__main__":
    main()