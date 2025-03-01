# Setup environment
import os
os.environ["HUGGING_FACE_HUB_TOKEN"] = "HF TOKEN"

# Initialize model and tokenizers
model, bert_tokenizer, llama_tokenizer = prepare_model_and_tokenizers()

# Prepare input
system_prompt = "You are a helpful AI assistant."
user_message = "What is machine learning?"
formatted_prompt = model.format_chat_prompt(system_prompt, user_message)

# Process input
inputs = process_input(formatted_prompt, bert_tokenizer)

# Generate response
outputs = model.generate(**inputs)
response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)