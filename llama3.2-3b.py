import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to your converted model
model_path = "C:/Users/hasan/.llama/hf_format/Llama3.2-3B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Text generation function
def generate_text(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            num_return_sequences=1
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "In a future powered by AI,"
print(generate_text(prompt))
