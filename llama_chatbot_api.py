import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Always use CPU
device = torch.device("cpu")

# Path to your converted LLaMA 3.2B model
model_path = "C:/Users/hasan/.llama/hf_format/Llama3.2-3B"

# Load tokenizer and model to CPU
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32  # Use float32 for CPU
).to(device)

# Simple FastAPI app
app = FastAPI()
chat_history = []  # Optional persistent history

# Input schema for API
class ChatInput(BaseModel):
    user: str

# Text generation function
def generate_reply(prompt, history=None, max_length=256):
    if history:
        full_prompt = "\n".join(history + [f"User: {prompt}", "AI:"])
    else:
        full_prompt = f"User: {prompt}\nAI:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_k=40,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("AI:")[-1].strip()
    return response

# Chat endpoint
@app.post("/chat")
def chat(input: ChatInput):
    user_message = input.user
    chat_history.append(f"User: {user_message}")
    ai_response = generate_reply(user_message, history=chat_history)
    chat_history.append(f"AI: {ai_response}")
    return {"response": ai_response}
