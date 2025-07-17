from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from Model.llama_model import LlamaModel

CUSTOM_CACHE_DIR = "Model Path"
os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)


HF_AUTH_TOKEN = "X"
# Load LLaMA 3 model (quantized recommended)
model_id = "meta-llama/Llama-3.2-3B"


llama_model = LlamaModel(model_id, CUSTOM_CACHE_DIR, HF_AUTH_TOKEN)
model, tokenizer = llama_model.get_model()


llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

def generate_response(prompt):
    response = llm_pipeline(prompt)[0]['generated_text']
    return response[len(prompt):].strip()
