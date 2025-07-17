# Model/llama_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class LlamaModel:
    def __init__(self, model_name, cache_dir, hf_token):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.hf_token = hf_token

    def get_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
            device_map="auto",  # Automatically use GPU if available
            torch_dtype="auto",
            trust_remote_code=True
        )
        return model, tokenizer
