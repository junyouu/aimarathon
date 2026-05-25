import os
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from dotenv import load_dotenv

try:
    load_dotenv(override=True, encoding='utf-8')
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")
    
api_key = os.getenv("CHUTES_API_KEY")
if not api_key:
    raise ValueError("CHUTES_API_KEY not set in .env")

class LLM:
  def __init__(self):
      self.client = OpenAI(
        api_key=api_key,
        base_url="https://llm.chutes.ai/v1"
      )

      self.embedding_model = SentenceTransformer(
          "BAAI/bge-base-en-v1.5"
      )

  def generate(self, prompt, max_tokens=512, temperature=0.1):
      response = self.client.chat.completions.create(
          model="Qwen/Qwen3-32B-TEE",
          messages=[
              {"role": "user", "content": prompt}
          ]
      )

      cleaned = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content, flags=re.DOTALL)
      return cleaned.strip()

  def embedding(self, content_to_embed):
      return self.embedding_model.encode(content_to_embed).tolist()

llm = LLM()
