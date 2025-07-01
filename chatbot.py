import json
import faiss
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import requests

# Load FAISS and metadata
index = faiss.read_index("doc_index.faiss")

with open("doc_metadata.json", "r", encoding="utf-8") as f:
    doc_chunks = json.load(f)

texts = [item["content"] for item in doc_chunks]

# Embedding model
model = SentenceTransformer('paraphase-MiniLM-L3-v2')

# OpenRouter config
OPENROUTER_API_KEY = "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"
MODEL_NAME = "mistralai/mistral-small-3.2-24b-instruct:free"

# FastAPI app
app = FastAPI()

class ChatRequest(BaseModel):
    query: str

def search_faiss(query, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    results = []
    for i, dist in zip(indices[0], distances[0]):
        if i != -1:
            results.append((texts[i], dist))
    return results

def generate_prompt(query, context_chunks):
    context = "\n\n".join(f"- {chunk}" for chunk, _ in context_chunks)
    return f"""You are an assistant who answers only using the provided document context. 
If the answer isn't in the context, say "Sorry, I don't know."

### Context:
{context}

### Question:
{query}

### Answer:"""

@app.post("/chat")
async def chat(req: ChatRequest):
    query = req.query
    matches = search_faiss(query)

    # Optional: use distance threshold to reject weak matches
    if not matches or matches[0][1] > 1.5:
        return {"answer": "Sorry, I don't know."}

    prompt = generate_prompt(query, matches)

    # Call OpenRouter
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "yourdomain.com",  # Replace with your domain or testing
        "X-Title": "InternalDocsBot"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        return {"answer": reply.strip()}
    else:
        return {"answer": "Error contacting LLM. Try again later."}
