# Inside llama_client.py
import requests

def ask_llama(prompt, max_tokens=800):
    response = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "num_predict": max_tokens,
            "stream": False
        }
    )
    return response.json()["response"]
