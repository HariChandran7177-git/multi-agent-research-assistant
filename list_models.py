import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    print("No GROQ_API_KEY found in .env")
    exit(1)

headers = {'Authorization': f'Bearer {api_key}'}
try:
    response = requests.get('https://api.groq.com/openai/v1/models', headers=headers)
    response.raise_for_status()
    models = response.json().get('data', [])
    print("Available Groq Models:")
    for m in models:
        print(f"- {m['id']}")
except Exception as e:
    print(f"Error fetching models: {e}")
