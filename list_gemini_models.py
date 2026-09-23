import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("No GOOGLE_API_KEY found in .env")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
try:
    response = requests.get(url)
    response.raise_for_status()
    models = response.json().get('models', [])
    
    print("Available Gemini Models for Text Generation:")
    for m in models:
        # Check if the model supports generateContent (text generation)
        if 'generateContent' in m.get('supportedGenerationMethods', []):
            model_id = m['name'].replace('models/', '')
            display_name = m.get('displayName', 'No Name')
            print(f"- {model_id} (Display Name: {display_name})")
except Exception as e:
    print(f"Error fetching Gemini models: {e}")
