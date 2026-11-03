"""
Utility script to list available models for configured LLM providers (Groq and Google Gemini).
Merged from legacy list_models.py and list_gemini_models.py.
"""

import argparse
import os
import requests
from dotenv import load_dotenv

load_dotenv()


def list_groq_models():
    """Query and display available Groq models."""
    api_key = os.getenv("GROQ_API_KEY")
    print("\n" + "=" * 50)
    print("GROQ MODELS")
    print("=" * 50)

    if not api_key:
        print("[!] No GROQ_API_KEY found in environment / .env")
        return

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
        response.raise_for_status()
        models = response.json().get("data", [])
        for m in sorted(models, key=lambda x: x.get("id", "")):
            print(f"- {m.get('id')}")
        print(f"\nTotal Groq models found: {len(models)}")
    except Exception as e:
        print(f"[!] Error fetching Groq models: {e}")


def list_gemini_models():
    """Query and display available Google Gemini models for text generation."""
    api_key = os.getenv("GOOGLE_API_KEY")
    print("\n" + "=" * 50)
    print("GOOGLE GEMINI MODELS")
    print("=" * 50)

    if not api_key:
        print("[!] No GOOGLE_API_KEY found in environment / .env")
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])

        count = 0
        for m in models:
            if "generateContent" in m.get("supportedGenerationMethods", []):
                model_id = m.get("name", "").replace("models/", "")
                display_name = m.get("displayName", "No Name")
                print(f"- {model_id} ({display_name})")
                count += 1
        print(f"\nTotal Gemini generation models found: {count}")
    except Exception as e:
        print(f"[!] Error fetching Gemini models: {e}")


def main():
    parser = argparse.ArgumentParser(description="List available models from configured LLM providers.")
    parser.add_argument(
        "--provider",
        choices=["all", "groq", "gemini"],
        default="all",
        help="Specify which provider models to list (default: all)",
    )
    args = parser.parse_args()

    if args.provider in ("all", "groq"):
        list_groq_models()

    if args.provider in ("all", "gemini"):
        list_gemini_models()


if __name__ == "__main__":
    main()
