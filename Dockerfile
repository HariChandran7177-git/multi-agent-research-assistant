FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first to avoid the 2GB CUDA download
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the SentenceTransformer model into the image layer
# so first startup doesn't stall. MODEL_CACHE_DIR can be overridden
# at runtime via env var (default: ./model_cache inside container).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./model_cache')"

# Persistent data lives in /app/data (mount a volume here)
RUN mkdir -p /app/data

EXPOSE 7860

CMD ["uvicorn", "api.research_api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
