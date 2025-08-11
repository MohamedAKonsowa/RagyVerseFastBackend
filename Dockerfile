FROM python:3.11-slim

# OS deps (FFmpeg for pydub)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_sm

# App code
COPY . .

# Cache location for HF models (optional)
ENV HF_HOME=/app/.cache/huggingface
ENV PORT=10000

# Start
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
