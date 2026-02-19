# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for PyMuPDF / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libffi-dev libssl-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

# Pre-download models at build time so cold starts are instant
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ARG LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
RUN python - <<'EOF'
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

embed = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
llm   = os.environ.get("LLM_MODEL",   "Qwen/Qwen2.5-0.5B-Instruct")

print(f"Downloading embedding model: {embed}")
SentenceTransformer(embed)

print(f"Downloading LLM: {llm}")
AutoTokenizer.from_pretrained(llm)
AutoModelForCausalLM.from_pretrained(llm)

print("Models cached ✓")
EOF

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]