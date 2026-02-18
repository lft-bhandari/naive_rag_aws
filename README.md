# RAG Microservices – Production Deployment

> **Retrieval-Augmented Generation** system built with FastAPI, Streamlit, Qdrant, BAAI/bge-small, and Qwen-0.5B – fully automated via GitHub Actions CI/CD onto AWS EC2.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Quick Start (Local)](#quick-start-local)
5. [EC2 Provisioning](#ec2-provisioning)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [API Reference](#api-reference)
8. [Configuration](#configuration)
9. [Logging](#logging)
10. [Troubleshooting](#troubleshooting)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                             AWS Cloud                                        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        EC2 Instance (t3.xlarge)                        │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                   Docker Compose Network                        │  │  │
│  │  │                                                                 │  │  │
│  │  │  ┌──────────┐    ┌───────────────────┐    ┌─────────────────┐  │  │  │
│  │  │  │          │    │                   │    │                 │  │  │  │
│  │  │  │  Nginx   │───▶│  Streamlit UI     │    │  FastAPI        │  │  │  │
│  │  │  │  :80     │    │  Frontend :8501   │    │  Backend :8000  │  │  │  │
│  │  │  │          │    │                   │    │                 │  │  │  │
│  │  │  └──────────┘    └───────────────────┘    └────────┬────────┘  │  │  │
│  │  │       │               │  (HTTP REST)            │   │          │  │  │
│  │  │       └───────────────┘                         │   │          │  │  │
│  │  │                                        Embed    │   │ Generate │  │  │
│  │  │                                     ┌───────────┘   └──────┐   │  │  │
│  │  │                                     ▼                      ▼   │  │  │
│  │  │                          ┌─────────────────┐    ┌──────────────┐│  │  │
│  │  │                          │  BAAI/bge-small  │    │ Qwen-0.5B   ││  │  │
│  │  │                          │  Embeddings      │    │ LLM         ││  │  │
│  │  │                          └────────┬─────────┘    └──────────────┘│  │  │
│  │  │                                   │ Store/Search                  │  │  │
│  │  │                                   ▼                               │  │  │
│  │  │                          ┌─────────────────┐                      │  │  │
│  │  │                          │  Qdrant          │                      │  │  │
│  │  │                          │  Vector DB :6333 │                      │  │  │
│  │  │                          │  (persisted vol) │                      │  │  │
│  │  │                          └─────────────────┘                      │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │   CloudWatch Logs Agent ──────────────────────────────────────────────▶│  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                  │                                                            │
│                  ▼                                                            │
│         ┌─────────────────┐                                                  │
│         │  CloudWatch Logs│                                                  │
│         │  /rag-microsvcs │                                                  │
│         └─────────────────┘                                                  │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          CI/CD Pipeline                                      │
│                                                                              │
│  Git Push → GitHub Actions                                                   │
│                │                                                             │
│    ┌───────────▼───────────┐                                                 │
│    │  Job 1: Lint & Test   │  (ruff + import checks)                        │
│    └───────────┬───────────┘                                                 │
│                │                                                             │
│    ┌───────────▼───────────┐                                                 │
│    │  Job 2: Build & Push  │  (docker buildx → GHCR)                        │
│    │  backend + frontend   │                                                 │
│    └───────────┬───────────┘                                                 │
│                │                                                             │
│    ┌───────────▼───────────┐                                                 │
│    │  Job 3: Deploy EC2    │  (SSH → docker compose pull → up -d)           │
│    │  + health check       │                                                 │
│    └───────────────────────┘                                                 │
└──────────────────────────────────────────────────────────────────────────────┘

RAG Data Flow
─────────────
                   ┌────────────┐
 User uploads PDF  │            │
 ──────────────────▶  /index   │
                   │  endpoint  │
                   └─────┬──────┘
                         │ 1. Extract text (PyMuPDF)
                         │ 2. Chunk (512 tokens, 64 overlap)
                         │ 3. Embed (bge-small-en-v1.5)
                         │ 4. Upsert → Qdrant
                         ▼
                   ┌────────────┐
 User asks query   │            │
 ──────────────────▶  /chat    │
                   │  endpoint  │
                   └─────┬──────┘
                         │ 1. Embed query
                         │ 2. Cosine search → top-k chunks
                         │ 3. Build prompt: system + context + query
                         │ 4. Generate → Qwen-0.5B
                         │ 5. Return answer + sources
                         ▼
                   ┌────────────┐
                   │  Response  │
                   │ + sources  │
                   └────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Qwen/Qwen2.5-0.5B-Instruct |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector DB | Qdrant (self-hosted) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Reverse Proxy | Nginx |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Registry | GitHub Container Registry (GHCR) |
| Cloud | AWS EC2 + CloudWatch Logs |

---

## Project Structure

```
rag-microservices/
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD pipeline
├── backend/
│   ├── main.py                 # FastAPI app (index + chat endpoints)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py                  # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── nginx/
│   └── nginx.conf              # Reverse proxy config
├── scripts/
│   └── provision_ec2.sh        # EC2 bootstrap script
├── docker-compose.yml          # Full stack definition
├── .gitignore
└── README.md                   ← you are here
```

---

## Quick Start (Local)

### Prerequisites

- Docker ≥ 24 and Docker Compose v2
- 8 GB RAM (16 GB recommended)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_ORG/rag-microservices.git
cd rag-microservices
```

### 2. (Optional) Customise environment

```bash
cp .env.example .env   # edit model names, ports, chunk sizes as needed
```

### 3. Build & launch

```bash
docker compose up --build -d
```

The first build downloads model weights (~1.5 GB) – expect 5–10 minutes.

### 4. Open the UI

Navigate to **http://localhost/** or **http://localhost:8501/** in your browser.

### 5. Index a document & chat

1. In the sidebar, upload a PDF or TXT file.
2. Click **Index Selected Files**.
3. Type a question in the chat box and press Enter.

---

## EC2 Provisioning

### Recommended AMI & instance

| Field | Value |
|---|---|
| AMI | Ubuntu 22.04 LTS (`ami-0c7217cdde317cfec`, us-east-1) |
| Instance type | `t3.xlarge` (4 vCPU / 16 GB) – CPU inference |
| Storage | 50 GB gp3 root volume |
| Security group | Allow SSH (22), HTTP (80), 8000, 8501 from your IP |

### Step 1 – Launch the instance via AWS Console or CLI

```bash
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type t3.xlarge \
  --key-name YOUR_KEY_PAIR \
  --security-group-ids sg-XXXXXXXX \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=rag-microservices}]'
```

### Step 2 – Run the provisioning script

```bash
# Copy script to instance
scp -i ~/.ssh/YOUR_KEY.pem scripts/provision_ec2.sh ubuntu@<EC2_PUBLIC_IP>:~

# SSH in and run
ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@<EC2_PUBLIC_IP>
sudo REPO_URL=https://github.com/YOUR_ORG/rag-microservices.git \
     BRANCH=main \
     bash provision_ec2.sh
```

The script will:
1. Install Docker Engine & Docker Compose v2
2. Clone the repository to `/opt/rag-microservices`
3. Configure UFW firewall
4. Install and configure the AWS CloudWatch Logs agent
5. Register a systemd service (`rag-stack`) that auto-starts on boot
6. Build and start all containers

### Step 3 – Verify

```bash
# Check services
docker compose -f /opt/rag-microservices/docker-compose.yml ps

# Tail logs
docker compose -f /opt/rag-microservices/docker-compose.yml logs -f

# API health
curl http://<EC2_PUBLIC_IP>/health
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/deploy.yml`) runs on every push to `main`.

### Required GitHub Secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret | Description |
|---|---|
| `EC2_SSH_PRIVATE_KEY` | Contents of your `.pem` private key |
| `EC2_KNOWN_HOSTS` | Output of `ssh-keyscan -H <EC2_PUBLIC_IP>` |
| `EC2_PUBLIC_IP` | Public IPv4 of your EC2 instance |

### Generating `EC2_KNOWN_HOSTS`

```bash
ssh-keyscan -H <EC2_PUBLIC_IP>
# Copy the output and paste it as the secret value
```

### Pipeline stages

```
push to main
     │
     ├─ Job 1: test ──── ruff lint + syntax check
     │
     ├─ Job 2: build-and-push
     │           ├─ Build backend image → ghcr.io/ORG/rag-backend:latest
     │           └─ Build frontend image → ghcr.io/ORG/rag-frontend:latest
     │
     └─ Job 3: deploy
                 ├─ SCP docker-compose.yml + nginx.conf to EC2
                 ├─ SSH: docker compose pull + up -d --remove-orphans
                 └─ HTTP health check against /health
```

---

## API Reference

Interactive docs available at `http://<HOST>:8000/docs`

### `POST /index`

Upload a PDF or TXT file to be chunked, embedded, and stored in Qdrant.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | File | PDF or TXT document |

**Response:**

```json
{
  "message": "Successfully indexed 'report.pdf'",
  "chunks_indexed": 42,
  "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

---

### `POST /chat`

Ask a question; receives a grounded answer with source attribution.

**Request body:**

```json
{
  "query": "What are the main findings of the report?",
  "top_k": 5,
  "max_new_tokens": 512
}
```

**Response:**

```json
{
  "answer": "The report concludes that…",
  "sources": [
    {
      "score": 0.9123,
      "text": "…chunk text…",
      "source": "report.pdf",
      "chunk_id": 7
    }
  ]
}
```

---

### `GET /health`

Returns service status (used by load balancers and the CI health check).

```json
{ "status": "ok", "device": "cpu", "collection": "rag_documents" }
```

---

### `DELETE /collection`

Wipes and recreates the Qdrant collection. Useful during development.

---

## Configuration

All values can be set via environment variables (in `.env` or in `docker-compose.yml`):

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `qdrant` | Qdrant service hostname |
| `QDRANT_PORT` | `6333` | Qdrant REST port |
| `QDRANT_COLLECTION` | `rag_documents` | Collection name |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `LLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace LLM |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Retrieved chunks per query |
| `MAX_NEW_TOKENS` | `512` | LLM generation budget |
| `API_BASE_URL` | `http://backend:8000` | Frontend → Backend URL |

---

## Logging

### Container logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f qdrant
```

### CloudWatch Logs

Logs are forwarded to CloudWatch under:

- `/rag-microservices/docker` – container stdout/stderr
- `/rag-microservices/provision` – provisioning script output

View in the AWS Console under **CloudWatch → Log groups**.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Backend container restarts | Models still loading – wait up to 3 min; `docker compose logs backend` |
| `connection refused` on port 80 | Check `docker compose ps`; Nginx depends on frontend health |
| GPU not detected | Ensure `nvidia-docker2` installed; add `deploy.resources.reservations.devices` in compose |
| `No relevant documents found` | Upload and index a document first via the sidebar |
| Out of memory | Use `t3.xlarge` (16 GB) or reduce `MAX_NEW_TOKENS` |
| SSH deploy fails | Regenerate `EC2_KNOWN_HOSTS` after stopping/starting instance (IP changes) |

---

## License

MIT – see [LICENSE](LICENSE) for details.