"""
RAG Microservice Backend - FastAPI Application
Handles document indexing and AI-powered chat with context retrieval.
"""

import os
import uuid
import logging
from typing import Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

import fitz  # PyMuPDF

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("rag-backend")

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_HOST     = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rag_documents")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
LLM_MODEL       = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K           = int(os.getenv("TOP_K", "5"))
MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS", "512"))
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Global model holders ───────────────────────────────────────────────────────
_embed_model: Optional[SentenceTransformer] = None
_llm_tokenizer = None
_llm_model = None
_qdrant: Optional[QdrantClient] = None


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embed_model, _llm_tokenizer, _llm_model, _qdrant

    logger.info("Loading embedding model: %s", EMBED_MODEL)
    _embed_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    logger.info("Loading LLM: %s  (device=%s)", LLM_MODEL, DEVICE)
    _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    _llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cpu":
        _llm_model = _llm_model.to(DEVICE)

    logger.info("Connecting to Qdrant at %s:%s", QDRANT_HOST, QDRANT_PORT)
    _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    _ensure_collection()

    logger.info("All models loaded. Service ready ✓")
    yield

    logger.info("Shutting down – releasing resources.")


def _ensure_collection():
    """Create the Qdrant collection if it doesn't already exist."""
    existing = [c.name for c in _qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        dim = _embed_model.get_sentence_embedding_dimension()
        _qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created collection '%s' (dim=%d)", COLLECTION_NAME, dim)
    else:
        logger.info("Collection '%s' already exists.", COLLECTION_NAME)


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Microservice",
    description="Production-grade Retrieval-Augmented Generation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    top_k: int = TOP_K
    max_new_tokens: int = MAX_NEW_TOKENS


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


class IndexResponse(BaseModel):
    message: str
    chunks_indexed: int
    document_id: str


# ── Utility helpers ────────────────────────────────────────────────────────────
def _extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from PDF or TXT uploads."""
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    return file_bytes.decode("utf-8", errors="replace")


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 20]  # drop trivially small chunks


def _embed(texts: list[str]) -> list[list[float]]:
    return _embed_model.encode(texts, normalize_embeddings=True).tolist()


def _retrieve(query: str, top_k: int) -> list[dict]:
    """Vector search in Qdrant."""
    q_vec = _embed([query])[0]
    hits = _qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vec,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "score": round(h.score, 4),
            "text": h.payload.get("text", ""),
            "source": h.payload.get("source", "unknown"),
            "chunk_id": h.payload.get("chunk_id", 0),
        }
        for h in hits
    ]


def _generate(query: str, context: str, max_new_tokens: int) -> str:
    """Run LLM inference with the retrieved context."""
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the provided context. If the answer is not in the context, say so."
    )
    user_content = f"Context:\n{context}\n\nQuestion: {query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    text = _llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _llm_tokenizer([text], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = _llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=_llm_tokenizer.eos_token_id,
        )

    # Strip prompt tokens from output
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return _llm_tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "collection": COLLECTION_NAME}


@app.post("/index", response_model=IndexResponse)
async def index_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file. The text is chunked, embedded, and stored
    in Qdrant for later retrieval.
    """
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    logger.info("Indexing file: %s", file.filename)
    raw = await file.read()
    text = _extract_text(raw, file.filename)

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from the file.")

    chunks = _chunk_text(text)
    doc_id = str(uuid.uuid4())
    embeddings = _embed(chunks)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "text": chunk,
                "source": file.filename,
                "document_id": doc_id,
                "chunk_id": idx,
            },
        )
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    _qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Indexed %d chunks from '%s' (doc_id=%s)", len(chunks), file.filename, doc_id)

    return IndexResponse(
        message=f"Successfully indexed '{file.filename}'",
        chunks_indexed=len(chunks),
        document_id=doc_id,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Submit a natural-language query. The system retrieves the most relevant
    chunks from Qdrant and uses the LLM to generate a grounded answer.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info("Chat query: %.80s …", request.query)
    sources = _retrieve(request.query, request.top_k)

    if not sources:
        return ChatResponse(
            answer="No relevant documents found. Please index some documents first.",
            sources=[],
        )

    context = "\n\n---\n\n".join(s["text"] for s in sources)
    answer = _generate(request.query, context, request.max_new_tokens)

    logger.info("Answer generated (len=%d chars)", len(answer))
    return ChatResponse(answer=answer, sources=sources)


@app.delete("/collection")
async def reset_collection():
    """Delete and recreate the Qdrant collection (useful for testing)."""
    _qdrant.delete_collection(COLLECTION_NAME)
    _ensure_collection()
    return {"message": f"Collection '{COLLECTION_NAME}' reset successfully."}