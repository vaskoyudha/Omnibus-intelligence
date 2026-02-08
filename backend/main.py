"""
FastAPI Backend for Omnibus Legal Compass

Indonesian Legal Q&A API with RAG using NVIDIA NIM Llama 3.1.
Provides legal document search, Q&A with citations, and health checks.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.rag_chain import LegalRAGChain, RAGResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global RAG chain instance (initialized on startup)
rag_chain: LegalRAGChain | None = None


# =============================================================================
# Pydantic Models
# =============================================================================


class QuestionRequest(BaseModel):
    """Request model for asking questions."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Pertanyaan tentang peraturan Indonesia",
        examples=["Apa itu Undang-Undang Cipta Kerja?"],
    )
    jenis_dokumen: str | None = Field(
        default=None,
        description="Filter berdasarkan jenis dokumen (UU, PP, Perpres, Perda, dll)",
        examples=["UU", "PP"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Jumlah dokumen yang diambil untuk konteks",
    )


class FollowUpRequest(BaseModel):
    """Request model for follow-up questions with history."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Pertanyaan lanjutan",
    )
    chat_history: list[dict[str, str]] = Field(
        default=[],
        description="Riwayat percakapan sebelumnya",
        examples=[[{"question": "Apa itu UU Cipta Kerja?", "answer": "..."}]],
    )
    jenis_dokumen: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class CitationInfo(BaseModel):
    """Citation information for a source."""

    number: int
    citation_id: str
    citation: str
    score: float
    metadata: dict[str, Any] = {}


class QuestionResponse(BaseModel):
    """Response model for Q&A endpoint."""

    answer: str = Field(description="Jawaban dalam Bahasa Indonesia dengan sitasi")
    citations: list[CitationInfo] = Field(description="Daftar sitasi terstruktur")
    sources: list[str] = Field(description="Daftar sumber dalam format ringkas")
    confidence: str = Field(
        description="Tingkat kepercayaan: tinggi, sedang, rendah, tidak ada"
    )
    processing_time_ms: float = Field(description="Waktu pemrosesan dalam milidetik")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    qdrant_connected: bool
    llm_configured: bool
    collection_count: int | None
    version: str


# =============================================================================
# Lifespan Event Handler
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    global rag_chain

    logger.info("Starting up Omnibus Legal Compass API...")

    try:
        # Initialize RAG chain (this also initializes retriever and LLM client)
        rag_chain = LegalRAGChain()
        logger.info("RAG chain initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}")
        # Don't raise - allow app to start for health checks
        rag_chain = None

    yield

    # Cleanup
    logger.info("Shutting down Omnibus Legal Compass API...")
    rag_chain = None


# =============================================================================
# FastAPI Application
# =============================================================================


app = FastAPI(
    title="Omnibus Legal Compass API",
    description="""
    API untuk sistem Q&A hukum Indonesia menggunakan RAG (Retrieval-Augmented Generation).
    
    ## Fitur Utama
    
    - **Tanya Jawab Hukum**: Ajukan pertanyaan tentang peraturan Indonesia dan dapatkan jawaban dengan sitasi
    - **Filter Dokumen**: Filter berdasarkan jenis dokumen (UU, PP, Perpres, dll)
    - **Percakapan Lanjutan**: Dukung pertanyaan lanjutan dengan konteks percakapan
    
    ## Teknologi
    
    - NVIDIA NIM dengan Llama 3.1 8B Instruct
    - Qdrant Vector Database dengan Hybrid Search
    - HuggingFace Embeddings (paraphrase-multilingual-MiniLM-L12-v2)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# CORS Middleware - allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Terjadi kesalahan internal. Silakan coba lagi.",
        },
    )


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns system status including Qdrant connection and LLM configuration.
    """
    global rag_chain

    qdrant_connected = False
    collection_count = None
    llm_configured = False

    if rag_chain is not None:
        try:
            # Check Qdrant connection
            collection_info = rag_chain.retriever.qdrant_client.get_collection(
                rag_chain.retriever.collection_name
            )
            qdrant_connected = True
            collection_count = collection_info.points_count
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")

        # Check LLM client
        llm_configured = rag_chain.llm_client is not None

    return HealthResponse(
        status="healthy" if (qdrant_connected and llm_configured) else "degraded",
        qdrant_connected=qdrant_connected,
        llm_configured=llm_configured,
        collection_count=collection_count,
        version="1.0.0",
    )


@app.post("/api/ask", response_model=QuestionResponse, tags=["Q&A"])
async def ask_question(request: QuestionRequest):
    """
    Tanya jawab hukum Indonesia.

    Ajukan pertanyaan tentang peraturan perundang-undangan Indonesia.
    Jawaban akan disertai dengan sitasi ke dokumen sumber.

    ## Contoh Request

    ```json
    {
        "question": "Apa itu Undang-Undang Cipta Kerja?",
        "jenis_dokumen": null,
        "top_k": 5
    }
    ```

    ## Response

    - **answer**: Jawaban dalam Bahasa Indonesia dengan sitasi [1], [2], dll
    - **citations**: Detail sitasi terstruktur
    - **sources**: Daftar sumber ringkas
    - **confidence**: Tingkat kepercayaan berdasarkan kualitas retrieval
    """
    global rag_chain

    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain not initialized. Please check system health.",
        )

    start_time = time.perf_counter()

    try:
        # Query RAG chain
        response: RAGResponse = rag_chain.query(
            question=request.question,
            filter_jenis_dokumen=request.jenis_dokumen,
            top_k=request.top_k,
        )

        processing_time = (time.perf_counter() - start_time) * 1000

        # Convert citations to Pydantic models
        citations = [
            CitationInfo(
                number=c["number"],
                citation_id=c["citation_id"],
                citation=c["citation"],
                score=c["score"],
                metadata=c.get("metadata", {}),
            )
            for c in response.citations
        ]

        return QuestionResponse(
            answer=response.answer,
            citations=citations,
            sources=response.sources,
            confidence=response.confidence,
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}",
        )


@app.post("/api/ask/followup", response_model=QuestionResponse, tags=["Q&A"])
async def ask_followup(request: FollowUpRequest):
    """
    Pertanyaan lanjutan dengan konteks percakapan.

    Gunakan endpoint ini untuk pertanyaan lanjutan yang membutuhkan
    konteks dari percakapan sebelumnya.

    ## Contoh Request

    ```json
    {
        "question": "Bagaimana dengan sanksinya?",
        "chat_history": [
            {
                "question": "Apa itu UU Cipta Kerja?",
                "answer": "UU Cipta Kerja adalah..."
            }
        ]
    }
    ```
    """
    global rag_chain

    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain not initialized. Please check system health.",
        )

    start_time = time.perf_counter()

    try:
        response: RAGResponse = rag_chain.query_with_history(
            question=request.question,
            chat_history=request.chat_history,
            filter_jenis_dokumen=request.jenis_dokumen,
            top_k=request.top_k,
        )

        processing_time = (time.perf_counter() - start_time) * 1000

        citations = [
            CitationInfo(
                number=c["number"],
                citation_id=c["citation_id"],
                citation=c["citation"],
                score=c["score"],
                metadata=c.get("metadata", {}),
            )
            for c in response.citations
        ]

        return QuestionResponse(
            answer=response.answer,
            citations=citations,
            sources=response.sources,
            confidence=response.confidence,
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        logger.error(f"Error processing followup: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process followup: {str(e)}",
        )


@app.get("/api/document-types", tags=["Metadata"])
async def get_document_types():
    """
    Daftar jenis dokumen yang tersedia untuk filter.

    Returns daftar jenis dokumen (UU, PP, Perpres, dll) yang dapat
    digunakan sebagai filter pada endpoint /api/ask.
    """
    return {
        "document_types": [
            {"code": "UU", "name": "Undang-Undang"},
            {"code": "PP", "name": "Peraturan Pemerintah"},
            {"code": "Perpres", "name": "Peraturan Presiden"},
            {"code": "Perda", "name": "Peraturan Daerah"},
            {"code": "Permen", "name": "Peraturan Menteri"},
            {"code": "Kepmen", "name": "Keputusan Menteri"},
            {"code": "Pergub", "name": "Peraturan Gubernur"},
            {"code": "Perbup", "name": "Peraturan Bupati"},
            {"code": "Perwal", "name": "Peraturan Walikota"},
        ]
    }


@app.get("/", tags=["System"])
async def root():
    """Root endpoint - API info."""
    return {
        "name": "Omnibus Legal Compass API",
        "version": "1.0.0",
        "description": "Indonesian Legal Q&A API with RAG",
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
