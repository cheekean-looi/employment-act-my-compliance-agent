# Employment Act Malaysia Compliance Agent - Production Dockerfile
# Hour 2 deployment with configurable retrieval parameters

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY environment.yml .

# Install Python dependencies
# Note: In production, prefer conda for FAISS compatibility
RUN pip install --no-cache-dir -r requirements.txt

# Install FAISS CPU (fallback if conda not available)
RUN pip install --no-cache-dir faiss-cpu

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Copy configuration
COPY .env.example .env

# Create data directories
RUN mkdir -p data/raw_pdfs data/processed data/indices data/eval outputs

# Hour 2 configurable parameters as build args
ARG BM25_TOPK=30
ARG DENSE_TOPK=20
ARG CE_MAX_PAIRS=40
ARG FINAL_TOPK=8
ARG MIN_CHUNKS=6

# Set environment variables for retrieval configuration
ENV BM25_TOPK=${BM25_TOPK}
ENV DENSE_TOPK=${DENSE_TOPK}
ENV CE_MAX_PAIRS=${CE_MAX_PAIRS}
ENV FINAL_TOPK=${FINAL_TOPK}
ENV MIN_CHUNKS=${MIN_CHUNKS}

# Additional model and API configuration
ENV MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
ENV EMBEDDING_MODEL=intfloat/e5-large-v2
ENV RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENV API_HOST=0.0.0.0
ENV API_PORT=8001
ENV VLLM_PORT=8000

# Expose ports
EXPOSE 8000 8001 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command - can be overridden at runtime
CMD ["python", "-m", "src.generation.rag_pipeline", "--help"]

# Production deployment examples:
# 
# Build with Hour 2 spec parameters:
# docker build --build-arg BM25_TOPK=100 --build-arg DENSE_TOPK=50 --build-arg CE_MAX_PAIRS=150 -t ea-agent:hour2-spec .
#
# Build with optimized parameters (default):
# docker build -t ea-agent:optimized .
#
# Run RAG pipeline:
# docker run -v $(pwd)/data:/app/data ea-agent:optimized python -m src.generation.rag_pipeline --faiss data/indices/faiss.index --store data/indices/store.pkl --query "annual leave entitlement"
#
# Run API server:
# docker run -p 8001:8001 -v $(pwd)/data:/app/data ea-agent:optimized uvicorn src.server.api:app --host 0.0.0.0 --port 8001
#
# Override parameters at runtime:
# docker run -e BM25_TOPK=100 -e DENSE_TOPK=50 -v $(pwd)/data:/app/data ea-agent:optimized python -m src.generation.rag_pipeline --faiss data/indices/faiss.index --store data/indices/store.pkl --query "annual leave"