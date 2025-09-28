# Employment Act Malaysia Compliance Agent - Hour 2 Makefile
# Deployment and testing targets for Hour 2 deliverables

.PHONY: help setup smoke test-hour2 docker-build docker-run clean

# Default target
help:
	@echo "Employment Act Malaysia Compliance Agent - Hour 2 Targets"
	@echo ""
	@echo "Setup targets:"
	@echo "  setup          Create conda environment and install dependencies"
	@echo "  setup-pip      Alternative pip-based setup"
	@echo ""
	@echo "Testing targets:"
	@echo "  smoke          Run quick smoke tests without heavy dependencies"
	@echo "  test-hour2     Run comprehensive Hour 2 tests"
	@echo "  test-retrieval Run retrieval smoke test specifically"
	@echo ""
	@echo "Docker targets:"
	@echo "  docker-build   Build Docker image with optimized defaults"
	@echo "  docker-spec    Build Docker image with Hour 2 spec parameters"
	@echo "  docker-run     Run RAG pipeline in container"
	@echo ""
	@echo "Demo targets:"
	@echo "  demo-retrieval Demo hybrid retrieval with query"
	@echo "  demo-spec      Demo with Hour 2 spec parameters"
	@echo ""
	@echo "Utility targets:"
	@echo "  clean          Clean up temporary files"

# Setup targets
setup:
	@echo "Setting up conda environment..."
	conda env create -f environment.yml
	@echo "Environment created. Activate with: conda activate faiss-env"

setup-pip:
	@echo "Setting up with pip..."
	pip install -r requirements.txt
	pip install "numpy==1.26.4" --force-reinstall
	@echo "Setup complete. Note: conda environment recommended for FAISS."

# Testing targets
smoke:
	@echo "Running smoke tests (no heavy dependencies)..."
	python tests/test_retrieval_smoke.py
	@echo "Basic smoke tests completed."

test-hour2:
	@echo "Running comprehensive Hour 2 tests..."
	python tests/test_hour2_comprehensive.py
	@echo "Hour 2 comprehensive tests completed."

test-retrieval:
	@echo "Running retrieval-specific smoke test..."
	python tests/test_retrieval_smoke.py
	@echo "Retrieval smoke test completed."

# Docker targets
docker-build:
	@echo "Building Docker image with optimized defaults..."
	docker build -t ea-agent:optimized .
	@echo "Docker image built: ea-agent:optimized"

docker-spec:
	@echo "Building Docker image with Hour 2 spec parameters..."
	docker build \
		--build-arg BM25_TOPK=100 \
		--build-arg DENSE_TOPK=50 \
		--build-arg CE_MAX_PAIRS=150 \
		--build-arg CE_BATCH_SIZE=32 \
		-t ea-agent:hour2-spec .
	@echo "Docker image built: ea-agent:hour2-spec"

docker-run:
	@echo "Running RAG pipeline in Docker container..."
	@echo "Note: Requires data/indices/ to be populated"
	docker run -v $(PWD)/data:/app/data ea-agent:optimized \
		python -m src.generation.rag_pipeline \
		--faiss data/indices/faiss.index \
		--store data/indices/store.pkl \
		--query "How many days of annual leave am I entitled to?"

# Demo targets
demo-retrieval:
	@echo "Demo: Hybrid retrieval with optimized parameters"
	@echo "Query: 'annual leave entitlement'"
	@if [ -f data/indices/faiss.index ] && [ -f data/indices/store.pkl ]; then \
		python -m src.retriever.hybrid_retriever \
			--faiss data/indices/faiss.index \
			--store data/indices/store.pkl \
			--query "annual leave entitlement"; \
	else \
		echo "Error: Missing indices. Run ingestion pipeline first."; \
		echo "  python src/ingest/build_index.py --in data/processed/chunks.jsonl --faiss data/indices/faiss.index --store data/indices/store.pkl"; \
	fi

demo-spec:
	@echo "Demo: Hour 2 spec parameters (BM25=100, Dense=50, CE=150)"
	@if [ -f data/indices/faiss.index ] && [ -f data/indices/store.pkl ]; then \
		BM25_TOPK=100 DENSE_TOPK=50 CE_MAX_PAIRS=150 \
		python -m src.retriever.hybrid_retriever \
			--faiss data/indices/faiss.index \
			--store data/indices/store.pkl \
			--query "annual leave entitlement"; \
	else \
		echo "Error: Missing indices. Run ingestion pipeline first."; \
	fi

# Environment override examples
demo-env-override:
	@echo "Demo: Environment variable overrides"
	@echo "Setting BM25_TOPK=100, DENSE_TOPK=50, CE_MAX_PAIRS=150..."
	BM25_TOPK=100 DENSE_TOPK=50 CE_MAX_PAIRS=150 CE_BATCH_SIZE=32 \
	python -c "import os; from src.retriever.hybrid_retriever import HybridRetriever; print('Config test passed')" 2>/dev/null || echo "Requires conda environment and indices"

# Utility targets
clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	find . -type f -name ".pytest_cache" -delete
	@echo "Cleanup completed."

# Quick validation
validate-hour2:
	@echo "Validating Hour 2 implementation..."
	@echo "1. Checking environment configuration..."
	@grep -q "BM25_TOPK" .env.example && echo "  ✓ BM25_TOPK configured" || echo "  ❌ Missing BM25_TOPK"
	@grep -q "DENSE_TOPK" .env.example && echo "  ✓ DENSE_TOPK configured" || echo "  ❌ Missing DENSE_TOPK"
	@grep -q "CE_MAX_PAIRS" .env.example && echo "  ✓ CE_MAX_PAIRS configured" || echo "  ❌ Missing CE_MAX_PAIRS"
	@echo "2. Checking Docker configuration..."
	@grep -q "ARG BM25_TOPK" Dockerfile && echo "  ✓ Docker build args present" || echo "  ❌ Missing Docker build args"
	@echo "3. Checking test files..."
	@[ -f tests/test_hour2_comprehensive.py ] && echo "  ✓ Comprehensive tests present" || echo "  ❌ Missing comprehensive tests"
	@[ -f tests/test_retrieval_smoke.py ] && echo "  ✓ Smoke tests present" || echo "  ❌ Missing smoke tests"
	@echo "4. Checking prompt templates..."
	@grep -q "EXAMPLE RESPONSE" src/generation/prompt_templates.py && echo "  ✓ Few-shot example present" || echo "  ❌ Missing few-shot example"
	@echo "Hour 2 validation completed."

# Help for environment setup
env-help:
	@echo "Environment Setup Guide:"
	@echo ""
	@echo "Option 1: Conda (Recommended)"
	@echo "  conda env create -f environment.yml"
	@echo "  conda activate faiss-env"
	@echo ""
	@echo "Option 2: pip"
	@echo "  pip install -r requirements.txt"
	@echo "  pip install 'numpy==1.26.4' --force-reinstall"
	@echo ""
	@echo "Hour 2 Spec Parameters:"
	@echo "  export BM25_TOPK=100"
	@echo "  export DENSE_TOPK=50" 
	@echo "  export CE_MAX_PAIRS=150"
	@echo "  export CE_BATCH_SIZE=32"
	@echo ""
	@echo "Performance Optimized (default):"
	@echo "  BM25_TOPK=30, DENSE_TOPK=20, CE_MAX_PAIRS=40"