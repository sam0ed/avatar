# Avatar server — slim image with Python 3.11 + uv
# GPU libraries (PyTorch, etc.) come via pre-compiled wheels, no CUDA toolkit needed.
# Later stages add: vLLM, Fish Speech, MuseTalk as pip/uv dependencies.

FROM python:3.11-slim-bookworm

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy server project files
COPY server/pyproject.toml server/uv.lock* ./

# Install dependencies (cached layer)
RUN uv sync --no-dev

# Copy server source code
COPY server/src/ ./src/

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["uv", "run", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
