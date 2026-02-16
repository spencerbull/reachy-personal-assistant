# Reachy Personal Assistant - Bot Service
# Build from project root: docker build -t reachy-bot .

# ============================================
# Stage 1: Build dependencies
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files from bot directory
COPY bot/pyproject.toml bot/
COPY bot/uv.lock* bot/

# Create venv and install dependencies
WORKDIR /app/bot
RUN uv venv /app/.venv
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="/app/.venv/bin:$PATH"
RUN uv sync --frozen --no-dev || uv sync --no-dev

# ============================================
# Stage 2: Runtime image
# ============================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    xvfb \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Copy application code
COPY bot/ /app/bot/
COPY agent/ /app/agent/
COPY REACHY_SOUL.md /app/

# Set Python path to find agent module
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Default environment variables (override in docker-compose)
ENV LLM_BACKEND=langgraph
ENV MAIN_MODEL_URL=http://agent-llm:8000/v1
ENV ROUTER_MODEL_URL=http://routing-llm:8000/v1
ENV REACHY_DAEMON_HOST=host.docker.internal
ENV SOUL_FILE_PATH=/app/REACHY_SOUL.md

# Expose WebRTC signaling port
EXPOSE 7860

# Start with Xvfb for headless MuJoCo rendering
CMD ["sh", "-c", "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & export DISPLAY=:99 && sleep 2 && python /app/bot/main.py --host 0.0.0.0"]
