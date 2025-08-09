# ===== Stage 1: Build dependencies =====
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir .[deployment]

# ===== Stage 2: Runtime =====
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy only necessary app files
COPY . .

EXPOSE 5000
CMD ["python", "application.py"]
