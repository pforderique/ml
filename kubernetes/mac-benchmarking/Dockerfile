FROM python:3.10-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy benchmark script
WORKDIR /app
COPY benchmark.py .

# Run benchmark on container start
CMD ["python", "benchmark.py"]
