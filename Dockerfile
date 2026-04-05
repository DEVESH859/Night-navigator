FROM python:3.11-slim

# Install system dependencies required for data libraries
RUN apt-get update && apt-get install -y wget git gcc g++ libgdal-dev && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
# Hugging Face Spaces mandates running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
	PYTHONUNBUFFERED=1 \
	PORT=8000

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gdown

# Create data directory
RUN mkdir -p data

# IMPORTANT HACKATHON OPTIMIZATION: 
# We download the large Google Drive files during the DOCKER BUILD instead of App Startup.
# This ensures that when the Hugging Face Space wakes up from sleep, 
# it launches INSTANTLY because the 1.5 GB of data is permanently baked into the image!
RUN gdown --id 125LFc55i-2wGkVYiTaUwz0W0ozeW_qw2 -O data/bangalore_graph.graphml && \
    gdown --id 1bNaIBxH6MzvwywR9P2rHsgoXya8FW3yM -O data/edges_with_safety.geojson && \
    gdown --id 1pg3zE5WXbv1GD3RFs78J7mcmXoy9-3gl -O data/pois.geojson

# Copy the rest of the application
COPY --chown=user . .

# Railway injects PORT at runtime; keep a sensible local default.
EXPOSE 8000

# Start the FastAPI backend
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
