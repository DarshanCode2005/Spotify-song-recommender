# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies with specific commands for transformers
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers && \
    pip install --no-cache-dir -r requirements.txt

# Download and cache the model before running the app
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)"

# Copy the rest of the application
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

# Create a healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 