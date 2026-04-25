FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies (no torch — inference uses OpenAI API)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir openai pydantic

# Copy project
COPY . .

# Environment variables (override at runtime)
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o-mini
ENV HF_TOKEN=""

# Default: run the Gradio Space app
EXPOSE 7860
CMD ["python", "app.py"]

# To run inference instead:
# docker run -e API_BASE_URL=... -e MODEL_NAME=... -e HF_TOKEN=... \
#   qstoreprice python inference.py --scenario STABLE_WEEK
