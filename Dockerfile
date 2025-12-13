# CPU-only Dockerfile for reproducible experiments
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies; add any RL native deps here (e.g., libgl1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["bash"]
