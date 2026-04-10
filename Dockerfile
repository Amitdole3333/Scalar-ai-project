FROM python:3.11-slim

LABEL maintainer="AI Hackathon Team"
LABEL description="Hospital ER Triage – OpenEnv AI Training Environment"

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# Copy requirements first for layer caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user . .

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Mandatory env vars (Hackathon Guidelines §3)
ENV HF_TOKEN=""
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

USER user

# Default command: web server for HF Spaces (port 7860)
# For CLI inference, run: python inference.py
CMD ["python", "server/app.py"]
