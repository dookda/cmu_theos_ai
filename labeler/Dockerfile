FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 wget git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MobileSAM
RUN pip install --no-cache-dir git+https://github.com/ChaoningZhang/MobileSAM.git

# Download MobileSAM checkpoint
RUN mkdir -p /app/checkpoints && \
    wget -q -O /app/checkpoints/mobile_sam.pt \
    https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
