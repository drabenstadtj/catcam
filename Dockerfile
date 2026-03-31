FROM python:3.11-slim

# System deps: FFmpeg + OpenCV native libs
RUN sed -i 's/^Components: main$/Components: main non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    intel-media-va-driver \
    libva-drm2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY templates/ templates/
COPY cat_classifier.pt .
COPY train_classifier.py .
COPY ["cat images", "./cat images/"]

# Pre-download the YOLO model so first-run startup is fast.
# Remove this RUN line if you prefer to download at runtime.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

RUN mkdir -p /data

EXPOSE 5000/udp
EXPOSE 8080/tcp

CMD ["python", "app.py"]
