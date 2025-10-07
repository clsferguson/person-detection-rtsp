#!/bin/bash
set -e

echo "Installing OpenCV dependencies..."
apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0t64 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1

# Create config.json from example if it doesn't exist
if [ ! -f /app/config/config.json ]; then
    echo "Creating config.json from example..."
    cp /app/config/config.json.example /app/config/config.json
fi

echo "Downloading YOLO model..."
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

echo "Starting application..."
exec "$@"
