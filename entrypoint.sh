#!/bin/bash
set -e

echo "Installing OpenCV dependencies..."
apt-get update && apt-get install -y libglx-mesa0 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

echo "Downloading YOLO model..."
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

echo "Starting application..."
exec "$@"
