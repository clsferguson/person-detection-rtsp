#!/bin/bash

apt-get update && apt-get install -y libglx-mesa0 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1


set -e

echo "Downloading YOLO model..."

python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

echo "Model downloaded."

exec "$@"
