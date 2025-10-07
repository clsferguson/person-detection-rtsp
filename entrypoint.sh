#!/bin/bash

set -e

echo "Downloading YOLO model..."

python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

echo "Model downloaded."

exec "$@"
