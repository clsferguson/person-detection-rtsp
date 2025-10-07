#!/bin/bash

python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

exec python3 app.py
