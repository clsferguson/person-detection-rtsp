#!/bin/bash
set -euo pipefail

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

log "Installing OpenCV dependencies..."
apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0t64 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    jq

CONFIG_DIR="/app/config"
CONFIG_FILE="${CONFIG_DIR}/config.json"
EXAMPLE_FILE="${CONFIG_DIR}/config.json.example"

mkdir -p "${CONFIG_DIR}"

if [ ! -f "${CONFIG_FILE}" ]; then
    if [ -f "${EXAMPLE_FILE}" ]; then
        log "Creating config.json from example..."
        cp "${EXAMPLE_FILE}" "${CONFIG_FILE}"
    else
        log "Config example missing; creating default configuration..."
        cat <<'JSON' > "${CONFIG_FILE}"
{
  "rtsp_url": "",
  "polygon": [[100, 100], [540, 100], [540, 380], [100, 380]],
  "point": [320, 240]
}
JSON
    fi
fi

mkdir -p /app/ultralytics && export YOLO_CONFIG_DIR=/app/ultralytics

WEIGHTS_PATH=${YOLO_WEIGHTS:-yolo11n.pt}
ENGINE_PATH=${YOLO_ENGINE:-}
ENABLE_TRT=${ENABLE_TRT:-0}
IMG_SIZE=${YOLO_IMAGE_SIZE:-640}
BATCH_SIZE=${YOLO_TRT_BATCH:-1}
DEVICE_ARG=${YOLO_EXPORT_DEVICE:-0}

log "Downloading YOLO weights (${WEIGHTS_PATH})..."
python3 - "${WEIGHTS_PATH}" <<'PY'
import sys
from pathlib import Path
from ultralytics import YOLO

weights = Path(sys.argv[1])
if weights.suffix == '.pt' and not weights.exists():
    YOLO(str(weights))
elif not weights.exists():
    YOLO(weights.name)
PY

should_export_engine() {
    [ "${ENABLE_TRT}" = "1" ] || return 1
    local weights_ext="${WEIGHTS_PATH##*.}"
    if [ -n "${ENGINE_PATH}" ]; then
        [ -f "${ENGINE_PATH}" ] && return 1
        echo "${ENGINE_PATH}"
        return 0
    fi
    if [ "${weights_ext}" = "pt" ]; then
        local candidate="${WEIGHTS_PATH%.pt}.engine"
        [ -f "${candidate}" ] && return 1
        echo "${candidate}"
        return 0
    fi
    return 1
}

if engine_target=$(should_export_engine); then
    log "TensorRT export requested; generating engine at ${engine_target}"
    python3 - <<PY
from ultralytics import YOLO
import os

weights = os.environ.get("WEIGHTS_PATH", "yolo11n.pt")
engine_target = os.environ.get("ENGINE_TARGET")
imgsz = int(os.environ.get("IMG_SIZE", 640))
batch = int(os.environ.get("BATCH_SIZE", 1))
device = os.environ.get("DEVICE_ARG", "0")

model = YOLO(weights)
model.export(format="onnx", imgsz=imgsz, device=device, simplify=True, dynamic=False)
onnx_path = weights.replace('.pt', '.onnx') if weights.endswith('.pt') else weights + '.onnx'
model = YOLO(onnx_path)
model.export(format="engine", imgsz=imgsz, device=device, batch=batch, simplify=True)

if engine_target and engine_target != "yolo11n.engine":
    import shutil
    if os.path.exists("yolo11n.engine") and engine_target != "yolo11n.engine":
        shutil.move("yolo11n.engine", engine_target)
PY
else
    log "TensorRT export not requested or engine already present."
fi

log "Starting application..."
exec "$@"
