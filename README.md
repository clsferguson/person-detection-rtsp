# Person Detection RTSP

Real-time person detection for RTSP streams using YOLO11, optimized for NVIDIA GPUs with an optional TensorRT inference path. The application scores the closest detected person inside a defined polygon from 0 (polygon boundary) to 1000 (target point) and exposes the results through an interactive web dashboard.

## Features

- 🚀 **YOLO11 detection pipeline** restricted to the person class with automatic TensorRT acceleration when enabled.
- 🧮 **Geometry-aware proximity scoring** that auto-computes the distance scale from the polygon and keeps the reference point inside the zone.
- 📈 **Live proximity card** powered by Server-Sent Events (SSE) that updates while the stream runs.
- 🛡️ **RTSP pre-flight connectivity checks** to surface offline status without freezing the UI.
- 🎯 **Interactive detection zone editor** directly on the main page for polygon and target point adjustments.
- 🧵 **Multithreaded inference worker** so health and metrics endpoints remain responsive during streaming.

## Getting Started

1. Clone this repository.
2. Ensure Docker is installed on your host. Install the NVIDIA Container Toolkit if you plan to leverage GPU acceleration.
3. Start the stack with the repository's Compose file:
   ```bash
   docker-compose up -d
   ```
4. Browse to <http://localhost:5000> to access the dashboard.

To stop the services run `docker-compose down`.

## Configuration & Environment

### Environment Variables

| Variable     | Default | Description |
|--------------|---------|-------------|
| `ENABLE_TRT` | `1`     | When set to `1`, the entrypoint exports the bundled YOLO model to ONNX and TensorRT if the CUDA/TensorRT toolchain is available, then loads the engine for inference. Set to `0` to force PyTorch inference. |

Configure additional runtime options through `config/config.json` or the in-app editor. The distributed `config/config.json.example` ships with a blank RTSP URL, a centered rectangular polygon, and a target point guaranteed to fall inside the zone. The application validates inputs, auto-adjusts the target point if necessary, and recalculates proximity scaling on every update.

## Using the Web UI

1. **Live Feed** – The left panel renders the MJPEG stream. If the RTSP source is unreachable an offline status frame is shown.
2. **Edit Mode** – Click **Enable Edit Mode** to overlay polygon controls.
   - Drag polygon vertices (red handles) to reshape the detection zone.
   - Right-click edges to add vertices or drag the green marker to reposition the target point.
   - Click **Reset Geometry** to revert to the default rectangle.
3. **Proximity Card** – The right panel displays the live proximity score, raw distance, last update time, and derived auto max distance.
4. **Configuration Form** – Update the RTSP URL from the form; geometry fields update automatically from the editor. Saving persists the configuration to disk.

### API Endpoints

| Endpoint       | Method | Description                                 |
|----------------|--------|---------------------------------------------|
| `/video_feed`  | GET    | MJPEG stream with bounding boxes and score. |
| `/config`      | GET    | Returns current configuration as JSON.      |
| `/config`      | POST   | Accepts JSON payload to update config.      |
| `/metrics`     | GET    | Latest proximity metrics snapshot.          |
| `/events/metrics` | GET | Server-Sent Events stream for live metrics. |
| `/health`      | GET    | Lightweight status check (JSON).            |
| `/status`      | GET    | Extended status snapshot (JSON).            |

## GPU Acceleration & TensorRT

- When `ENABLE_TRT=1` the entrypoint script:
  1. Downloads `yolo11n.pt` if it is not present.
  2. Exports the model to ONNX using Ultralytics.
  3. Builds a TensorRT engine from the ONNX artifact and caches it inside the container volume.
  4. Loads the TensorRT engine for inference at runtime, falling back to PyTorch if conversion fails.
- TensorRT execution requires CUDA, TensorRT, and compatible NVIDIA drivers on the host. The provided Docker image (`ghcr.io/clsferguson/person-detection-rtsp:latest`) expects the NVIDIA runtime to be available.
- Inference is locked to 5 FPS; extra frames are skipped to balance throughput and responsiveness.

## Make Targets

| Target | Description |
| ------ | ----------- |
| `make format` | Run formatting tools (host/CI only). |
| `make lint` | Run static linters (host/CI only). |
| `make test` | Execute unit tests (host/CI only). |
| `make verify-static` | Run non-invasive static checks required before commit. |

## Testing

Run tests on your host machine:

```bash
make test
```

## References

- numpy: <https://pypi.org/project/numpy/> (accessed 2025-10-11)
- ultralytics: <https://pypi.org/project/ultralytics/> (accessed 2025-10-11)
- opencv-python: <https://pypi.org/project/opencv-python/> (accessed 2025-10-11)
- flask: <https://pypi.org/project/Flask/> (accessed 2025-10-11)
- Ultralytics TensorRT integration: <https://docs.ultralytics.com/integrations/tensorrt/> (accessed 2025-10-11)
- Ultralytics export mode: <https://docs.ultralytics.com/modes/export/> (accessed 2025-10-11)
- Ultralytics ONNX integration: <https://docs.ultralytics.com/integrations/onnx/> (accessed 2025-10-11)
- Issue report: internal notes (accessed 2025-10-11)
