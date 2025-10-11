# Person Detection RTSP

Real-time multi-person detection for RTSP streams using YOLO11. The application scores the closest detected person inside a defined polygon from 0 (polygon boundary) to 1000 (target point) and exposes the results through an interactive web dashboard.

## Features

- 🚀 **YOLO11 detection pipeline** tuned for person-class inference.
- 🛡️ **RTSP pre-flight connectivity checks** prevent UI stalls when the camera is offline and display an informative placeholder instead.
- 🎯 **Interactive detection zone editor** directly on the main page:
  - Toggle edit mode to draw/reshape the polygonal target zone.
  - Right-click edges to add vertices, drag existing vertices to reposition them.
  - Drag the target point marker to redefine the priority location.
- 💾 **Config persistence** to `config/config.json`, editable both through the UI and a JSON API.
- 📈 **Proximity scoring** (0–1000) reported on the live stream overlay for the person closest to your target point while inside the polygon.

## Getting Started

1. Clone this repository.
2. Ensure Docker (and NVIDIA Docker if you intend to leverage GPUs) is installed on your host machine.
3. Build and start the stack:
   ```bash
   docker-compose up --build
   ```
4. Browse to <http://localhost:5000> to access the dashboard.

> **Tip:** The default configuration creates a simple rectangular detection zone. You can reshape it as soon as the UI loads—even if no RTSP stream is currently reachable.

## Using the Web UI

1. **Live Feed** – The left panel renders the MJPEG stream. When the stream is offline you will see a status frame instead of a frozen UI.
2. **Edit Mode** – Click **Enable Edit Mode** to overlay the polygon controls.
   - Drag polygon vertices (red handles) to reshape the detection zone.
   - Right-click along polygon edges to insert new vertices on the fly.
   - Drag the green marker to update the target proximity point.
   - Click **Reset Geometry** while in edit mode to start from a centered rectangle.
3. **Configuration Form** – The right panel mirrors the current RTSP URL, max distance, polygon, and target point.
   - Adjust the RTSP URL or maximum distance as needed.
   - Save your changes to persist them to disk and apply immediately.
4. **Persistence** – Settings are written to `config/config.json`. They will be reloaded automatically on the next startup.

### API Endpoints

| Endpoint       | Method | Description                                 |
|----------------|--------|---------------------------------------------|
| `/video_feed`  | GET    | MJPEG stream with bounding boxes and score. |
| `/config`      | GET    | Returns current configuration as JSON.      |
| `/config`      | POST   | Accepts JSON payload to update config.      |
| `/health`      | GET    | Lightweight status check (JSON).            |
| `/status`      | GET    | Extended status snapshot (JSON).            |

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
- Issue report: internal notes (accessed 2025-10-11)
