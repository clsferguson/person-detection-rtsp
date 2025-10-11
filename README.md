# Person Detection RTSP

Real-time multi-person detection on RTSP streams using YOLO11, with proximity value calculation and web UI.

## Setup

1. Clone the repository.
2. Ensure Docker (and NVIDIA Docker for GPU acceleration) are available on your host.
3. Build and start the stack:
   ```bash
   docker-compose up --build
   ```

## Usage

- Access the web UI at http://localhost:5000.
- The main page now performs a lightweight RTSP reachability test before trying to stream. If the camera is offline, a friendly placeholder is displayed instead of freezing the UI.
- Configure the RTSP URL, detection zone polygon, target point, and maximum distance on the `/config` page.
- The `/health` endpoint remains available for automation or monitoring even though the link was removed from the home page.

## Configuration Notes

- Polygon coordinates should be entered as space-separated `(x,y)` pairs, e.g. `(0,0) (640,0) (640,480) (0,480)`.
- The target point field accepts a single `(x,y)` pair.
- Invalid polygon or point entries will surface an error message instead of crashing the template.

## Make Targets

| Target | Description |
| ------ | ----------- |
| `make format` | Run formatting tools (host/CI only). |
| `make lint` | Run static linters (host/CI only). |
| `make test` | Execute unit tests (host/CI only). |
| `make verify-static` | Run non-invasive static checks required before commit. |

> ⚠️ **Note:** Do not run build/test targets inside the SSH sandbox environment; execute them locally or in CI.

## Testing

Run tests on your host machine:

```bash
make test
```

## References

- numpy: https://pypi.org/project/numpy/ (accessed 2025-10-11)
- ultralytics: https://pypi.org/project/ultralytics/ (accessed 2025-10-11)
- opencv-python: https://pypi.org/project/opencv-python/ (accessed 2025-10-11)
- flask: https://pypi.org/project/Flask/ (accessed 2025-10-11)
- Issue report: internal notes (accessed 2025-10-11)
