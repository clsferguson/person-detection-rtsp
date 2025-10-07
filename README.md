# Person Detection RTSP

Real-time multi-person detection on RTSP streams using YOLO11, with proximity value calculation and web UI.

## Setup

1. Clone the repo.
2. Ensure Docker and NVIDIA Docker are installed.
3. Run: `docker-compose up --build`

## Usage

- Access web UI at http://localhost:5000
- Configure RTSP URL, detection area (polygon as list of (x,y)), target point, and max distance in the config page.
- View the annotated stream on the main page.

## Dependencies

Pinned versions in requirements.txt. Update via pip-tools if needed.

References:
- numpy: https://pypi.org/project/numpy/ (accessed Oct 2025)
- ultralytics: https://pypi.org/project/ultralytics/ (accessed Oct 2025)
- opencv-python: https://pypi.org/project/opencv-python/ (accessed Oct 2025)
- flask: https://pypi.org/project/Flask/ (accessed Oct 2025)

## Testing

Run tests: `pytest`

## Running from Docker Image

To run the pre-built image from GitHub Container Registry:

```bash
docker pull ghcr.io/clsferguson/person-detection-rtsp:latest
docker run -p 5000:5000 ghcr.io/clsferguson/person-detection-rtsp:latest
```

Access the web UI at http://localhost:5000
