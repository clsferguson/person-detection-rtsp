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
- ultralytics: https://pypi.org/project/ultralytics/ (accessed Oct 2024)
- opencv-python: https://pypi.org/project/opencv-python/ (accessed Oct 2024)
- flask: https://pypi.org/project/Flask/ (accessed Oct 2024)

## Testing

Run tests: `pytest`
