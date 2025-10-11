"""Core application logic for the Person Detection RTSP service."""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from datetime import datetime
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from gevent.pywsgi import WSGIServer
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../templates')
model = None
config: Dict[str, object] = {
    'rtsp_url': '',
    'polygon': [(0, 0), (640, 0), (640, 480), (0, 480)],
    'point': (320, 240),
    'max_dist': 200,
}


def normalize_polygon(points: Iterable[Iterable[int]]) -> List[Tuple[int, int]]:
    """Ensure polygon points are stored as a list of integer tuples."""
    normalized: List[Tuple[int, int]] = []
    for pair in points:
        if isinstance(pair, (tuple, list)) and len(pair) == 2:
            normalized.append((int(pair[0]), int(pair[1])))
        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Invalid polygon point: {pair}")
    return normalized


def normalize_point(point: Iterable[int]) -> Tuple[int, int]:
    """Ensure the reference point is stored as an integer tuple."""
    if isinstance(point, (tuple, list)) and len(point) == 2:
        return int(point[0]), int(point[1])
    raise ValueError(f"Invalid point value: {point}")  # pragma: no cover - defensive guard


def update_config(new_config: Dict[str, object]) -> None:
    """Merge new configuration values and normalize collections."""
    global config
    config.update(new_config)
    if 'polygon' in config and config['polygon']:
        config['polygon'] = normalize_polygon(config['polygon'])  # type: ignore[arg-type]
    if 'point' in config:
        config['point'] = normalize_point(config['point'])  # type: ignore[arg-type]


def save_config(config_path: str = 'config/config.json') -> None:
    """Persist the current configuration to disk."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    payload = {
        'rtsp_url': config['rtsp_url'],
        'polygon': [list(pt) for pt in config['polygon']],
        'point': list(config['point']),
        'max_dist': config['max_dist'],
    }
    with open(config_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)


def load_model() -> bool:
    """Load the YOLO model with error handling."""
    global model
    try:
        logger.info("Loading YOLO model...")
        model = YOLO('yolo11n.pt')
        logger.info("✓ YOLO model loaded successfully")
        return True
    except Exception as exc:  # pragma: no cover - logging path
        logger.error(f"✗ Failed to load YOLO model: {exc}")
        return False


def load_config(config_path: str = 'config/config.json') -> None:
    """Load configuration from file and normalize values."""
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding='utf-8') as file:
                loaded = json.load(file)
            update_config(loaded)
            logger.info(f"✓ Configuration loaded from {config_path}")
            logger.info(f"  RTSP URL: {config['rtsp_url'] or '(not configured)'}")
        except Exception as exc:  # pragma: no cover - logging path
            logger.error(f"✗ Failed to load config: {exc}")
    else:
        logger.warning(f"⚠ Config file not found: {config_path}")


def encode_frame(frame: np.ndarray) -> bytes:
    """Encode an image frame as JPEG bytes."""
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        raise ValueError('Failed to encode frame to JPEG')
    return buffer.tobytes()


def format_mjpeg_frame(frame_bytes: bytes) -> bytes:
    """Wrap JPEG bytes for MJPEG streaming."""
    return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'


def build_message_frame(
    lines: List[str],
    background_color: Tuple[int, int, int] = (35, 35, 35),
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Create a simple status frame with centered text lines."""
    frame = np.full((480, 640, 3), background_color, dtype=np.uint8)
    base_y = 240 - ((len(lines) - 1) * 25)
    for idx, line in enumerate(lines):
        text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        text_x = max((640 - text_size[0]) // 2, 20)
        text_y = base_y + idx * 45
        cv2.putText(
            frame,
            line,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            text_color,
            2,
            cv2.LINE_AA,
        )
    return frame


def can_connect_rtsp(rtsp_url: str, timeout: float = 3.0) -> bool:
    """Perform a lightweight TCP connectivity check for an RTSP endpoint."""
    parsed = urlparse(rtsp_url)
    if parsed.scheme not in {'rtsp', 'rtsps'}:
        logger.warning(f"Unsupported RTSP scheme: {parsed.scheme}")
        return False
    if not parsed.hostname:
        logger.warning("RTSP URL missing hostname")
        return False

    port = parsed.port or (322 if parsed.scheme == 'rtsps' else 554)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError as exc:
        logger.warning(
            "RTSP connectivity check failed for %s:%s (%s)",
            parsed.hostname,
            port,
            exc,
        )
        return False


def gen_frames():
    """Generate MJPEG frames from the configured RTSP stream."""
    while True:
        rtsp_url = str(config.get('rtsp_url', '') or '').strip()

        if not rtsp_url:
            frame_bytes = encode_frame(
                build_message_frame(
                    [
                        'No RTSP stream configured',
                        'Visit the configuration page to add a stream.',
                    ],
                    background_color=(20, 20, 20),
                )
            )
            yield format_mjpeg_frame(frame_bytes)
            time.sleep(1.5)
            continue

        if not can_connect_rtsp(rtsp_url):
            frame_bytes = encode_frame(
                build_message_frame(
                    [
                        'Stream offline or unreachable',
                        'Retrying connection...',
                    ],
                    background_color=(35, 35, 70),
                    text_color=(255, 220, 220),
                )
            )
            yield format_mjpeg_frame(frame_bytes)
            time.sleep(2.0)
            continue

        logger.info("Opening RTSP stream: %s", rtsp_url)
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            logger.error("✗ Failed to open RTSP stream")
            cap.release()
            frame_bytes = encode_frame(
                build_message_frame(
                    [
                        'Unable to open stream',
                        'Check credentials and camera status.',
                    ],
                    background_color=(60, 0, 0),
                    text_color=(220, 220, 255),
                )
            )
            yield format_mjpeg_frame(frame_bytes)
            time.sleep(2.0)
            continue

        logger.info("✓ RTSP stream opened successfully")
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Stream ended or failed to read frame")
                    break

                frame_count += 1

                polygon_points: List[Tuple[int, int]] = config['polygon']  # type: ignore[assignment]
                target_point: Tuple[int, int] = config['point']  # type: ignore[assignment]
                max_dist = max(int(config['max_dist']), 1)

                closest_val = 0
                best_box = None

                if polygon_points:
                    poly = np.array(polygon_points, np.int32)

                    results = model(frame, classes=[0], verbose=False)  # type: ignore[misc]
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                                dist = np.linalg.norm(np.array(target_point) - np.array([cx, cy]))
                                val = max(0, 1000 * (1 - dist / max_dist))
                                if val > closest_val:
                                    closest_val = val
                                    best_box = (x1, y1, x2, y2)

                    poly_draw = poly.reshape((-1, 1, 2))
                    cv2.polylines(frame, [poly_draw], True, (255, 255, 0), 2)

                cv2.circle(frame, target_point, 10, (0, 255, 255), -1)

                if best_box:
                    cv2.rectangle(
                        frame,
                        (int(best_box[0]), int(best_box[1])),
                        (int(best_box[2]), int(best_box[3])),
                        (0, 255, 0),
                        3,
                    )

                cv2.putText(
                    frame,
                    f"Proximity: {int(closest_val)}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0) if closest_val > 0 else (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Frame: {frame_count}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                frame_bytes = encode_frame(frame)
                yield format_mjpeg_frame(frame_bytes)

        except Exception as exc:  # pragma: no cover - logging path
            logger.error(f"Error in frame generation: {exc}")
        finally:
            cap.release()
            logger.info("Stream closed after %s frames", frame_count)

        time.sleep(0.5)


@app.route('/')
def index():
    """Main page with video stream."""
    return render_template('index.html', config=config)


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/config', methods=['GET', 'POST'])
def config_page():
    """Configuration page for RTSP and detection settings."""
    global config
    message = None

    if request.method == 'POST':
        try:
            rtsp_url = request.form.get('rtsp', '').strip()

            polygon_input = request.form.get('polygon', '').strip()
            polygon_pairs: List[Tuple[int, int]] = []
            if polygon_input:
                for token in polygon_input.split():
                    cleaned = token.strip().strip('()')
                    if not cleaned:
                        continue
                    parts = [part.strip() for part in cleaned.split(',')]
                    if len(parts) != 2:
                        raise ValueError(f"Invalid polygon point: {token}")
                    polygon_pairs.append((int(parts[0]), int(parts[1])))

            point_input = request.form.get('point', '').strip()
            point_tuple = None
            if point_input:
                cleaned_point = point_input.strip('()')
                parts = [part.strip() for part in cleaned_point.split(',') if part.strip()]
                if len(parts) == 2:
                    point_tuple = (int(parts[0]), int(parts[1]))
                else:
                    raise ValueError(f"Invalid point value: {point_input}")

            max_dist = int(request.form.get('max_dist', config['max_dist']))

            updates: Dict[str, object] = {
                'rtsp_url': rtsp_url,
                'max_dist': max_dist,
            }
            if polygon_pairs:
                updates['polygon'] = polygon_pairs
            if point_tuple is not None:
                updates['point'] = point_tuple

            update_config(updates)
            save_config()

            message = "✓ Configuration saved successfully!"
            logger.info("Configuration updated and saved")

        except Exception as exc:
            message = f"✗ Error saving configuration: {exc}"
            logger.error(f"Failed to save config: {exc}")

    polygon_display = ' '.join(f'({x},{y})' for x, y in config['polygon'])
    point_display = f"({config['point'][0]},{config['point'][1]})"

    return render_template(
        'config.html',
        config=config,
        message=message,
        polygon_display=polygon_display,
        point_display=point_display,
    )


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify(
        {
            'status': 'healthy',
            'model_loaded': model is not None,
            'rtsp_configured': bool(config['rtsp_url']),
            'timestamp': datetime.now().isoformat(),
        }
    )


@app.route('/status')
def status():
    """Status information endpoint."""
    return jsonify(
        {
            'config': config,
            'model': 'yolo11n.pt',
            'model_loaded': model is not None,
        }
    )


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Person Detection RTSP Application Starting")
    logger.info("=" * 60)

    if not load_model():
        logger.error("Failed to initialize. Exiting.")
        sys.exit(1)

    load_config()

    host = '0.0.0.0'
    port = 5000
    logger.info(f"🌐 Starting web server on http://{host}:{port}")
    logger.info(f"📺 Access the stream at: http://{host}:{port}/")
    logger.info(f"⚙️  Configuration page: http://{host}:{port}/config")
    logger.info(f"❤️  Health check: http://{host}:{port}/health")
    logger.info("=" * 60)

    try:
        http_server = WSGIServer((host, port), app, log=logger)
        http_server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as exc:
        logger.error(f"Server error: {exc}")
        sys.exit(1)
