"""Core application logic for the Person Detection RTSP service."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import socket
import sys
import time
from datetime import datetime
from threading import Event, Lock, Thread
from typing import Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from gevent import sleep
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

MODEL_DEVICE = os.environ.get('YOLO_DEVICE')
PERSON_CLASS_ID = 0
FRAME_INTERVAL_SECONDS = 0.2  # 5 FPS
PROXIMITY_SCALE = 1000.0

CONFIG_LOCK = Lock()
config: Dict[str, object] = {
    'rtsp_url': '',
    'polygon': [(0, 0), (640, 0), (640, 480), (0, 480)],
    'point': (320, 240),
    'frame_width': 640,
    'frame_height': 480,
}

model: Optional[YOLO] = None
_detection_worker: Optional['DetectionWorker'] = None


def normalize_polygon(points: Iterable[Iterable[Union[int, float]]]) -> List[Tuple[int, int]]:
    """Ensure polygon points are stored as a list of integer tuples."""
    normalized: List[Tuple[int, int]] = []
    for pair in points:
        if isinstance(pair, (tuple, list)) and len(pair) >= 2:
            normalized.append((int(float(pair[0])), int(float(pair[1]))))
        elif isinstance(pair, dict) and {'x', 'y'} <= pair.keys():
            normalized.append((int(float(pair['x'])), int(float(pair['y']))))
        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Invalid polygon point: {pair}")

    if len(normalized) < 3:
        raise ValueError('Polygon must contain at least three points')

    return normalized


def normalize_point(point: Union[Iterable[Union[int, float]], Dict[str, Union[int, float]]]) -> Tuple[int, int]:
    """Ensure the reference point is stored as an integer tuple."""
    if isinstance(point, dict) and {'x', 'y'} <= point.keys():
        return int(float(point['x'])), int(float(point['y']))
    if isinstance(point, (tuple, list)) and len(point) >= 2:
        return int(float(point[0])), int(float(point[1]))
    raise ValueError(f"Invalid point value: {point}")  # pragma: no cover - defensive guard


def polygon_centroid(points: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Calculate the centroid of a polygon using the shoelace formula."""
    if len(points) < 3:
        return (0, 0)

    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross

    area *= 0.5
    if abs(area) < 1e-6:
        avg_x = sum(p[0] for p in points) / len(points)
        avg_y = sum(p[1] for p in points) / len(points)
        return int(round(avg_x)), int(round(avg_y))

    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return int(round(cx)), int(round(cy))


def is_point_inside_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    if len(polygon) < 3:
        return False
    poly = np.array(polygon, np.int32)
    result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
    return result >= 0


def compute_auto_max_distance(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> float:
    if len(polygon) < 3:
        return 1.0
    distances = [math.dist(point, vertex) for vertex in polygon]
    max_dist = max(distances) if distances else 1.0
    return max(max_dist, 1.0)


def _recalculate_geometry_locked() -> List[str]:
    """Recalculate derived geometry fields while holding CONFIG_LOCK."""
    warnings: List[str] = []
    polygon: List[Tuple[int, int]] = config.get('polygon', [])  # type: ignore[assignment]
    point: Tuple[int, int] = config.get('point', (0, 0))  # type: ignore[assignment]

    if not polygon or len(polygon) < 3:
        warnings.append('Polygon must contain at least three points; using default rectangle.')
        config['polygon'] = [(0, 0), (640, 0), (640, 480), (0, 480)]
        polygon = config['polygon']  # type: ignore[assignment]

    if not is_point_inside_polygon(point, polygon):
        centroid = polygon_centroid(polygon)
        warnings.append('Target point adjusted to polygon centroid to remain inside zone.')
        config['point'] = centroid
        point = centroid

    config['auto_max_dist'] = compute_auto_max_distance(point, polygon)
    return warnings


def serialize_config() -> Dict[str, object]:
    """Return the current configuration in a JSON-safe structure."""
    with CONFIG_LOCK:
        snapshot = copy.deepcopy(config)
    return {
        'rtsp_url': snapshot.get('rtsp_url', ''),
        'polygon': [list(pt) for pt in snapshot.get('polygon', [])],
        'point': list(snapshot.get('point', (0, 0))),
        'frame': {
            'width': snapshot.get('frame_width', 640),
            'height': snapshot.get('frame_height', 480),
        },
        'auto_max_dist': snapshot.get('auto_max_dist', 1.0),
    }


def update_config(new_config: Dict[str, object]) -> List[str]:
    """Merge new configuration values and normalize collections."""
    warnings: List[str] = []
    with CONFIG_LOCK:
        if 'rtsp_url' in new_config:
            config['rtsp_url'] = str(new_config['rtsp_url']).strip()

        if 'polygon' in new_config and new_config['polygon']:
            config['polygon'] = normalize_polygon(new_config['polygon'])  # type: ignore[arg-type]

        if 'point' in new_config:
            config['point'] = normalize_point(new_config['point'])  # type: ignore[arg-type]

        warnings.extend(_recalculate_geometry_locked())

    return warnings


def save_config(config_path: str = 'config/config.json') -> None:
    """Persist the current configuration to disk."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with CONFIG_LOCK:
        payload = {
            'rtsp_url': config['rtsp_url'],
            'polygon': [list(pt) for pt in config['polygon']],
            'point': list(config['point']),
        }
    with open(config_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)


def load_config(config_path: str = 'config/config.json') -> None:
    """Load configuration from file and normalize values."""
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding='utf-8') as file:
                loaded = json.load(file)
            warnings = update_config(loaded)
            logger.info(f"✓ Configuration loaded from {config_path}")
            logger.info(f"  RTSP URL: {config.get('rtsp_url') or '(not configured)'}")
            for warning in warnings:
                logger.warning(f"  ⚠ {warning}")
        except Exception as exc:  # pragma: no cover - logging path
            logger.error(f"✗ Failed to load config: {exc}")
    else:
        logger.warning(f"⚠ Config file not found: {config_path}")


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


def update_frame_dimensions(width: int, height: int) -> None:
    with CONFIG_LOCK:
        config['frame_width'] = width
        config['frame_height'] = height


def get_auto_max_distance() -> float:
    with CONFIG_LOCK:
        return float(config.get('auto_max_dist', 1.0))


def resolve_model_paths() -> Tuple[str, Optional[str]]:
    weights = os.environ.get('YOLO_WEIGHTS', 'yolo11n.pt')
    engine_env = os.environ.get('YOLO_ENGINE')
    engine_candidate: Optional[str] = None

    if engine_env:
        engine_candidate = engine_env
    else:
        root, ext = os.path.splitext(weights)
        if ext.lower() == '.pt':
            candidate = root + '.engine'
            if os.path.exists(candidate):
                engine_candidate = candidate

    return weights, engine_candidate


def load_model() -> bool:
    """Load the YOLO model with optional TensorRT engine."""
    global model
    weights_path, engine_path = resolve_model_paths()
    prefer_trt = os.environ.get('ENABLE_TRT', '0') == '1'

    try:
        if prefer_trt and engine_path and os.path.exists(engine_path):
            logger.info("Loading YOLO TensorRT engine: %s", engine_path)
            model = YOLO(engine_path)
        else:
            if prefer_trt and engine_path and not os.path.exists(engine_path):
                logger.warning("TensorRT engine requested but not found: %s", engine_path)
            logger.info("Loading YOLO model: %s", weights_path)
            model = YOLO(weights_path)
            if MODEL_DEVICE:
                try:
                    model.to(MODEL_DEVICE)
                except Exception as exc:  # pragma: no cover - logging path
                    logger.warning(f"Could not move model to device {MODEL_DEVICE}: {exc}")
        logger.info("✓ YOLO model loaded successfully")
        return True
    except Exception as exc:  # pragma: no cover - logging path
        logger.error(f"✗ Failed to load YOLO model: {exc}")
        model = None
        return False


class DetectionWorker:
    """Background worker that handles frame retrieval, inference, and metric updates."""

    def __init__(self) -> None:
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self._frame_event = Event()
        self._frame_lock = Lock()
        self._metrics_lock = Lock()
        self._latest_frame: Optional[bytes] = None
        self._metrics: Dict[str, object] = self._empty_metrics('idle')
        self._last_rtsp: Optional[str] = None

    @staticmethod
    def _empty_metrics(stream_status: str) -> Dict[str, object]:
        return {
            'stream_status': stream_status,
            'last_frame_ts': None,
            'proximity': {
                'score': 0,
                'raw_distance': None,
                'max_distance': get_auto_max_distance(),
                'normalized': 0.0,
                'inside_polygon': False,
                'timestamp': None,
            },
            'detection': None,
            'rtsp_configured': False,
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name='DetectionWorker', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=2.0)

    def get_frame(self, timeout: float = 1.0) -> Optional[bytes]:
        if self._frame_event.wait(timeout):
            with self._frame_lock:
                frame = self._latest_frame
            self._frame_event.clear()
            return frame
        return None

    def get_metrics(self) -> Dict[str, object]:
        with self._metrics_lock:
            return copy.deepcopy(self._metrics)

    def current_status(self) -> str:
        with self._metrics_lock:
            return str(self._metrics.get('stream_status', 'idle'))

    def _set_metrics(self, metrics: Dict[str, object]) -> None:
        with self._metrics_lock:
            self._metrics = metrics

    def _set_frame(self, frame_bytes: bytes) -> None:
        with self._frame_lock:
            self._latest_frame = frame_bytes
        self._frame_event.set()

    def _update_stream_status(self, status: str, rtsp_configured: bool) -> None:
        metrics = self._empty_metrics(status)
        metrics['rtsp_configured'] = rtsp_configured
        self._set_metrics(metrics)

    def _run(self) -> None:
        global model
        last_frame_emit = 0.0
        while not self._stop_event.is_set():
            if model is None:
                self._update_stream_status('model_unloaded', False)
                time.sleep(1.0)
                continue

            with CONFIG_LOCK:
                current_rtsp = str(config.get('rtsp_url', '')).strip()
            if not current_rtsp:
                self._update_stream_status('unconfigured', False)
                time.sleep(1.0)
                continue

            if not can_connect_rtsp(current_rtsp):
                self._update_stream_status('unreachable', True)
                time.sleep(2.0)
                continue

            logger.info("Opening RTSP stream: %s", current_rtsp)
            cap = cv2.VideoCapture(current_rtsp)

            if not cap.isOpened():
                logger.error("✗ Failed to open RTSP stream")
                self._update_stream_status('open_failed', True)
                cap.release()
                time.sleep(2.0)
                continue

            logger.info("✓ RTSP stream opened successfully")
            self._update_stream_status('streaming', True)
            self._last_rtsp = current_rtsp

            try:
                while not self._stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Stream ended or failed to read frame")
                        self._update_stream_status('read_failed', True)
                        break

                    now = time.time()
                    if now - last_frame_emit < FRAME_INTERVAL_SECONDS:
                        continue
                    last_frame_emit = now

                    cfg_snapshot = serialize_config()
                    if cfg_snapshot['rtsp_url'] != current_rtsp:
                        logger.info("RTSP URL changed; restarting stream reader")
                        break

                    processed_frame, metrics = self._process_frame(frame, cfg_snapshot)
                    update_frame_dimensions(processed_frame.shape[1], processed_frame.shape[0])

                    frame_bytes = encode_frame(processed_frame)
                    self._set_frame(frame_bytes)

                    metrics['last_frame_ts'] = datetime.utcnow().isoformat()
                    metrics['rtsp_configured'] = True
                    self._set_metrics(metrics)
            except Exception as exc:  # pragma: no cover - logging path
                logger.error(f"Error in detection worker: {exc}")
            finally:
                cap.release()
                self._update_stream_status('idle', True)
                time.sleep(0.5)

    def _process_frame(
        self,
        frame: np.ndarray,
        cfg_snapshot: Dict[str, object],
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        polygon_points = [tuple(pt) for pt in cfg_snapshot.get('polygon', [])]
        target_point = tuple(cfg_snapshot.get('point', (0, 0)))
        auto_max = float(cfg_snapshot.get('auto_max_dist') or 1.0)
        frame_draw = frame.copy()

        poly_array = np.array(polygon_points, np.int32) if polygon_points else None
        if poly_array is not None and len(poly_array) >= 3:
            cv2.polylines(frame_draw, [poly_array.reshape((-1, 1, 2))], True, (255, 255, 0), 2)
        cv2.circle(frame_draw, (int(target_point[0]), int(target_point[1])), 10, (0, 255, 255), -1)

        best_score = 0.0
        best_distance: Optional[float] = None
        best_box: Optional[Tuple[int, int, int, int]] = None
        best_conf: Optional[float] = None
        inside_polygon = False

        if model is not None:
            results = model(
                frame,
                classes=[PERSON_CLASS_ID],
                verbose=False,
                device=MODEL_DEVICE or None,
            )
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    if poly_array is not None and cv2.pointPolygonTest(poly_array, (float(cx), float(cy)), False) >= 0:
                        inside_polygon = True
                        distance = math.dist(target_point, (cx, cy))
                        normalized = max(0.0, min(1.0, 1 - (distance / auto_max)))
                        score = normalized * PROXIMITY_SCALE
                        if score > best_score:
                            best_score = score
                            best_distance = distance
                            best_box = (int(x1), int(y1), int(x2), int(y2))
                            best_conf = float(box.conf[0].item()) if box.conf is not None else None

        if best_box:
            cv2.rectangle(
                frame_draw,
                (best_box[0], best_box[1]),
                (best_box[2], best_box[3]),
                (0, 255, 0),
                3,
            )

        score_int = int(round(best_score)) if best_score > 0 else 0
        cv2.putText(
            frame_draw,
            f"Proximity: {score_int}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0) if score_int > 0 else (255, 255, 255),
            2,
        )

        metrics: Dict[str, object] = {
            'stream_status': 'streaming',
            'proximity': {
                'score': score_int,
                'raw_distance': best_distance,
                'max_distance': auto_max,
                'normalized': (best_score / PROXIMITY_SCALE) if best_score > 0 else 0.0,
                'inside_polygon': inside_polygon and best_box is not None,
                'timestamp': datetime.utcnow().isoformat(),
            },
            'detection': None,
        }

        if best_box:
            metrics['detection'] = {
                'bbox': {
                    'x1': best_box[0],
                    'y1': best_box[1],
                    'x2': best_box[2],
                    'y2': best_box[3],
                },
                'confidence': best_conf,
                'centroid': {'x': (best_box[0] + best_box[2]) / 2, 'y': (best_box[1] + best_box[3]) / 2},
            }

        return frame_draw, metrics


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


def get_detection_worker() -> DetectionWorker:
    global _detection_worker
    if _detection_worker is None:
        _detection_worker = DetectionWorker()
        _detection_worker.start()
    return _detection_worker


def gen_frames():
    """Generate MJPEG frames using the detection worker output."""
    placeholder = encode_frame(
        build_message_frame(
            [
                'Waiting for stream...',
                'Configure an RTSP URL to begin.',
            ],
            background_color=(20, 20, 20),
        )
    )

    while True:
        worker = get_detection_worker()
        frame_bytes = worker.get_frame(timeout=1.0)
        if frame_bytes is None:
            yield format_mjpeg_frame(placeholder)
            sleep(1.0)
            continue
        yield format_mjpeg_frame(frame_bytes)


@app.route('/', methods=['GET'])
def index():
    """Main page with video stream and inline configuration."""
    return render_template(
        'index.html',
        config=config,
        config_json=json.dumps(serialize_config()),
        datetime=datetime,
    )


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    get_detection_worker()
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/config', methods=['GET', 'POST'])
def config_page():
    """JSON configuration endpoint for the UI."""
    if request.method == 'GET':
        return jsonify({'status': 'ok', 'config': serialize_config()})

    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({'status': 'error', 'message': 'Invalid JSON payload'}), 400

    if not isinstance(payload, dict):
        return jsonify({'status': 'error', 'message': 'Configuration payload must be an object'}), 400

    updates: Dict[str, object] = {}

    if 'rtsp_url' in payload:
        updates['rtsp_url'] = str(payload.get('rtsp_url', '')).strip()

    if 'polygon' in payload:
        try:
            updates['polygon'] = normalize_polygon(payload['polygon'])  # type: ignore[arg-type]
        except ValueError as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 400

    if 'point' in payload:
        try:
            updates['point'] = normalize_point(payload['point'])  # type: ignore[arg-type]
        except ValueError as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 400

    try:
        warnings = update_config(updates)
        save_config()
        logger.info("Configuration updated via API")
    except Exception as exc:  # pragma: no cover - logging path
        logger.error(f"Failed to save config: {exc}")
        return jsonify({'status': 'error', 'message': f'Failed to save configuration: {exc}'}), 500

    response_payload = {'status': 'ok', 'config': serialize_config()}
    if warnings:
        response_payload['warnings'] = warnings
    return jsonify(response_payload)


@app.route('/health')
def health():
    """Health check endpoint."""
    worker = get_detection_worker()
    return jsonify(
        {
            'status': 'healthy',
            'model_loaded': model is not None,
            'rtsp_configured': bool(config.get('rtsp_url')),
            'worker_status': worker.current_status(),
            'timestamp': datetime.utcnow().isoformat(),
        }
    )


@app.route('/status')
def status():
    """Status information endpoint."""
    worker = get_detection_worker()
    return jsonify(
        {
            'config': serialize_config(),
            'model_loaded': model is not None,
            'worker_status': worker.current_status(),
        }
    )


@app.route('/metrics')
def metrics():
    """Return the latest proximity metrics as JSON."""
    worker = get_detection_worker()
    return jsonify(worker.get_metrics())


@app.route('/events/metrics')
def metrics_stream():
    """Server-Sent Events endpoint for continuous metrics updates."""
    worker = get_detection_worker()

    def event_stream():
        while True:
            payload = worker.get_metrics()
            yield f"data: {json.dumps(payload)}\n\n"
            sleep(0.5)

    response = Response(event_stream(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    return response


# Initialize derived geometry fields on module load
with CONFIG_LOCK:
    _recalculate_geometry_locked()


if __name__ == '__main__':  # pragma: no cover - application entry point
    load_config()
    load_model()
    get_detection_worker()
    port = int(os.environ.get('PORT', 5000))
    server = WSGIServer(('0.0.0.0', port), app)
    logger.info("============================================================")
    logger.info("🚀 Person Detection RTSP Application Starting")
    logger.info("============================================================")
    logger.info("🌐 Starting web server on http://0.0.0.0:%s", port)
    logger.info("📺 Access the stream at: http://0.0.0.0:%s/", port)
    logger.info("⚙️  Configuration page: inline on main view")
    logger.info("❤️  Health check: http://0.0.0.0:%s/health", port)
    logger.info("============================================================")
    server.serve_forever()
