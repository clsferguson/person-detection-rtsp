import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, Response, render_template, request, jsonify
from gevent.pywsgi import WSGIServer
from ultralytics import YOLO
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = None
config = {
    'rtsp_url': '',
    'polygon': [(0, 0), (640, 0), (640, 480), (0, 480)],
    'point': (320, 240),
    'max_dist': 200
}

def load_model():
    """Load YOLO model with error handling."""
    global model
    try:
        logger.info("Loading YOLO model...")
        model = YOLO('yolo11n.pt')
        logger.info("✓ YOLO model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load YOLO model: {e}")
        return False

def load_config():
    """Load configuration from file."""
    global config
    config_path = 'config/config.json'
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                loaded = json.load(f)
                config.update(loaded)
            logger.info(f"✓ Configuration loaded from {config_path}")
            logger.info(f"  RTSP URL: {config['rtsp_url'] or '(not configured)'}")
        except Exception as e:
            logger.error(f"✗ Failed to load config: {e}")
    else:
        logger.warning(f"⚠ Config file not found: {config_path}")

def gen_frames():
    """Generate video frames with person detection."""
    if not config['rtsp_url']:
        logger.warning("⚠ RTSP URL not configured")
        # Generate placeholder frame
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No RTSP Stream Configured", (80, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Go to /config to set up", (120, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            import time
            time.sleep(1)
        return

    logger.info(f"Opening RTSP stream: {config['rtsp_url']}")
    cap = cv2.VideoCapture(config['rtsp_url'])
    
    if not cap.isOpened():
        logger.error("✗ Failed to open RTSP stream")
        # Generate error frame
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Stream Connection Failed", (100, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            import time
            time.sleep(1)
        return

    logger.info("✓ RTSP stream opened successfully")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Stream ended or failed to read frame")
                break
            
            frame_count += 1
            
            # Run person detection
            results = model(frame, classes=[0], verbose=False)  # person class, suppress logs
            closest_val = 0
            best_box = None
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    poly = np.array(config['polygon'], np.int32)
                    
                    # Check if center is inside polygon
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        dist = np.linalg.norm(np.array(config['point']) - np.array([cx, cy]))
                        val = max(0, 1000 * (1 - dist / config['max_dist']))
                        if val > closest_val:
                            closest_val = val
                            best_box = (x1, y1, x2, y2)
            
            # Draw detection zone polygon
            poly = np.array(config['polygon'], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [poly], True, (255, 255, 0), 2)
            
            # Draw target point
            cv2.circle(frame, config['point'], 10, (0, 255, 255), -1)
            
            # Draw best detection box
            if best_box:
                cv2.rectangle(frame, (int(best_box[0]), int(best_box[1])), 
                            (int(best_box[2]), int(best_box[3])), (0, 255, 0), 3)
            
            # Draw proximity value
            cv2.putText(frame, f"Proximity: {int(closest_val)}", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if closest_val > 0 else (255, 255, 255), 2)
            
            # Draw frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Encode and yield frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
    except Exception as e:
        logger.error(f"Error in frame generation: {e}")
    finally:
        cap.release()
        logger.info(f"Stream closed after {frame_count} frames")

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
    """Configuration page."""
    global config
    message = None
    
    if request.method == 'POST':
        try:
            config['rtsp_url'] = request.form['rtsp'].strip()
            
            # Parse polygon
            polygon_str = request.form['polygon'].strip()
            config['polygon'] = [
                tuple(map(int, p.strip('()').split(','))) 
                for p in polygon_str.split()
            ]
            
            # Parse point
            point_str = request.form['point'].strip('()')
            config['point'] = tuple(map(int, point_str.split(',')))
            
            # Parse max distance
            config['max_dist'] = int(request.form['max_dist'])
            
            # Save to file
            with open('config/config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            message = "✓ Configuration saved successfully!"
            logger.info("Configuration updated and saved")
            
        except Exception as e:
            message = f"✗ Error saving configuration: {e}"
            logger.error(f"Failed to save config: {e}")
    
    return render_template('config.html', config=config, message=message)

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'rtsp_configured': bool(config['rtsp_url']),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/status')
def status():
    """Status information page."""
    return jsonify({
        'config': config,
        'model': 'yolo11n.pt',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Person Detection RTSP Application Starting")
    logger.info("=" * 60)
    
    # Load model
    if not load_model():
        logger.error("Failed to initialize. Exiting.")
        sys.exit(1)
    
    # Load configuration
    load_config()
    
    # Start server
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
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
