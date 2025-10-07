import os
import json
import threading
import time
from flask import Flask, Response, render_template, request
from gevent.pywsgi import WSGIServer
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
model = YOLO('yolo11n.pt')
config = {
    'rtsp_url': 'rtsp://example.com/stream',
    'polygon': [(0, 0), (100, 0), (100, 100), (0, 100)],
    'point': (50, 50),
    'max_dist': 100
}

def gen_frames():
    cap = cv2.VideoCapture(config['rtsp_url'])
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, classes=[0])  # person class
        closest_val = 0
        best_box = None
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                poly = np.array(config['polygon'], np.int32)
                if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                    dist = np.linalg.norm(np.array(config['point']) - np.array([cx, cy]))
                    val = max(0, 1000 * (1 - dist / config['max_dist']))
                    if val > closest_val:
                        closest_val = val
                        best_box = (x1, y1, x2, y2)
        if best_box:
            cv2.rectangle(frame, (int(best_box[0]), int(best_box[1])), (int(best_box[2]), int(best_box[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"Closest Proximity: {int(closest_val)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/config', methods=['GET', 'POST'])
def config_page():
    if request.method == 'POST':
        config['rtsp_url'] = request.form['rtsp']
        config['polygon'] = [tuple(map(int, p.strip('()').split(','))) for p in request.form['polygon'].split()]
        config['point'] = tuple(map(int, request.form['point'].strip('()').split(',')))
        config['max_dist'] = int(request.form['max_dist'])
        with open('config/config.json', 'w') as f:
            json.dump(config, f)
    return render_template('config.html', config=config)

if __name__ == '__main__':
    if os.path.exists('config/config.json'):
        with open('config/config.json') as f:
            config.update(json.load(f))
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
