FROM nvidia/cuda:12.6.1-base-ubuntu24.04

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO11 model
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY . .

EXPOSE 5000

CMD ["python3", "app/main.py"]
RUN python3 -m pip install --upgrade pip
