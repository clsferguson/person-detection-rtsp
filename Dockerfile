FROM ubuntu:24.04

RUN apt-get update && apt-get install -y python3 python3-pip libglib2.0-0 libsm6 libxext6 libgomp1 && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

WORKDIR /app


COPY . .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

EXPOSE 5000

CMD ["python3", "app/main.py"]
