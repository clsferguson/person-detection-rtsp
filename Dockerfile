FROM nvidia/cuda:12.6.1-base-ubuntu24.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

WORKDIR /app

# Copy application files
COPY app/ ./app/
COPY templates/ ./templates/
COPY config/ ./config/
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
EXPOSE 5000

CMD ["python3", "app/main.py"]
