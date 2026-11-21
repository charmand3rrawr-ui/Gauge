# Gauge: run the novel engine headless, in a container, anywhere.
FROM python:3.12-slim

# Non-root user; data lives on a mounted volume.
RUN useradd -m -u 1000 gauge
WORKDIR /app

# Install dependencies first for layer caching.
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install Gauge itself.
COPY . .
RUN pip install --no-cache-dir . && mkdir -p /data && chown -R gauge:gauge /app /data
USER gauge

ENV GAUGE_DATA_DIR=/data
EXPOSE 8800

# Bind to all interfaces INSIDE the container; publish it with care (see docs/DEPLOY.md).
# A token is required for a non-loopback bind: set GAUGE_SERVER_TOKEN or pass --token.
ENTRYPOINT ["gauge"]
CMD ["--data-dir", "/data", "serve", "--host", "0.0.0.0", "--port", "8800"]
