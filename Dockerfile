FROM python:3.12-slim

WORKDIR /app

COPY inference/ .

# Install build dependencies and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc libc-dev \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 5001

CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5001", "--log-level=info", "app:app"]