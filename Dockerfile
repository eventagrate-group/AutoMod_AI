# Use official Python 3.12 slim image for CPU compatibility
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy inference directory contents
COPY inference/ .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader -d /root/nltk_data punkt punkt_tab stopwords wordnet

# Expose port 5001
EXPOSE 5001

# Run Gunicorn with 4 workers
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5001", "--log-level=info", "app:app"]