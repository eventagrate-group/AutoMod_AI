
FROM python:3.12-slim

WORKDIR /app

COPY inference/ .

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc-dev \
    libgomp1 \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet'], download_dir='/app/nltk_data')" \
    && python -c "import nltk; print(nltk.data.path); nltk.data.find('corpora/stopwords')" \
    && chmod -R 755 /app/nltk_data \
    && apt-get purge -y --auto-remove gcc g++ libc-dev \
    && rm -rf /var/lib/apt/lists/*

ENV NLTK_DATA=/app/nltk_data

EXPOSE 5001

CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5001", "--log-level=info", "app:app"]