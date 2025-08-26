FROM python:3.12-slim

WORKDIR /app

COPY inference/ .

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc-dev \
    libgomp1 \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /app/nltk_data \
    && mkdir -p /app/nltk_data/corpora/stopwords \
    && cp -r /app/nltk_data/corpora/stopwords/* /app/nltk_data/corpora/stopwords/ \
    && find /app/nltk_data -type f -ls \
    && ls -l /app/nltk_data \
    && ls -l /app/nltk_data/corpora \
    && ls -l /app/nltk_data/corpora/stopwords \
    && python -c "import nltk; nltk.data.path.append('/app/nltk_data'); from nltk.corpus import stopwords; print('Stopwords sample:', stopwords.words('english')[:10])" \
    && chmod -R 777 /app/nltk_data \
    && apt-get purge -y --auto-remove gcc g++ libc-dev \
    && rm -rf /var/lib/apt/lists/*

ENV NLTK_DATA=/app/nltk_data

EXPOSE 5001

CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5001", "--log-level=info", "app:app"]
