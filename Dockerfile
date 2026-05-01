FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY strohhalme/ strohhalme/

ENV STROHHALME_ROOT=/data
ENV PYTHONUNBUFFERED=1

VOLUME ["/data/raw", "/data/processed", "/results"]

ENTRYPOINT ["python", "-m", "strohhalme.pipeline"]
CMD ["--help"]
