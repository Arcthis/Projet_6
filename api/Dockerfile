FROM python:3.9-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3000

CMD ["bentoml", "serve", "energy_prediction_service:lhuckqagjsryrxh3"]

