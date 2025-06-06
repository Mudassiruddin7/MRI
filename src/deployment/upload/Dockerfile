FROM python:3.10-slim-buster

WORKDIR /app

COPY src/deployment/upload/requirements.txt .
RUN pip install -r requirements.txt

COPY src/deployment/upload/app/main.py .
COPY src/deployment/upload/app/templates /app/templates

ENV PORT 8080

CMD gunicorn --bind 0.0.0.0:$PORT main:app --workers 1 --threads 8 --timeout 960

