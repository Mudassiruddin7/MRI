FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cu123.2-17.py310

WORKDIR /app

COPY src/deployment/inference/requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

COPY src/deployment/inference/main.py .
COPY src/common/preprocessing_utils.py .
COPY src/deployment/inference/inference_preprocessing_utils.py .
COPY src/deployment/inference/ml_inference_utils.py .


ENV PORT 8080

CMD gunicorn --bind 0.0.0.0:$PORT main:app --workers 1