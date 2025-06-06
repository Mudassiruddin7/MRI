FROM python:3.9  

WORKDIR /app

COPY src/data/preprocessing.py .
COPY src/common/preprocessing_utils.py .

RUN pip install --no-cache-dir tensorflow SimpleITK scikit-image opencv-python-headless google-cloud-storage matplotlib

ENTRYPOINT ["python", "preprocessing.py"]