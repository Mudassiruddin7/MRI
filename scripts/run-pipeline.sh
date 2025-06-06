#!/bin/bash

if [ "$#" -ne 9 ]; then
  echo "Usage: $0 <project_id> <region> <raw_data_bucket_suffix> <processed_data_bucket_suffix> <pipeline_root_bucket_suffix> <tensorboard_instance> <service_account_email> <run_segmentation_flag>"
  echo "  <raw_data_bucket_suffix>: Suffix for bucket holding original uploads (e.g., 'raw-data')"
  echo "  <processed_data_bucket_suffix>: Suffix for bucket holding seg processed data (e.g., 'processed-data')"
  echo "  <pipeline_root_bucket_suffix>: Suffix for bucket holding pipeline artifacts/results/models (e.g., 'pipeline')"
  echo "  <deployment_bucket_suffix>: Suffix for bucket holding pipeline artifacts/results/models (e.g., 'pipeline')"
  echo "  <tensorboard_instance>: Full Tensorboard resource name (or empty string \"\")"
  echo "  <service_account_email>: Service account for pipeline execution"
  echo "  <run_segmentation_flag>: 'true' to run segmentation training, 'false' otherwise"
  exit 1
fi

PROJECT_ID="$1"
REGION="$2"
RAW_DATA_BUCKET_SUFFIX="$3"         
PROCESSED_DATA_BUCKET_SUFFIX="$4"   
PIPELINE_ROOT_BUCKET_SUFFIX="$5"    
DEPLOYMMENT_BUCKET_SUFFIX="$6"   

TENSORBOARD_INSTANCE="$7"        
SERVICE_ACCOUNT="$8"
RUN_SEGMENTATION_PIPELINE="$9" 

REPOSITORY="discai" 
IMAGE_TAG="latest" 

PREPROCESS_SEG_IMAGE_NAME="preprocessing-seg-image"
TRAIN_SEG_IMAGE_NAME="training-seg-image"
SEG_MODEL_SERVING_IMAGE_NAME="training-seg-image" 
PREPROCESS_CLASS_IMAGE_NAME="preprocess-corrections-image"
CLASS_TRAIN_IMAGE_NAME="training-class-image"
EVALUATION_IMAGE_NAME="evaluation-image"
TARGET_URL="https://mri-backend-service-yew6jcdx7a-uc.a.run.app/reload-models"

PREPROCESS_SEG_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${PREPROCESS_SEG_IMAGE_NAME}:${IMAGE_TAG}"
TRAIN_SEG_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${TRAIN_SEG_IMAGE_NAME}:${IMAGE_TAG}"
SEG_MODEL_SERVING_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SEG_MODEL_SERVING_IMAGE_NAME}:${IMAGE_TAG}" # Adjust if needed
PREPROCESS_CLASS_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${PREPROCESS_CLASS_IMAGE_NAME}:${IMAGE_TAG}"
CLASS_TRAIN_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${CLASS_TRAIN_IMAGE_NAME}:${IMAGE_TAG}"
EVALUATION_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${EVALUATION_IMAGE_NAME}:${IMAGE_TAG}"
PIPELINE_ROOT="gs://${PROJECT_ID}-${PIPELINE_ROOT_BUCKET_SUFFIX}"

SEG_BASE_DIR="/gcs/${PROJECT_ID}-${RAW_DATA_BUCKET_SUFFIX}/SPIDER/" # Example input path, adjust if needed
SEG_OUTPUT_DIR="/gcs/${PROJECT_ID}-${PROCESSED_DATA_BUCKET_SUFFIX}/segmentation_processed/" # Output for seg preprocess
SEG_MODEL_OUTPUT_DIR="gs://${PROJECT_ID}-${PIPELINE_ROOT_BUCKET_SUFFIX}/segmentation_model/" # Model output using pipeline/result suffix

CORRECTIONS_GCS_DIR="gs://${PROJECT_ID}-${DEPLOYMMENT_BUCKET_SUFFIX}/corrections/" # Corrections using pipeline/result suffix
ACCEPTED_NO_CORRECTIONS_GCS_DIR="gs://${PROJECT_ID}-${DEPLOYMMENT_BUCKET_SUFFIX}/accepted_predictions/" 
ORIGINAL_SEQUENCES_GCS_DIR="gs://${PROJECT_ID}-${DEPLOYMMENT_BUCKET_SUFFIX}/input_sequences/" # Originals using raw data suffix
CLASSIFICATION_OUTPUT_DIR="gs://${PROJECT_ID}-${PROCESSED_DATA_BUCKET_SUFFIX}/classification_preprocessed/"
ORIGINAL_PROCESSED__DIR="gs://${PROJECT_ID}-${PROCESSED_DATA_BUCKET_SUFFIX}/Full Data/" # For training input


ORIGINAL_PROCESSED_TRAIN_DIR="/gcs/${PROJECT_ID}-${PROCESSED_DATA_BUCKET_SUFFIX}/Full Data/train/" # For training input
ORIGINAL_PROCESSED_VAL_DIR="/gcs/${PROJECT_ID}-${PROCESSED_DATA_BUCKET_SUFFIX}/Full Data/val/"   # For training input
ORIGINAL_PROCESSED_TEST_DIR="/gcs/${PROJECT_ID}-${PROCESSED_DATA_BUCKET_SUFFIX}/Full Data/test/"

EVALUATION_DATA_GCS_DIR="gs://${PROJECT_ID}-${PROCESSED_DATA_BUCKET_SUFFIX}/Full Data/test/"
PRODUCTION_MODEL_URI="gs://${PROJECT_ID}-${DEPLOYMMENT_BUCKET_SUFFIX}/models/"
EVALUATION_METRIC_NAME="accuracy"
EVALUATION_METRIC_THRESHOLD="0.0"

DISPLAY_NAME="mri-unified-run-$(date +%Y%m%d%H%M%S)"

set -e; set -u; set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd ) 
cd "${PROJECT_ROOT_DIR}" || exit 1 


echo "Building Corrections Preprocessing Docker image: ${PREPROCESS_CLASS_IMAGE_URI}"
docker build -t "${PREPROCESS_CLASS_IMAGE_URI}" -f src/data/Dockerfile.preprocess_class .
echo "Pushing Corrections Preprocessing Docker image: ${PREPROCESS_CLASS_IMAGE_URI}"
docker push "${PREPROCESS_CLASS_IMAGE_URI}"

echo "Building/Pushing Classification Training Image: ${CLASS_TRAIN_IMAGE_URI}"
docker build -t "${CLASS_TRAIN_IMAGE_URI}" -f src/training/Dockerfile.class . 
docker push "${CLASS_TRAIN_IMAGE_URI}"

echo "Building/Pushing Evaluation Image: ${EVALUATION_IMAGE_URI}"
docker build -t "${EVALUATION_IMAGE_URI}" -f src/evaluation/Dockerfile.evaluate . # Assuming this path
echo "Pushing Evaluation Image: ${EVALUATION_IMAGE_URI}"
docker push "${EVALUATION_IMAGE_URI}"

if [[ "$RUN_SEGMENTATION_PIPELINE" == "true" ]]; then
  echo "Building Segmentation Preprocessing Docker image: ${PREPROCESS_SEG_IMAGE_URI}"
  docker build -t "${PREPROCESS_SEG_IMAGE_URI}" -f src/data/Dockerfile . 
  echo "Pushing Segmentation Preprocessing Docker image: ${PREPROCESS_SEG_IMAGE_URI}"
  docker push "${PREPROCESS_SEG_IMAGE_URI}"

  echo "Building Segmentation Training/Serving Docker image: ${TRAIN_SEG_IMAGE_URI}"
  docker build -t "${TRAIN_SEG_IMAGE_URI}" -f src/training/Dockerfile.seg .
  echo "Pushing Segmentation Training Docker image: ${TRAIN_SEG_IMAGE_URI}"
  docker push "${TRAIN_SEG_IMAGE_URI}"
  if [[ "${SEG_MODEL_SERVING_IMAGE_URI}" != "${TRAIN_SEG_IMAGE_URI}" ]]; then
      echo "Pushing Segmentation Serving image: ${SEG_MODEL_SERVING_IMAGE_URI}"
      docker push "${SEG_MODEL_SERVING_IMAGE_URI}"
  fi
else
  echo "Skipping build/push for Segmentation images as run_segmentation_flag is false."
fi


echo "Compiling and submitting the Unified pipeline..."
python src/pipeline.py \
    --project-id="${PROJECT_ID}" \
    --region="${REGION}" \
    --pipeline-root="${PIPELINE_ROOT}" \
    --display-name="${DISPLAY_NAME}" \
    --service-account="${SERVICE_ACCOUNT}" \
    --tensorboard-instance="${TENSORBOARD_INSTANCE}" \
    --run-segmentation-pipeline="${RUN_SEGMENTATION_PIPELINE}" \
    --seg-base-dir="${SEG_BASE_DIR}" \
    --seg-output-dir="${SEG_OUTPUT_DIR}" \
    --seg-model-output-dir="${SEG_MODEL_OUTPUT_DIR}" \
    --preprocess-seg-image-uri="${PREPROCESS_SEG_IMAGE_URI}" \
    --train-seg-image-uri="${TRAIN_SEG_IMAGE_URI}" \
    --seg-model-serving-image-uri="${SEG_MODEL_SERVING_IMAGE_URI}" \
    --corrections-gcs-dir="${CORRECTIONS_GCS_DIR}" \
    --original-sequences-gcs-dir="${ORIGINAL_SEQUENCES_GCS_DIR}" \
    --preprocess-class-image-uri="${PREPROCESS_CLASS_IMAGE_URI}" \
    --classification-processed-data-gcs-path="${CLASSIFICATION_OUTPUT_DIR}" \
    --evaluation-data-gcs-dir="${EVALUATION_DATA_GCS_DIR}" \
    --evaluation-image-uri="${EVALUATION_IMAGE_URI}" \
    --production-model-uri="${PRODUCTION_MODEL_URI}" \
    --evaluation-metric-name="${EVALUATION_METRIC_NAME}" \
    --evaluation-metric-threshold="${EVALUATION_METRIC_THRESHOLD}" \
    --seg-model-display-name="mri-segmentation-model" \
    --seg-epochs=50 \
    --seg-batch-size=8 \
    --seg-image-height=512 \
    --seg-image-width=512 \
    --patch-height=40 \
    --patch-width=80 \
    --preprocess-class-image-uri="${PREPROCESS_CLASS_IMAGE_URI}" \
    --original-processed-train-dir="${ORIGINAL_PROCESSED_TRAIN_DIR}" \
    --original-processed-val-dir="${ORIGINAL_PROCESSED_VAL_DIR}" \
    --original-processed-test-dir="${ORIGINAL_PROCESSED_TEST_DIR}" \
    --class-train-image-uri="${CLASS_TRAIN_IMAGE_URI}" \
    --class-serving-image-uri="${CLASS_TRAIN_IMAGE_URI}" \
    --accepted-predictions-gcs-dir="${ACCEPTED_NO_CORRECTIONS_GCS_DIR}" \
    --target-url="${TARGET_URL}" 

echo "Pipeline submission initiated. Check the Vertex AI Pipelines console for status."
