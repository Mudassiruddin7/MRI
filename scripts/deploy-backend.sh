#!/bin/bash

# --- Configuration ---
# Usage: ./deploy_backend.sh <backend_service_name> <project_id> <region> <seg_model_gcs_uri> <class_model_gcs_uri> <result_bucket_name> <invoker_principal1> [... <invoker_principalN>] <cloud_run_sa_email>

# --- Argument Parsing and Validation ---
# Now requires at least 8 arguments: service_name, project, region, seg_model_uri, class_model_uri, result_bucket, invoker1, sa_email
if [ "$#" -lt 8 ]; then
  echo "Usage: $0 <name> <proj> <region> <seg_uri> <class_uri> <result_bucket> <invoker1> [... <invokerN>] <sa_email>"
  echo "  <backend_service_name>: Name for the backend Cloud Run service"
  echo "  <project_id>:           Your Google Cloud Project ID"
  echo "  <region>:               The Google Cloud region"
  echo "  <seg_model_gcs_uri>:    GCS URI of the segmentation model file (gs://...)"
  echo "  <class_model_gcs_uri>:  GCS URI of the classification model file (gs://...)"
  echo "  <result_bucket_name>:   Name of the GCS bucket for output masks"
  echo "  <invoker_principal>:    One or more IAM principal(s) to grant invoke access"
  echo "  <cloud_run_sa_email>:   (Required) Service account email for the Cloud Run service identity"
  exit 1
fi

BACKEND_SERVICE_NAME="$1"
PROJECT_ID="$2"
REGION="$3"
UPLOAD_BUCKET_NAME="$4"

SEG_MODEL_URI="$5"
CLASS_MODEL_URI="$6"
RESULT_BUCKET_NAME="$7"
shift 7

NUM_REMAINING_ARGS=$#
if [ $NUM_REMAINING_ARGS -lt 2 ]; then 
    echo "Error: At least one invoker principal must be provided before the required service account email."
    exit 1
fi

CLOUD_RUN_SA="${!NUM_REMAINING_ARGS}" 
INVOKER_PRINCIPALS=("${@:1:$#-1}")

if [[ ! "$CLOUD_RUN_SA" == *"@"* || ! "$CLOUD_RUN_SA" == *".iam.gserviceaccount.com"* ]]; then
    echo "Error: The last argument ('$CLOUD_RUN_SA') does not look like a valid service account email (*@*.iam.gserviceaccount.com)."
    exit 1
fi
echo "Service Account for Cloud Run instance: ${CLOUD_RUN_SA}"
echo "Invoker Principals to grant access: ${INVOKER_PRINCIPALS[*]}"
echo "Segmentation Model URI: ${SEG_MODEL_URI}"
echo "Classification Model URI: ${CLASS_MODEL_URI}"

# --- Use Environment Variables or Defaults ---
REPO_DIR="${REPO_DIR:-src/deployment/inference}" # Or your backend_service dir
DOCKER_REPO="${DOCKER_REPO:-discai}"
IMAGE_NAME="${IMAGE_NAME:-mri-backend-service}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}" # Default to L4 based on previous script
GPU_COUNT="${GPU_COUNT:-1}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
MEMORY="${MEMORY:-16Gi}"
CPU="${CPU:-4}"
TIMEOUT="${TIMEOUT:-900s}"
CONCURRENCY="${CONCURRENCY:-1}"
PORT="8080"

BACKEND_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${DOCKER_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

cd ..

# --- Error Handling ---
set -e
set -u
set -o pipefail

if [ ! -d "$REPO_DIR" ]; then
    echo "Error: Backend source directory '$REPO_DIR' not found in ${PWD}."
    echo "Please run this script from the repository root directory or set REPO_DIR."
    exit 1
fi
echo "Using backend source directory: ${PWD}/${REPO_DIR}"

echo "Configuring Docker authentication for ${REGION}..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

echo "Building Docker image: ${BACKEND_IMAGE_URI}"
docker build -t "${BACKEND_IMAGE_URI}" -f "${REPO_DIR}/Dockerfile" .

echo "Pushing Docker image to Artifact Registry: ${BACKEND_IMAGE_URI}"
docker push "${BACKEND_IMAGE_URI}"

echo "Deploying service '${BACKEND_SERVICE_NAME}' with image '${BACKEND_IMAGE_URI}' to Cloud Run..."
echo "---"

gcloud beta run deploy "${BACKEND_SERVICE_NAME}" \
    --image "${BACKEND_IMAGE_URI}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --platform "managed" \
    --execution-environment "gen2" \
    --port "${PORT}" \
    --timeout "${TIMEOUT}" \
    --concurrency "${CONCURRENCY}" \
    --min-instances "${MIN_INSTANCES}" \
    --max-instances "${MAX_INSTANCES}" \
    --memory "${MEMORY}" \
    --cpu "${CPU}" \
    --no-gpu-zonal-redundancy \
    --no-allow-unauthenticated \
    --gpu=${GPU_COUNT} \
    --gpu-type=${GPU_TYPE} \
    --set-env-vars="SEG_MODEL_URI=${SEG_MODEL_URI},CLASS_MODEL_URI=${CLASS_MODEL_URI},RESULT_BUCKET_NAME=${RESULT_BUCKET_NAME},UPLOAD_BUCKET_NAME=${UPLOAD_BUCKET_NAME}" \
    --service-account="${CLOUD_RUN_SA}"



echo "---"
echo "Retrieving service URL..."
SERVICE_URL=$(gcloud run services describe "${BACKEND_SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --platform "managed" \
    --format 'value(status.url)' || echo "failed-to-get-url")

if [[ -z "$SERVICE_URL" || "$SERVICE_URL" == "failed-to-get-url" ]]; then
    echo "Deployment potentially failed or service URL could not be retrieved immediately."
    echo "Check Cloud Run console for deployment status and logs. IAM permissions not set."
    exit 1
else
    echo "Service deployed/updated successfully. URL: ${SERVICE_URL}"
    echo "Setting IAM invoker role for specified principals..."
    for principal in "${INVOKER_PRINCIPALS[@]}"; do
        if [[ $principal == *"@"* && $principal != *":"* ]]; then
            principal_member="user:${principal}"
        else
            principal_member="${principal}"
        fi
        echo "Adding roles/run.invoker for: ${principal_member}"
        gcloud run services add-iam-policy-binding "${BACKEND_SERVICE_NAME}" \
            --member="${principal_member}" \
            --role="roles/run.invoker" \
            --project "${PROJECT_ID}" \
            --region "${REGION}" \
            --platform=managed \
            --quiet || echo "Warning: Failed to grant IAM permission to ${principal_member}."
    done
    echo "IAM policy update attempted for specified principals."
    echo "---"
    echo "Deployment script finished."
    echo "Backend Service URL: ${SERVICE_URL}"
    echo "Ensure the Service Account (${CLOUD_RUN_SA}) has necessary GCS read/write permissions."
    echo "---"
fi
