#!/bin/bash


if [ "$#" -lt 8 ]; then
  echo "Usage: $0 <frontend_service_name> <project_id> <region> <upload_bucket_name> <backend_service_url> <frontend_sa_email> <user_email1> [<user_email2> ...]"
  echo "  <frontend_service_name>: Name for the frontend Cloud Run service"
  echo "  <project_id>:            Your Google Cloud Project ID"
  echo "  <region>:                The Google Cloud region"
  echo "  <upload_bucket_name>:    Name of the GCS bucket for user uploads"
  echo "  <RESULT_bucket_name>:    Name of the GCS bucket for user uploads"
  echo "  <backend_service_url>:   Full HTTPS URL of the deployed backend service"
  echo "  <frontend_sa_email>:     Service account email the frontend service will run as"
  echo "  <user_email1> ... :      Email addresses of users to grant invoke access"
  exit 1
fi

FRONTEND_SERVICE_NAME="$1"
PROJECT_ID="$2"
REGION="$3"
UPLOAD_BUCKET_NAME="$4"
RESULT_BUCKET_NAME="$5"
BACKEND_SERVICE_URL="$6"
FRONTEND_SA_EMAIL="$7"
shift 7

INVOKER_USERS=("$@")

REPOSITORY="discai" 
WEB_APP_IMAGE_NAME="mri-upload-website" 
WEB_APP_IMAGE_TAG="latest" 
WEB_APP_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${WEB_APP_IMAGE_NAME}:${WEB_APP_IMAGE_TAG}"
FRONTEND_SOURCE_DIR="src/deployment/upload" 

set -e
set -u
set -o pipefail

cd ..

if [ ! -d "$FRONTEND_SOURCE_DIR" ]; then
    echo "Error: Frontend source directory '$FRONTEND_SOURCE_DIR' not found in ${PWD}."
    echo "Please run this script from the repository root directory or adjust FRONTEND_SOURCE_DIR."
    exit 1
fi
echo "Using frontend source directory: ${PWD}/${FRONTEND_SOURCE_DIR}"
# --------------------------------------------------------------------

# --- Configure Docker Auth ---
echo "Configuring Docker authentication for ${REGION}..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# --- Build Docker image ---
echo "Building Docker image: ${WEB_APP_IMAGE_URI}"
docker build -t "${WEB_APP_IMAGE_URI}" -f "${FRONTEND_SOURCE_DIR}/Dockerfile" .

# --- Push Docker image ---
echo "Pushing Docker image to Artifact Registry: ${WEB_APP_IMAGE_URI}"
docker push "${WEB_APP_IMAGE_URI}"

# --- Grant access to specified users *before* deployment (Mimics Original) ---
# NOTE: Might fail if the service doesn't exist yet on the very first run.
echo "Attempting to grant invoke access to specified users before deployment..."
IAM_FAILED=0
for user_email in "${INVOKER_USERS[@]}"; do
    echo "Attempting grant roles/run.invoker to user:${user_email}..."
    gcloud run services add-iam-policy-binding "${FRONTEND_SERVICE_NAME}" \
        --member="user:${user_email}" \
        --role="roles/run.invoker" \
        --project "${PROJECT_ID}" \
        --region "${REGION}" \
        --platform=managed --quiet || {
            echo "Warning: Failed to grant IAM permission to user:${user_email}. Service might not exist yet, or insufficient permissions."
            IAM_FAILED=1
        }
done
if [ $IAM_FAILED -eq 0 ]; then
    echo "IAM policy update commands executed for specified users."
else
    echo "One or more IAM policy updates failed (this may be expected on first deploy)."
fi
echo "---"

# --- Deploy to Cloud Run and Retrieve URL ---
echo "Deploying service '${FRONTEND_SERVICE_NAME}' to Cloud Run..."
# Note: Removed --format from deploy command to see full output, added it to describe later
gcloud run deploy "${FRONTEND_SERVICE_NAME}" \
    --image "${WEB_APP_IMAGE_URI}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --platform managed \
    --service-account="${FRONTEND_SA_EMAIL}" \
    --set-env-vars="PROJECT_ID=${PROJECT_ID},UPLOAD_BUCKET_NAME=${UPLOAD_BUCKET_NAME},BACKEND_SERVICE_URL=${BACKEND_SERVICE_URL},FRONTEND_SERVICE_ACCOUNT_EMAIL=${FRONTEND_SA_EMAIL},RESULT_BUCKET_NAME=${RESULT_BUCKET_NAME}" \
    --allow-unauthenticated \

# Retrieve URL after deployment
echo "Retrieving service URL..."
SERVICE_URL=$(gcloud run services describe "${FRONTEND_SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --platform "managed" \
    --format 'value(status.url)' || echo "failed-to-get-url")


if [[ -n "$SERVICE_URL" && "$SERVICE_URL" != "failed-to-get-url" ]]; then
    echo "---"
    echo "Deployment successful!"
    echo "Service URL: $SERVICE_URL"
    echo "Ensure the Service Account (${FRONTEND_SA_EMAIL}) has necessary permissions."
    echo "Ensure users (${INVOKER_USERS[*]}) have necessary permissions (check IAM if pre-deploy grant failed)."
    echo "---"

    # --- Run gcloud run services proxy (Mimics Original) ---
    echo "Starting gcloud run services proxy..."
    # Run in background, suppress job control messages temporarily
    set +m
    gcloud run services proxy "${FRONTEND_SERVICE_NAME}" --port=8080 --project="${PROJECT_ID}" --region="${REGION}" &
    set -m
    echo "gcloud run services proxy started in the background. Access via http://localhost:8080"
    echo "NOTE: Proxy uses your local gcloud credentials for authentication."
    echo "---"

else
    echo "---"
    echo "Deployment failed or URL could not be retrieved."
    echo "Check Cloud Run console and Cloud Build logs for errors."
    echo "---"
    exit 1
fi