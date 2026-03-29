#!/usr/bin/env bash
set -euo pipefail

# Deploy to Google Cloud Run
# Usage: ./deploy.sh

CONFIG="$(dirname "$0")/deploy-config.json"
PROJECT_ID=$(jq -r .project_id "$CONFIG")
SERVICE=$(jq -r .service "$CONFIG")
REGION=$(jq -r .region "$CONFIG")
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE}"

echo "Building Docker image..."
docker build -t "${IMAGE}:latest" .

echo "Pushing to Artifact Registry..."
docker push "${IMAGE}:latest"

echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --project "${PROJECT_ID}" \
  --image "${IMAGE}:latest" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080

echo "Done. Service URL:"
gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" --format 'value(status.url)'
