#!/usr/bin/env bash
set -euo pipefail

# Deploy to Google Cloud Run
# Usage: ./deploy.sh

PROJECT_ID="gcp-p-hte-os-mrewa"
SERVICE="zscore-dashboard"
REGION="europe-west6"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE}"

echo "Building Docker image..."
docker build -t "${IMAGE}:latest" .

echo "Pushing to Artifact Registry..."
docker push "${IMAGE}:latest"

echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}:latest" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080

echo "Done. Service URL:"
gcloud run services describe "${SERVICE}" --region "${REGION}" --format 'value(status.url)'
