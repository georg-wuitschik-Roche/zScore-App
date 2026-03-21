#!/bin/bash
# Deploy z-Score Dashboard to Google Cloud Run
# Run from Google Cloud Shell

set -e

SERVICE="zscore-dashboard"
REGION="europe-west6"
PROJECT="gcp-p-hte-os-mrewa"
REPO="https://github.com/georg-wuitschik-Roche/zScore-App.git"
BUILD_SA="projects/${PROJECT}/serviceAccounts/rpt-parser-service-account@${PROJECT}.iam.gserviceaccount.com"
RUNTIME_SA="rpt-parser-service-account@${PROJECT}.iam.gserviceaccount.com"

echo "=== z-Score Dashboard Deploy ==="
echo "  Project: $PROJECT"
echo "  Service: $SERVICE"
echo "  Region:  $REGION"
echo ""

# Set project
gcloud config set project "$PROJECT"

# Clone or update repo
if [ -d "zScore-App" ]; then
    echo "Updating existing repo..."
    cd zScore-App
    git fetch origin main
    git reset --hard origin/main
else
    echo "Cloning repo..."
    git clone "$REPO"
    cd zScore-App
fi

# Deploy
echo ""
echo "Building and deploying (this may take several minutes)..."
gcloud run deploy "$SERVICE" \
    --source . \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --build-service-account "$BUILD_SA" \
    --service-account "$RUNTIME_SA" \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300

echo ""
echo "Deployment complete!"
gcloud run services describe "$SERVICE" --region "$REGION" --format="value(status.url)"
