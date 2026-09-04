#!/usr/bin/env bash
# ==============================================================================
# DepthWizard (SIH26175) — Google Cloud Run Production Deployment Script
# Author: DepthWizard Systems Engineering Lead
# ==============================================================================
set -euo pipefail

echo "=================================================================="
echo " DEPTHWIZARD SIH26175 — GOOGLE CLOUD RUN PRODUCTION DEPLOYMENT"
echo "=================================================================="

# 1. Resolve GCP Project ID and Project Number
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    echo "ERROR: GCP project is not configured in gcloud CLI."
    echo "Run: gcloud config set project <YOUR_PROJECT_ID>"
    exit 1
fi

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
REGION="${REGION:-asia-south1}"
SERVICE_NAME="${SERVICE_NAME:-depthwizard-api}"
REPO_NAME="${REPO_NAME:-depthwizard-repo}"
BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-depthwizard-scenes}"
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"

echo "Project ID:       $PROJECT_ID"
echo "Project Number:   $PROJECT_NUMBER"
echo "Target Region:    $REGION"
echo "Service Name:     $SERVICE_NAME"
echo "Storage Bucket:   gs://${BUCKET_NAME}"
echo "Image Tag:        $IMAGE_TAG"
echo "=================================================================="

# 2. Enable Required Google Cloud APIs
echo "--> [1/6] Enabling required Google Cloud APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com \
    --project="$PROJECT_ID"

# 3. Create Artifact Registry Repository (if not existing)
echo "--> [2/6] Ensuring Artifact Registry repository exists..."
if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
    echo "    Creating Artifact Registry repository '$REPO_NAME' in $REGION..."
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="DepthWizard SIH26175 Production Containers" \
        --project="$PROJECT_ID"
else
    echo "    Artifact Registry repository '$REPO_NAME' exists."
fi

# 4. Create Google Cloud Storage Bucket for Persistent Scenes
echo "--> [3/6] Ensuring Cloud Storage bucket exists..."
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="$PROJECT_ID" >/dev/null 2>&1; then
    echo "    Creating GCS bucket 'gs://${BUCKET_NAME}' in $REGION..."
    gcloud storage buckets create "gs://${BUCKET_NAME}" \
        --location="$REGION" \
        --uniform-bucket-level-access \
        --project="$PROJECT_ID"
else
    echo "    Cloud Storage bucket 'gs://${BUCKET_NAME}' exists."
fi

# 5. Build and Push Container Image with Cloud Build
echo "--> [4/6] Building production container image via Cloud Build..."
gcloud builds submit \
    --config deploy/gcp/cloudbuild.yaml \
    --substitutions="_IMAGE_TAG=${IMAGE_TAG}" \
    --project="$PROJECT_ID" \
    .

# 6. Deploy to Google Cloud Run with GCS Volume Mount
echo "--> [5/6] Deploying service '$SERVICE_NAME' to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_TAG" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --cpu 2 \
    --memory 4Gi \
    --concurrency 1 \
    --timeout 300 \
    --max-instances 2 \
    --min-instances 0 \
    --cpu-boost \
    --execution-environment gen2 \
    --add-volume "name=depthwizard-storage,type=cloud-storage,bucket=${BUCKET_NAME}" \
    --add-volume-mount "volume=depthwizard-storage,mount-path=/mnt/depthwizard" \
    --set-env-vars "DEPTHWIZARD_DEVICE=cpu,DEPTHWIZARD_MODEL_FAMILY=M3-FINAL,DEPTHWIZARD_STORAGE_DIR=/mnt/depthwizard/scenes,DEPTHWIZARD_UPLOAD_TEMP_DIR=/mnt/depthwizard/uploads,DEPTHWIZARD_CORS_ORIGINS=https://depth-wizard-sih-26175.vercel.app" \
    --project="$PROJECT_ID"

# 7. Obtain and Validate Public Service URL
echo "--> [6/6] Verifying public endpoint..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" --format='value(status.url)' --project="$PROJECT_ID")

echo "=================================================================="
echo "DEPLOYMENT COMPLETE!"
echo "Public Cloud Run URL: $SERVICE_URL"
echo "Health Endpoint:      $SERVICE_URL/health"
echo "API Docs:             $SERVICE_URL/docs"
echo "=================================================================="
echo ""
echo "NEXT STEPS FOR VERCEL FRONTEND:"
echo "1. Set VITE_API_BASE_URL=$SERVICE_URL"
echo "2. Set VITE_DEPLOYMENT_MODE=workspace"
echo "3. Redeploy frontend."
