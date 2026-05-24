#!/bin/bash

# ==============================================================================
# Google Cloud Run Deployment Helper for Wine Quality Workspace
# ==============================================================================
# This script guides you through deploying your containerized Streamlit application.
# It uses Google Cloud Build (compiling in the cloud, no local Docker needed)
# and deploys to serverless Google Cloud Run.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration (Replace with your actual project ID and region)
PROJECT_ID="vintedge-wine-workspace-497319"
REGION="us-central1"
SERVICE_NAME="wine-quality-app"
REPO_NAME="wine-app-repo"

echo "==============================================="
echo "🍷 Starting GCP Cloud Run Deployment Helper"
echo "==============================================="

# Check if gcloud CLI is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Step 1: Authentication
echo -e "\n🔑 Step 1: Authenticating with Google Cloud..."
gcloud auth login

# Step 2: Configure Project
echo -e "\n⚙️ Step 2: Configuring active project to '$PROJECT_ID'..."
gcloud config set project "$PROJECT_ID"

# Step 3: Enable APIs
echo -e "\n🚀 Step 3: Enabling required GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    builds.googleapis.com \
    artifactregistry.googleapis.com

# Step 4: Create Artifact Registry Repository (if not exists)
echo -e "\n📦 Step 4: Creating Artifact Registry Docker repository..."
# Check if repository already exists
if gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" &>/dev/null; then
    echo "   Artifact repository '$REPO_NAME' already exists. Skipping creation."
else
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="Docker repository for Wine Quality Streamlit app"
    echo "   Artifact repository created successfully."
fi

# Step 5: Build and Push Container via Google Cloud Build
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"
echo -e "\n🏗️ Step 5: Building container image in the cloud using Cloud Build..."
echo "   Tag: $IMAGE_TAG"
gcloud builds submit --tag "$IMAGE_TAG" .

# Step 6: Deploy to Cloud Run
echo -e "\n🌐 Step 6: Deploying container image to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_TAG" \
    --platform managed \
    --region "$REGION" \
    --allow-unauthenticated

echo -e "\n==============================================="
echo "🎉 Deployment Complete!"
echo "   Your Streamlit app is now live on Google Cloud."
echo "==============================================="
