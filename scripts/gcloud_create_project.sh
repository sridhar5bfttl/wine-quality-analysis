#!/bin/bash

# ==============================================================================
# GCP Project Creation & Configuration Helper for Wine Workspace
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Generate a unique project ID using a random number suffix
RAND_ID=$((100000 + RANDOM % 900000))
PROJECT_ID="vintedge-wine-${RAND_ID}"

PROJECT_NAME="VintEdge Wine Workspace"

echo "==============================================="
echo "⚙️ Creating New Google Cloud Project"
echo "==============================================="
echo "Project Name: $PROJECT_NAME"
echo "Project ID:   $PROJECT_ID"
echo "==============================================="

# 1. Create the project
echo -e "\n🛠️ Step 1: Creating project '$PROJECT_ID'..."
gcloud projects create "$PROJECT_ID" --name="$PROJECT_NAME"

# 2. Set the configuration active project
echo -e "\n⚙️ Step 2: Setting active config project to '$PROJECT_ID'..."
gcloud config set project "$PROJECT_ID"

# 3. Check for billing accounts (Cloud Run & Cloud Build require billing)
echo -e "\n💳 Step 3: Checking available billing accounts..."
echo "Google Cloud Build and Cloud Run require billing to be enabled."
echo "Here are your available billing accounts:"
echo "--------------------------------------------------"
gcloud billing accounts list || echo "Could not list billing accounts. Please link billing manually."
echo "--------------------------------------------------"

echo -n "Please copy & paste your Billing Account ID from the table above (or press Enter to skip): "
read -r BILLING_ACCOUNT_ID

if [ -n "$BILLING_ACCOUNT_ID" ]; then
    echo "Linking project to billing account: $BILLING_ACCOUNT_ID..."
    gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT_ID"
else
    echo "⚠️ Warning: Skipping billing link. The subsequent API enablement might fail if billing is required."
fi

# 4. Enable required APIs with propagation delay safety
echo -e "\n🚀 Step 4: Enabling Cloud Build, Cloud Run, and Artifact Registry APIs..."
echo "Waiting 15 seconds for project IAM policies to propagate in GCP..."
sleep 15

# Retry loop for API enablement to handle GCP sync latency
MAX_RETRIES=3
RETRY_DELAY=10
for i in $(seq 1 $MAX_RETRIES); do
    if gcloud services enable run.googleapis.com builds.googleapis.com artifactregistry.googleapis.com; then
        echo "✅ APIs enabled successfully."
        break
    else
        if [ $i -lt $MAX_RETRIES ]; then
            echo "⚠️ GCP IAM propagation delay encountered. Retrying in $RETRY_DELAY seconds (Attempt $i/$MAX_RETRIES)..."
            sleep $RETRY_DELAY
        else
            echo "❌ Failed to enable APIs after $MAX_RETRIES attempts. Please check permissions."
            exit 1
        fi
    fi
done

# 5. Automatically update gcloud_deploy.sh
echo -e "\n📝 Step 5: Updating PROJECT_ID in scripts/gcloud_deploy.sh..."
DEPLOY_SCRIPT="scripts/gcloud_deploy.sh"
if [ -f "$DEPLOY_SCRIPT" ]; then
    # Use macOS sed compat style (with empty string backup argument)
    sed -i '' "s/PROJECT_ID=\".*\"/PROJECT_ID=\"$PROJECT_ID\"/g" "$DEPLOY_SCRIPT"
    echo "✅ Successfully updated $DEPLOY_SCRIPT to use PROJECT_ID=\"$PROJECT_ID\""
else
    echo "❌ Error: Could not find $DEPLOY_SCRIPT"
fi

echo -e "\n==============================================="
echo "🎉 Project configured successfully!"
echo "   Now you can run the deployment script:"
echo "   ./scripts/gcloud_deploy.sh"
echo "==============================================="
