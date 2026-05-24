# Data Science Workspaces

This repository contains two interactive machine learning workspaces designed with premium dark-mode Streamlit visualization dashboards:

1.  **VintEdge Wine Quality Workspace** (Root directory): Analyzes the physicochemical properties of red and white wines, incorporating a Random Forest Classifier to serve as a "Virtual Sommelier".
2.  **PetroPulse Upstream Oil & Gas Workspace** (`oil_gas_production/`): Simulates daily operational telemetry of a production well, incorporating a Virtual Flow Metering Random Forest Regressor to predict oil flow rates from sensor data.

---

## 📁 Workspace Structure

### 🍷 VintEdge Wine Quality (Root)
*   `app.py`: Wine quality Streamlit application.
*   `requirements.txt`: Python dependencies.
*   `data/`: CSV datasets, trained classification models, and feature metadata.
*   `notebooks/eda_analysis.ipynb`: Jupyter notebook for wine data analysis.
*   `scripts/`: Automation scripts (`download_data.py`, `train_model.py`, `verify_app_logic.py`).

### 🛢️ PetroPulse Oil & Gas (`oil_gas_production/`)
*   `oil_gas_production/app.py`: Flow meter and decline analysis Streamlit dashboard.
*   `oil_gas_production/data/`: Simulated daily production CSV and flow meter models.
*   `oil_gas_production/notebooks/oil_gas_eda.ipynb`: Jupyter notebook for production time-series analysis.
*   `oil_gas_production/scripts/`:
    *   `generate_data.py`: Physically grounded well simulation engine.
    *   `train_forecaster.py`: Model training script for the regression pipeline.
    *   `verify_pipeline.py`: Pipeline automation tester.

---

## 💻 Local Setup & Execution

1.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **To run the Wine Quality Workspace:**
    ```bash
    python3 scripts/train_model.py
    python3 -m streamlit run app.py
    ```

4.  **To run the PetroPulse Oil & Gas Workspace:**
    ```bash
    python3 oil_gas_production/scripts/train_forecaster.py
    python3 -m streamlit run oil_gas_production/app.py
    ```

---

## ☁️ Manual Deployment to Google Cloud Run

To bypass command-line rate limits and local Docker requirements, follow these manual steps using the **Google Cloud Web Console** and **Cloud Shell**.

### Step 1: Create a Project & Link Billing
1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Click the **Project Dropdown** at the top left, select **New Project**, enter a name (e.g. `VintEdge Wine Workspace`), and click **Create**.
3. Copy the generated **Project ID** (e.g., `vintedge-wine-123456`).
4. Go to the [Billing Console](https://console.cloud.google.com/billing), select **Manage Projects**, click the options dots next to your new project, and select your billing account.

### Step 2: Enable Required APIs
Using the search bar at the top of the Google Cloud Console, search for and click **Enable** for each of these three APIs:
*   **Cloud Run API**
*   **Cloud Build API**
*   **Artifact Registry API**

### Step 3: Open Cloud Shell & Upload Files
1. Click the **Activate Cloud Shell** button (the `>_` terminal icon in the top-right header of the console).
2. Click the **Three Dots (More options)** icon in the Cloud Shell toolbar and click **Upload** $\rightarrow$ **Folder**.
3. Select your local `Datasets` project directory to upload it into the Cloud Shell container.

### Step 4: Run Manual Deployment Commands
In your Cloud Shell terminal, execute the following commands step-by-step (replacing `YOUR_PROJECT_ID` with the Project ID from Step 1):

```bash
# 1. Navigate to the uploaded directory
cd Datasets

# 2. Set the active Cloud Shell project
gcloud config set project YOUR_PROJECT_ID

# 3. Create the Artifact Registry Repository
gcloud artifacts repositories create wine-app-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for Wine app"

# 4. Compile the Docker image in the cloud using Cloud Build
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/wine-app-repo/wine-app:latest .

# 5. Deploy the image to Cloud Run
gcloud run deploy wine-quality-app \
    --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/wine-app-repo/wine-app:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

Once Step 5 completes, Cloud Shell will print a public HTTPS URL where your Streamlit app is live on Google Cloud.
Example: https://wine-quality-app-1000479190734.us-central1.run.app
