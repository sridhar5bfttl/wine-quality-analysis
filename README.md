# VintEdge Wine Quality Workspace

An interactive machine learning workspace analyzing the physicochemical properties of red and white wines. The project contains a model training pipeline (Random Forest Classifier) and a premium Streamlit visualization dashboard showcasing an interactive "Virtual Sommelier" and chemical analytics.

---

## 📁 Workspace Structure

*   `app.py`: Main Streamlit application with custom glassmorphic styling, analytics plots, and prediction dashboards.
*   `requirements.txt`: Python package dependencies.
*   `data/`: Directory containing generated CSV datasets, trained classification models, and feature metadata.
*   `notebooks/`: Contains `eda_analysis.ipynb` for interactive data analysis, correlation testing, and model prototyping.
*   `scripts/`: Automation utilities:
    *   `download_data.py`: Sourcing helper with statistically accurate synthetic data generation fallback.
    *   `train_model.py`: Random Forest model training and evaluation script.
    *   `verify_app_logic.py`: Logic verification script to programmatically validate datasets and inference.

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

3.  **Train the Classification Models:**
    ```bash
    python3 scripts/train_model.py
    ```

4.  **Run the Streamlit App:**
    ```bash
    python3 -m streamlit run app.py
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
