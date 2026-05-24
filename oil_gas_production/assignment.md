# Upstream Oil & Gas Data Science Assignment
## Objective: Virtual Flow Metering & Decline Curve Analytics

### 1. Business Context & Problem Statement
In upstream oil and gas operations, knowing the exact amount of oil, gas, and water flowing out of a well daily is critical for reservoir management and financial accounting. Historically, this is measured using a physical **Multiphase Flow Meter (MPFM)** installed at the wellhead. 

**The Scenario:**
The physical flow meter on your platform has failed due to sand erosion. Procuring and replacing this physical meter requires a production shutdown costing the company **$150,000/day**. 

**Your Task:**
Rather than shutting down the well, you are tasked with building a data-driven **Virtual Flow Meter (VFM)**. You must design a machine learning system that utilizes existing pressure and temperature sensor logs to predict the daily oil flow rate (`oil_volume_bbl`), analyze the well's depletion trend, and deploy an interactive monitoring dashboard.

---

### 2. The Dataset (`well_production_data.csv`)
You are provided with 3 years of daily sensor and production logs containing the following fields:

| Column Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `date` | Date | YYYY-MM-DD | Log timestamp |
| `on_stream_hours` | Float | Hours | Daily uptime of the well (0 to 24 hrs) |
| `choke_size_percentage` | Float | % | Choke valve opening size (0% = closed, 100% = wide open) |
| `bottomhole_pressure_psi` | Float | PSI | Sensor reading at the bottom of the well (reservoir level) |
| `bottomhole_temperature_f` | Float | °F | Temperature at the reservoir level |
| `wellhead_pressure_psi` | Float | PSI | Surface pressure sensor reading (upstream of the choke) |
| `wellhead_temperature_f` | Float | °F | Surface temperature sensor reading (upstream of the choke) |
| `oil_volume_bbl` | Float | bbl/day | **Target Variable:** Volume of oil produced |
| `gas_volume_mscf` | Float | mscf/day | Volume of associated natural gas produced |
| `water_volume_bbl` | Float | bbl/day | Volume of reservoir water produced |

---

### 3. Assignment Milestones & Deliverables

#### 🗺️ Milestone 1: Exploratory Data Analysis (EDA) & Physics Verification
*   **Decline Plotting:** Plot daily Oil, Gas, and Water volumes over the 3-year timeline. Describe the overall production profile (e.g. what happens to the oil rate over time? What happens to the water rate?).
*   **Sensor Correlations:** Compute a correlation matrix heatmap. Explain the physical relationship between:
    1.  `choke_size_percentage` vs. `bottomhole_pressure_psi` (drawdown).
    2.  `wellhead_temperature_f` vs. `oil_volume_bbl` (why does higher temperature correlate with higher flow?).
*   **Outage Analysis:** Identify days where the well was shut down. Describe how you will handle these records in training (Hint: predicting flow during a planned shutdown is trivial and should not bias your regression model).

#### ⚙️ Milestone 2: Build the Virtual Flow Meter (Regression Model)
*   **Data Preparation:** Filter out inactive days. Perform an 80/20 train/test split. Explain the difference and risk between a *Random Split* and a *Time-Series Split* for this dataset.
*   **Model Training:** Train a regressor (e.g. Linear Regression, Random Forest, or XGBoost) to predict `oil_volume_bbl` using the operational and sensor columns.
*   **Evaluation:** Report $R^2$, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) on the test set.
*   **Feature Importance:** Extract and plot feature importances. Do the model's top predictors match physical expectations?

#### 📉 Milestone 3: Decline Curve & Lifecycle Forecasting
*   **Water Cut Diagnostics:** Calculate the daily Water Cut percentage: 
    $$\text{Water Cut} = \frac{\text{Water Volume}}{\text{Oil Volume} + \text{Water Volume}} \times 100$$
    Plot this over time. Identify when the well experiences "Water Breakthrough" (rapid water coning).
*   **Economic Limit Forecasting:** Assuming the well is decommissioned when the oil flow rate drops below **200 bbl/day**, fit an exponential decline curve to forecast the exact month and year this well will reach its economic limit.

#### 🖥️ Milestone 4: Interactive Dashboard Deployment (Streamlit)
Build a Streamlit web application that lets production engineers interact with your models:
*   **Metrics View:** Display current flow rate, cumulative production, and water cut.
*   **VFM Inference Panel:** Create sliders for `on_stream_hours`, `choke_size_percentage`, pressures, and temperatures. When an engineer adjusts the sliders, pass the inputs to your trained model and show the predicted daily oil rate in real-time.
*   **Visualizations:** Embed interactive Plotly charts showing the decline curves and sensor correlations.

---

### 4. Submission Requirements
Your freshers should submit:
1.  A clean, documented Jupyter Notebook (`oil_gas_analysis.ipynb`) answering the EDA and modeling questions.
2.  A python script containing their trained model serializer (`train.py` resulting in `model.pkl`).
3.  A Streamlit application script (`app.py`).
