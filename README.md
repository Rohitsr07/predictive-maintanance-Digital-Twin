# Predictive Maintenance Digital Twin using NASA C-MAPSS

An AI-powered Predictive Maintenance Digital Twin developed using the NASA C-MAPSS FD001 turbofan engine dataset. The project predicts the Remaining Useful Life (RUL) of aircraft engines using Machine Learning and provides an interactive Streamlit dashboard for monitoring engine health, explainability, fleet management, business intelligence, and predictive maintenance.

---

## Key Highlights

- Real NASA C-MAPSS FD001 Dataset
- XGBoost Regression Model
- GroupKFold Cross Validation (Leakage-Free)
- Clipped RUL Labeling
- Sensors-only Ablation Study
- SHAP Explainable AI
- Interactive Digital Twin Dashboard
- Fleet Management System
- Business Intelligence Dashboard
- What-If Simulator
- PDF & CSV Report Export

---

## Project Overview

Unexpected equipment failures can lead to significant downtime, safety risks, and maintenance costs. This project develops an AI-powered Digital Twin capable of predicting the Remaining Useful Life (RUL) of aircraft engines using historical sensor telemetry from the NASA C-MAPSS FD001 dataset.

The project covers the complete predictive maintenance pipeline—from data engineering and machine learning to explainable AI and an interactive digital twin dashboard for real-time visualization and decision support.

---

## Dataset

**Dataset:** NASA C-MAPSS FD001

Files used:

- train_FD001.txt
- test_FD001.txt
- RUL_FD001.txt

Dataset contains:

- 100 Training Engines
- 100 Testing Engines
- 21 Sensor Measurements
- 3 Operational Settings
- Complete Engine Run-to-Failure Data

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- XGBoost
- Scikit-learn
- SHAP
- Plotly
- Matplotlib
- ReportLab

---

## Project Structure

```text
Industrial_Digital_Twin/

│
├── app/
│   └── app.py
│
├── notebook/
│   ├── Digital_Twin.ipynb
│   └── Digital_Twin_cleaned.ipynb
│
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── artifacts/
│   ├── xgb_rul_model.pkl
│   ├── train_processed.csv
│   ├── digital_twin_snapshot.csv
│   ├── test_predictions.csv
│   └── feature_importance.csv
│
├── requirements.txt
└── README.md
```

---

## Machine Learning Pipeline

1. Load NASA C-MAPSS FD001 Dataset
2. Assign meaningful column names
3. Remove constant features
4. Handle missing values (if present)
5. Calculate Remaining Useful Life (RUL)
6. Apply Clipped RUL Labeling
7. Select predictive sensor features
8. Train XGBoost Regressor
9. Validate using GroupKFold Cross Validation
10. Perform Sensors-only Ablation Study
11. Generate SHAP Explainability
12. Export trained model

---

## Model Validation

Validation Strategy:

- GroupKFold Cross Validation
- Engine-wise Split
- Leakage-Free Evaluation

Performance Metrics:

- Mean Absolute Error (MAE)
- R² Score

Additional Experiments:

- Feature Importance Analysis
- Sensors-only Ablation Study
- SHAP Explainability

---

## Dashboard Features

### Home

- Fleet Overview
- KPIs
- Fleet Health Distribution

### Data Engineering

- Raw Dataset Preview
- Cleaned Dataset
- Missing Value Summary
- Feature Engineering Pipeline
- RUL Formula
- Data Flow Diagram

### Exploratory Data Analysis

- Dataset Statistics
- RUL Distribution
- Correlation Heatmap
- Sensor Trend Analysis
- Sensor vs RUL Visualization

### Model Performance

- MAE
- R² Score
- Feature Importance
- Actual vs Predicted
- Residual Plot
- Error Distribution

### Explainable AI

- SHAP Waterfall Plot
- SHAP Beeswarm Plot
- Top Feature Contributions

### Digital Twin

- Predicted RUL
- Health Index
- Failure Risk
- Maintenance Recommendation
- Live Sensor Readings
- RUL Trend

### Fleet Management

- Fleet Risk Heatmap
- Maintenance Priority Queue
- Fleet Statistics

### Business Intelligence

- Failure Cost Estimation
- Scheduled Maintenance Cost
- Estimated Savings
- Cost Comparison

### What-If Simulator

Interactively modify sensor values to observe:

- Predicted RUL
- Health Index
- Failure Risk

### Export

- CSV Report
- PDF Report

---

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Rohitsr07/predictive-maintanance-Digital-Twin.git
cd predictive-maintanance-Digital-Twin
```

### 2. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 3. Launch Dashboard

```bash
streamlit run app/app.py
```

The dashboard opens at http://localhost:8501.

---

## Docker

```bash
docker build -t pdm-digital-twin .
docker run -p 8501:8501 pdm-digital-twin
```

The image contains only the app and the trained artifacts, so no training runs at startup.

---

## Deployment (Streamlit Community Cloud)

1. Sign in at https://share.streamlit.io with GitHub.
2. Click **Create app** and select this repository, branch `main`, main file `app/app.py`.
3. Under **Advanced settings**, choose Python **3.12**.
4. Click **Deploy**.

---

## Maintenance Updates

- Re-serialized the XGBoost model with XGBoost 3.x, removing the version-mismatch warning on load (predictions unchanged).
- Migrated deprecated `use_container_width` arguments to `width="stretch"` for current Streamlit releases.
- Added a Dockerfile, `.dockerignore`, and Streamlit server config for containerized deployment.
- Cleared notebook outputs to keep the repository free of machine-specific paths.

---

## Project Workflow

```text
NASA C-MAPSS Dataset
          │
          ▼
Data Engineering
          │
          ▼
Feature Engineering
          │
          ▼
RUL Calculation
          │
          ▼
Clipped RUL Labeling
          │
          ▼
XGBoost Training
          │
          ▼
GroupKFold Validation
          │
          ▼
SHAP Explainability
          │
          ▼
Model Export
          │
          ▼
Interactive Streamlit Dashboard
```

---

## Future Scope

- Live IoT Sensor Integration
- Real-Time Digital Twin Updates
- Predictive Maintenance Scheduling
- Cloud Deployment
- Edge AI Deployment
- REST API Integration
- Multi-Fleet Monitoring

---

## Author

**Kashisa Padhy**

Atharva College of Engineering

AI-Powered Predictive Maintenance Digital Twin using NASA C-MAPSS FD001