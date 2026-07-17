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
git clone <repository-url>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Dashboard

```bash
streamlit run app/app.py
```

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