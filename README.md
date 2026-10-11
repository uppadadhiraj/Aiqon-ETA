# Predicting Shipment ETA and Delay Propagation Using AI

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **College-Level AI Internship Assignment for Aiqon**  
> **Author:** Venkata Dhiraj Reddy Uppada  
> **Focus:** Supply Chain Machine Learning, Multi-Stage Delay Propagation, and Explainable AI

---

## 📌 1. Project Overview

In international freight logistics, an **Estimated Time of Arrival (ETA)** is rarely static. Delays rarely occur in isolation—an initial departure holdup, adverse ocean weather, container terminal congestion, or customs inspections trigger a **cascading domino effect** downstream, altering the final delivery schedule.

This project implements an end-to-end machine learning pipeline to:
1. **Predict final arrival delays (in hours)** and continuous final shipment ETAs.
2. **Benchmark machine learning** against an operational median baseline.
3. **Capture delay propagation dynamics** across multi-stage international freight journeys.
4. **Explain top risk factors** influencing shipping disruptions using feature importance.

---

## 📁 2. Repository Structure

```text
Aiqon-ETA/
├── data/
│   └── shipments.csv                               # 300 simulated shipment records (20 attributes)
├── notebooks/
│   └── eta_analysis.ipynb                          # Complete EDA, preprocessing, model training & plots
├── reports/
│   ├── model_predictions.csv                       # Test set actual vs. predicted delay with absolute errors
│   ├── predicted_etas.csv                          # Scheduled vs. actual vs. predicted final arrival timestamps
│   ├── feature_importance.csv                      # Ranked Random Forest feature importance scores
│   ├── model_evaluation.txt                        # Benchmark summary (MAE, RMSE, baseline comparisons)
│   ├── Aiqon_Shipment_ETA_Technical_Report_Revised.pdf # Formatted technical report
│   └── Aiqon_Shipment_ETA_Technical_Report.pdf     # Original technical report
├── generate_dataset.py                             # Multi-stage physical simulation dataset generator
├── README.md                                       # Project documentation and guide
├── requirements.txt                                # Python dependencies
└── .gitignore                                      # Standard Git exclusions for Python/Jupyter
```

---

## 🚢 3. Dataset & Simulation Engine

Because proprietary end-to-end container tracking telemetry is often private or fragmented across carriers, this project utilizes a deterministic, physically grounded **multi-stage simulation engine** (`generate_dataset.py`, Random Seed `42`).

### Corridors & Transport Modes
- **Routes:** 12 realistic international trade corridors connecting 5 major global logistics hubs:
  - **Shanghai (CNSHA)**, **Singapore (SGSIN)**, **Rotterdam (NLRTM)**, **Mumbai (INBOM)**, and **Los Angeles (USLAX)**.
- **Modes:** Ocean Container Freight (~70%), Air Express Cargo (~20%), and Rail-Sea Multimodal Freight (~10%).

### Multi-Stage Delay Propagation
Rather than assigning independent random delays at the destination, delays propagate sequentially through four operational gates:
```text
[ Origin Departure ] ──(+ departure delay)──>
      │
      ▼
[ In-Transit (Ocean/Air) ] ──(+ transit duration & weather disruptions)──>
      │
      ▼
[ Port Berthing & Customs ] ──(+ terminal congestion & customs workload)──>
      │
      ▼
[ Last-Mile Inland Drayage ] ──> Final Delivery Delay
```

### Real-World Data Realism
- **Missing Telemetry:** Injects ~7% missing intermediate milestone timestamps to represent incomplete tracking signals common in global supply chains.
- **SLA Delay Classification:** Classifies shipments exceeding a 6-hour delay threshold as `Delayed` (115 shipments, 38.3%) versus `On-Time` (185 shipments, 61.7%).

---

## ⚙️ 4. Machine Learning Methodology

### Feature Selection
To avoid target leakage, predictions rely strictly on information available at or before departure:
- Categorical features: `origin`, `destination`, `transport_mode`, `port_congestion_level`, `weather_severity`, `customs_workload`.
- Engineered numerical features:
  - `journey_hours`: Planned duration between scheduled departure and scheduled delivery.
  - `departure_month`: Captures seasonal shipping variations.

### Preprocessing & Pipeline
1. **One-Hot Encoding:** Categorical attributes converted via dummy variables.
2. **Train/Test Split:** 80% training (240 shipments) and 20% testing (60 shipments) with a fixed seed (`random_state=42`).
3. **Leak-Free Imputation:** Missing feature values in test data are imputed strictly using **training set medians**.
4. **Models Evaluated:**
   - **Baseline:** `DummyRegressor` predicting the median training delay.
   - **Main Model:** `RandomForestRegressor` (100 estimators, `random_state=42`).

---

## 📊 5. Key Results & Performance

The Random Forest regressor demonstrated substantial predictive power over the operational median baseline:

| Metric | Median Baseline | Random Forest Regressor | Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **15.93 hours** | **8.01 hours** | **~49.7% Error Reduction** |
| **Root Mean Squared Error (RMSE)** | — | **12.37 hours** | — |

> **Key takeaway:** The Random Forest halved average error from ~16 hours down to ~8 hours. A higher RMSE relative to MAE indicates that a subset of extreme delays (e.g., severe typhoon reroutings) incurred larger residuals.

---

## 🔍 6. Interpretability & Delay Insights

### Top Predictive Features
Feature importance analysis revealed that environmental and corridor parameters strongly govern delay severity:

1. **`weather_severity_Severe` (~51.6%)**: By far the strongest driver; severe weather forces vessel slowdowns and protective anchorages.
2. **`journey_hours` (~11.1%)**: Longer transit corridors accumulate more operational variance and exposure.
3. **`transport_mode_Air` (~5.7%)**: Air freight operates on tight schedules where minor delays represent large relative shifts.
4. **`port_congestion_level_High` (~5.2%)**: Berth hold-ups at destination container terminals.
5. **`departure_month` (~3.4%)**: Seasonal cargo demand shifts.

### Stage-to-Stage Delay Propagation
Analysis of average delays across milestones:
- Average Departure Delay: **3.32 hours**
- Average Port Arrival Delay: **11.26 hours**
- Average Customs Clearance Delay: **8.14 hours**
- **Correlation with Final Delay:** Port-arrival delay showed a **0.91 correlation**, while customs delay had a **1.00 correlation** with final delivery delay, demonstrating the compounding nature of bottleneck stages.

---

## 🚀 7. Quickstart Guide

### Prerequisites
- Python 3.9 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/Aiqon-ETA.git
   cd Aiqon-ETA
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

- **Option A: Run the Jupyter Notebook**
  ```bash
  jupyter notebook notebooks/eta_analysis.ipynb
  ```
  *(Or open directly in VS Code / PyCharm with the Jupyter extension).*

- **Option B: Re-generate the Dataset (Optional)**
  ```bash
  python generate_dataset.py
  ```
  Generates `data/shipments.csv` deterministically.

---

## ⚠️ 8. Limitations and Future Scope

- **Synthetic Data:** While built upon physical logistics rules, synthetic simulation cannot capture unexpected real-world shocks (labor strikes, geopolitical canal closures, canal draught restrictions).
- **Dataset Size:** 300 records serve as a proof-of-concept. Real-world applications require millions of AIS transponder pings.
- **Future Directions:**
  - Incorporating live AIS transponder GPS telemetry.
  - Applying **Bayesian Networks** or **Directed Acyclic Graphs (DAGs)** to explicitly model causal delay propagation rather than correlation alone.
  - Adding quantile regression to provide confidence intervals for ETAs.

---

## 👨‍💻 9. Author & Acknowledgments

- **Author:** Venkata Dhiraj Reddy Uppada
- **Assignment:** Aiqon AI Internship Take-Home Assignment
- **Special Thanks:** Aiqon Team for the challenging and practical supply chain logistics problem statement.
