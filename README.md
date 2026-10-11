# Predicting Shipment ETA and Delay Propagation Using AI

This repository contains my take-home assignment for the Aiqon AI Internship.

The goal of this project is to predict how many hours a shipment might be delayed and analyze how delays at earlier stages (departure, port arrival, customs) carry over into the final delivery time.

## Project Overview

In shipping logistics, when a delay happens early on (like at departure or due to bad weather), it often causes delays later at the port or during customs clearance. 

For this assignment, I:
1. Created a synthetic dataset of 300 shipments across different international routes (Shanghai, Singapore, Rotterdam, Mumbai, Los Angeles).
2. Cleaned and preprocessed the data (handled missing tracking timestamps, calculated journey hours, and one-hot encoded categories).
3. Trained a Random Forest Regressor to predict `final_delay_hours`.
4. Compared the model against a baseline that predicts the median delay.
5. Checked feature importances and looked at how delays at each stage correlate with the final delay.

## Project Structure

```
Aiqon-ETA/
├── data/
│   └── shipments.csv           # 300 synthetic shipment records
├── notebooks/
│   └── eta_analysis.ipynb      # Jupyter notebook with data analysis & model
├── reports/
│   ├── model_predictions.csv   # Actual vs predicted delay on test data
│   ├── predicted_etas.csv      # Actual vs predicted arrival times
│   ├── feature_importance.csv  # Feature importance scores from Random Forest
│   ├── model_evaluation.txt    # Summary of error metrics
│   ├── Aiqon_Shipment_ETA_Technical_Report_Revised.pdf # Technical report
│   └── Aiqon_Shipment_ETA_Technical_Report.pdf         # ReportLab copy
├── generate_dataset.py         # Python script used to generate the dataset
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

Because real global shipping data with complete milestone tracking is hard to get, I generated 300 simulated shipments using `generate_dataset.py` with random seed `42`.

Key details:
- **Routes:** Connects 5 major hubs: Shanghai, Singapore, Rotterdam, Mumbai, and Los Angeles.
- **Transport Modes:** Ocean, Air, and Multimodal.
- **Stages:** Tracks departure, transit, port arrival, customs clearance, and delivery.
- **Conditions:** Includes port congestion (Low/Medium/High), weather severity (Mild/Moderate/Severe), and customs workload.
- **Missing Data:** About 7% of intermediate tracking timestamps are missing to represent real-world tracking gaps.
- **Delay status:** Shipments with more than 6 hours of delay are marked as "Delayed" (115 delayed, 185 on-time).

## Model & Results

I split the data into 80% training (240 shipments) and 20% testing (60 shipments) using `random_state=42`. Missing values in the test set were filled using medians from the training set.

I trained a **Random Forest Regressor** (100 estimators) and compared it with a simple **Median Baseline** (`DummyRegressor`):

| Model | MAE | RMSE | Notes |
|---|---|---|---|
| Median Baseline | 15.93 hours | - | Predicts the median delay from training data |
| Random Forest | 8.01 hours | 12.37 hours | Reduces average error by ~49.7% |

### Key Findings
- **Severe weather** had the highest feature importance (~0.516).
- **Planned journey hours** (~0.111), **air transport** (~0.057), and **high port congestion** (~0.052) were the next most influential features.
- Average delays were 3.32 hours at departure, 11.26 hours at port arrival, and 8.14 hours at customs.
- Port-arrival delay had a correlation of 0.91 with final delay, and customs delay had a correlation of 1.00 with final delay, showing how delays compound downstream.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/uppadadhiraj/Aiqon-ETA.git
   cd Aiqon-ETA
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook notebooks/eta_analysis.ipynb
   ```

4. (Optional) Re-generate the dataset:
   ```bash
   python generate_dataset.py
   ```

## Limitations & Next Steps

- The dataset is simulated, so the model learns patterns from the simulation rules rather than real-world supply chain behavior.
- 300 rows is a small sample size.
- Feature importance shows what the model relied on, but does not prove direct causation.
- A natural next step would be testing this workflow on historical shipment data (like AIS vessel tracking).

## Author
Venkata Dhiraj Reddy Uppada  
AI Internship Assignment for Aiqon
