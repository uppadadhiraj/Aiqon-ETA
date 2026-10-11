# Data Generation Notes: Predicting Shipment ETA & Delay Propagation

**Project Title:** Predicting Shipment ETA and Delay Propagation Using AI  
**Dataset Artifact:** `data/shipments.csv` (300 records × 20 features)  
**Generation Script:** `generate_dataset.py`  
**Target Domain:** Global Container Shipping, Air Cargo, and Multimodal Freight Logistics  
**Random Seed:** `42` (Fixed for 100% deterministic reproducibility)

---

## 1. Overview and Objective

The objective of this synthetic dataset is to support the training, validation, and evaluation of machine learning models designed to solve two core logistical challenges:
1. **Regression Task:** Predicting the final arrival delay in hours (`final_delay_hours`) and continuous Estimated Time of Arrival (ETA).
2. **Classification Task:** Classifying shipment delay status (`delay_status`: `On-Time` vs. `Delayed` under an operational SLA threshold of 6 hours).

Unlike simplistic random data generation where features are sampled independently, this dataset is generated through a **multi-stage physical simulation engine**. Delays propagate naturally downstream from departure through maritime/air transit, port congestion, customs clearance, and inland drayage, realistically reflecting supply chain bottleneck dynamics.

---

## 2. Core Assumptions and Operational Parameters

### 2.1 Route Network & Hubs
The dataset samples from plausible international trade corridors linking five of the world's most critical maritime container ports and cargo hubs:
- **Shanghai (CNSHA):** World's busiest container seaport and major East Asian manufacturing hub.
- **Singapore (SGSIN):** Premier global transshipment hub connecting Asia, Europe, and the Middle East.
- **Rotterdam (NLRTM):** Europe's largest container port and gateway to continental European inland corridors.
- **Mumbai (INBOM / JNPT):** Major Indian subcontinent industrial export hub.
- **Los Angeles (USLAX):** Primary North American Pacific Rim maritime gateway.

### 2.2 Transport Modes and Transit Velocities
- **Ocean (Containerized Maritime Freight):** Represents ~70% of shipments. Typical scheduled transit durations range from 168 hours (7 days, Singapore–Mumbai) up to 740 hours (~31 days, Shanghai–Rotterdam).
- **Air (Air Cargo Express):** Represents ~20% of shipments. Scheduled transit durations range from 16 to 32 hours between major international cargo airports.
- **Multimodal (Rail-Sea Corridors):** Represents ~10% of shipments (e.g., Eurasia rail-sea intermodal corridors), with transit durations from 380 to 400 hours (~16 days).

### 2.3 Operational Buffers (Schedule Slack)
In real-world logistics, carriers build **schedule padding** (slack) into published timetables to absorb minor operational friction. The simulation reflects this by budgeting standard customs and terminal turnaround times (24–28 hours for Ocean, 12–14 hours for Air). Under optimal conditions (Low congestion, Low customs workload), operations finish ahead of schedule, allowing minor preceding delays to be recovered.

---

## 3. How Delays Propagate Across Shipment Stages

A critical design requirement of this dataset is that **downstream timestamps are strictly dependent on upstream actual milestones**. Delays are not assigned independently at the final stage.

The journey progresses sequentially through four interconnected operational gates:

```
[ Scheduled Plan ] -------------------------------------------------------------+
   sched_departure  --> sched_port_arrival  --> sched_customs_comp --> sched_delivery
          |                       |                       |                  |
[ Stage 1: Departure ]            |                       |                  |
   act_departure = sched_departure + departure_delay      |                  |
          |                       |                       |                  |
          v                       |                       |                  |
[ Stage 2: Transit ]              v                       |                  |
   act_port_arrival = act_departure + transit_duration(weather)              |
          |                                               |                  |
          v                                               v                  |
[ Stage 3: Port & Customs ]                               |                  |
   act_customs_comp = act_port_arrival + port_wait(congestion) + customs(workload)
          |                                                                  |
          v                                                                  v
[ Stage 4: Last-Mile Delivery ]                                              |
   act_delivery = act_customs_comp + delivery_duration                       |
          |                                                                  |
          +--------------------> final_delay_hours = act_delivery - sched_delivery
```

### Stage 1: Origin Departure
- **Formula:** `actual_departure = scheduled_departure + timedelta(hours=departure_delay_hours)`
- **Mechanism:**
  - *Early departure (-2.5h to -0.2h, ~20%):* Early completion of container stevedoring or expedited terminal clearance.
  - *On-time (-0.2h to 1.5h, ~50%):* Standard departure within terminal schedule window.
  - *Moderate delay (2.0h to 10.0h, ~20%):* Gate-in congestion, documentation discrepancies, feeder vessel delay.
  - *Severe delay (12.0h to 36.0h, ~10%):* Mechanical breakdown, vessel blank sailing, extreme berth queue at origin.

### Stage 2: In-Transit to Destination Port
- **Formula:** `actual_port_arrival = actual_departure + timedelta(hours=actual_transit_hours)`
- **Propagation Impact:** Any departure delay is carried forward directly into `actual_port_arrival`.
- **Environmental Modifier (`weather_severity`):**
  - *Mild (68%):* Sailing/flight operates at normal cruising speed. Slight speed variation (-1.5% to +0.8%).
  - *Moderate (22%):* Strong sea states or adverse winds require slow steaming (+1% to +3% duration) plus minor weather diversions (+5h to +16h for Ocean; +1.5h to +4h for Air).
  - *Severe (10%):* Severe storms/typhoons force vessel re-routing or protective anchorage (+3% to +8% speed loss + 24h to 64h storm delay for Ocean; +6h to +18h for Air).

### Stage 3: Port Berthing & Customs Processing
- **Formula:** `actual_customs_completion = actual_port_arrival + timedelta(hours=actual_port_wait + actual_customs_processing)`
- **Propagation Impact:** The start of port operations depends strictly on `actual_port_arrival`. An arrival that is late by 20 hours automatically delays the initiation of berthing and customs by 20 hours.
- **Physical Drivers:**
  - *Port Congestion (`port_congestion_level`):*
    - `Low`: Rapid berthing (2–5h Ocean, 1–2.5h Air).
    - `Medium`: Yard crane queues and berthing hold (7–16h Ocean, 2.5–6h Air).
    - `High`: Severe anchorage congestion (20–48h Ocean, 6–14h Air).
  - *Customs Workload (`customs_workload`):*
    - `Low`: Automated green channel clearance (4–10h Ocean, 3–6h Air).
    - `Medium`: Standard document inspection and cargo screening (10–18h Ocean, 6–11h Air).
    - `High`: Red channel physical container devanning and manual auditing (20–40h Ocean, 12–22h Air).

### Stage 4: Last-Mile Delivery
- **Formula:** `actual_delivery = actual_customs_completion + timedelta(hours=actual_delivery_hours)`
- **Propagation Impact:** Accumulates the total composite delay from all three prior stages. Last-mile drayage includes stochastic road traffic and gate appointment variance ($\sim \mathcal{N}(0, 2\text{ hours})$).

---

## 4. Field Categorization: Simulated vs. Calculated

| Column Name | Category | Type | Description / Generating Mechanism |
| :--- | :--- | :--- | :--- |
| `shipment_id` | Metadata | String | Unique shipment identifier (`SHP-0001` through `SHP-0300`). |
| `origin` | Feature (Static) | Categorical | Origin departure port/hub (e.g., Shanghai, Singapore, Mumbai). |
| `destination` | Feature (Static) | Categorical | Destination arrival port/hub (e.g., Los Angeles, Rotterdam). |
| `transport_mode` | Feature (Static) | Categorical | Primary mode of transit: `Ocean`, `Air`, or `Multimodal`. |
| `scheduled_departure` | Feature (Schedule) | Timestamp | Baseline contractual departure slot (ISO-8601). |
| `actual_departure` | Feature (Dynamic Tracking) | Timestamp | Simulated actual operational departure. Contains 7% missing telemetry. |
| `scheduled_port_arrival` | Feature (Schedule) | Timestamp | Baseline planned port arrival based on scheduled transit speed. |
| `actual_port_arrival` | Feature (Dynamic Tracking) | Timestamp | Simulated actual port arrival. Contains 7% missing telemetry. |
| `scheduled_customs_completion` | Feature (Schedule) | Timestamp | Contractual customs clearance target timestamp. |
| `actual_customs_completion` | Feature (Dynamic Tracking) | Timestamp | Simulated actual clearance completion. Contains 7% missing telemetry. |
| `scheduled_delivery` | Feature (Schedule) | Timestamp | Contractual destination consignee delivery SLA timestamp. |
| `actual_delivery` | Ground Truth (Target) | Timestamp | **Ground truth** actual delivery timestamp at customer facility (100% complete). |
| `port_congestion_level` | Feature (Contextual) | Categorical | Environmental indicator: `Low`, `Medium`, or `High`. |
| `weather_severity` | Feature (Contextual) | Categorical | Voyage meteorological condition: `Mild`, `Moderate`, or `Severe`. |
| `customs_workload` | Feature (Contextual) | Categorical | Destination customs inspection queue: `Low`, `Medium`, or `High`. |
| `departure_delay_hours` | **Calculated Metric** | Float | `(actual_departure - scheduled_departure)` in decimal hours. |
| `port_arrival_delay_hours` | **Calculated Metric** | Float | `(actual_port_arrival - scheduled_port_arrival)` in decimal hours. |
| `customs_delay_hours` | **Calculated Metric** | Float | `(actual_customs_completion - scheduled_customs_completion)` in decimal hours. |
| `final_delay_hours` | **Calculated Target (Regression)** | Float | `(actual_delivery - scheduled_delivery)` in decimal hours. 100% complete. |
| `delay_status` | **Calculated Target (Classification)**| Categorical | `Delayed` if `final_delay_hours > 6.0`, else `On-Time`. 100% complete. |

---

## 5. Incomplete Tracking Simulation (Missing Telemetry)

In practical supply chain visibility platforms, Internet of Things (IoT) sensors, Automatic Identification System (AIS) transponders, and Electronic Data Interchange (EDI) feeds frequently experience temporary blackouts (e.g., loss of satellite AIS in deep waters, customs API downtime, or un-scanned container milestones).

To replicate this real-world phenomenon:
- **7.0% (21/300 records)** of intermediate actual milestones (`actual_departure`, `actual_port_arrival`, `actual_customs_completion`) are masked as missing (`NaN` / `NaT`).
- Because stage delay metrics are calculated as `actual - scheduled`, when an intermediate actual milestone is unrecorded, its corresponding observed stage delay (`departure_delay_hours`, etc.) is also unobserved (`NaN`), maintaining mathematical consistency.
- **Ground Truth Target Preservation:** All `scheduled_*` timestamps, environmental context indicators, and the final targets (`actual_delivery`, `final_delay_hours`, `delay_status`) remain **100% complete** (0 missing values), enabling clean loss calculation and benchmark evaluation for predictive AI models.

---

## 6. Ground Truth Definition and Delay Distribution

### 6.1 Ground Truth Formulae
$$\text{final\_delay\_hours} = \frac{\text{actual\_delivery} - \text{scheduled\_delivery}}{3600 \text{ seconds}}$$

$$\text{delay\_status} = \begin{cases} \text{"Delayed"}, & \text{if } \text{final\_delay\_hours} > 6.0 \\ \text{"On-Time"}, & \text{otherwise} \end{cases}$$

### 6.2 Empirical Distribution in the Generated Dataset
Based on the fixed seed (`RANDOM_SEED = 42`):
- **On-Time Shipments:** 185 shipments (**61.67%**)
- **Delayed Shipments:** 115 shipments (**38.33%**)
  - *Moderately Delayed (6h to 36h):* ~26% of total records (due to adverse weather or medium port queue).
  - *Severely Delayed (> 36h to 113h):* ~12% of total records (multi-stage compounded disruption: departure delay + severe storms + high congestion + high customs inspection).

This distribution closely aligns with real-world industry benchmarks reported by maritime research organizations (e.g., Sea-Intelligence Global Liner Performance reports), avoiding artificial 50/50 synthetic distributions or pathological class imbalances.

---

## 7. Machine Learning Modeling Recommendations & Preventing Data Leakage

When building an AI model using this dataset:
1. **Target Variables:**
   - Regression: `final_delay_hours` (or directly predicting `actual_delivery`).
   - Classification: `delay_status`.
2. **Preventing Target Leakage:**
   - **Never** include `final_delay_hours` or `actual_delivery` as input features when predicting ETA or delay status.
   - For an **Early Stage Prediction Model** (at departure), use only `origin`, `destination`, `transport_mode`, `scheduled_*`, `actual_departure` (if available), and environmental forecasts.
   - For a **Dynamic In-Transit Update Model** (re-predicting ETA after port arrival), use upstream stage delays as sequential features to evaluate how well the model captures delay propagation.
3. **Handling Missing Telemetry:**
   - Implement imputation strategies (e.g., KNN imputation, iterative imputer, or missing indicator flags) to handle the 7% unrecorded intermediate milestones without dropping rows.

---

## 8. Limitations and Synthetic Data Disclaimer

> [!WARNING]
> **Synthetic Data Disclaimer:**  
> This dataset was generated synthetically using mathematical and statistical simulation rules for educational and machine learning benchmark purposes. While calibrated to mirror realistic physical behaviors, transit times, and delay distributions of global container trade:
> - It **does not** represent verified real-world operational logs or proprietary telemetry from any commercial carrier (e.g., Maersk, MSC, CMA CGM) or port authority.
> - Actual ocean transit times are subject to geopolitical events (e.g., Red Sea rerouting around the Cape of Good Hope), seasonal blank sailings (e.g., Lunar New Year factory closures), fluctuating canal draft restrictions (e.g., Panama Canal droughts), and complex intermodal carrier alliances that are not modeled here.
> - This dataset should be used strictly for coursework, educational assignments, prototype model benchmarking, and algorithmic development.
