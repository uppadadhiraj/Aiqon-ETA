#!/usr/bin/env python3
"""
Shipment ETA and Delay Propagation Synthetic Dataset Generator

Assignment: Predicting Shipment ETA and Delay Propagation Using AI
Description:
    Generates a realistic, synthetically simulated supply chain dataset containing
    300 multimodal and ocean shipment records. Features multi-stage delay propagation,
    environmental disruption factors (weather, port congestion, customs workload),
    plausible international shipping routes, missing tracking telemetry (5-10%),
    and consistent ground-truth ETA delay calculations.

Requirements:
    - Python >= 3.9
    - pandas >= 2.0.0
    - numpy >= 1.24.0

Usage:
    python generate_dataset.py
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURATION & REPRODUCIBILITY
# ==============================================================================
RANDOM_SEED = 42
NUM_SHIPMENTS = 300
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "shipments.csv")

# Set fixed random seed for 100% reproducibility across runs
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Simulation date range: shipments departing across H1 2026
START_DATE_SIMULATION = datetime(2026, 1, 1, 6, 0, 0)
END_DATE_SIMULATION = datetime(2026, 5, 31, 23, 0, 0)

# Plausible international shipping corridors between global logistics hubs:
# Shanghai, Singapore, Rotterdam, Mumbai, Los Angeles
# Tuple format: (Origin, Destination, Mode, Sched Transit Hours, Sched Customs Hours, Sched Delivery Hours)
ROUTE_CATALOG = [
    ("Shanghai", "Los Angeles", "Ocean", 360.0, 24.0, 24.0),      # ~15 days Transpacific Ocean
    ("Shanghai", "Los Angeles", "Air", 28.0, 14.0, 12.0),         # ~1.2 days Air Cargo
    ("Shanghai", "Rotterdam", "Ocean", 720.0, 28.0, 24.0),        # ~30 days Asia-Europe Maritime
    ("Shanghai", "Rotterdam", "Multimodal", 380.0, 24.0, 24.0),   # ~16 days Eurasia Rail-Sea
    ("Shanghai", "Rotterdam", "Air", 32.0, 14.0, 12.0),           # Air Freight
    ("Shanghai", "Mumbai", "Ocean", 336.0, 24.0, 24.0),           # ~14 days East Asia-South Asia
    ("Shanghai", "Mumbai", "Air", 24.0, 12.0, 10.0),              # Direct Air
    ("Singapore", "Rotterdam", "Ocean", 552.0, 26.0, 24.0),       # ~23 days Maritime
    ("Singapore", "Rotterdam", "Air", 30.0, 14.0, 12.0),          # Air Transit
    ("Singapore", "Los Angeles", "Ocean", 456.0, 24.0, 24.0),     # ~19 days Transpacific
    ("Singapore", "Mumbai", "Ocean", 168.0, 20.0, 18.0),          # ~7 days South Asian Straits
    ("Singapore", "Mumbai", "Air", 16.0, 10.0, 10.0),             # Fast Air Cargo
    ("Mumbai", "Rotterdam", "Ocean", 528.0, 26.0, 24.0),          # ~22 days Maritime
    ("Mumbai", "Rotterdam", "Air", 26.0, 14.0, 12.0),             # Air Freight
    ("Mumbai", "Singapore", "Ocean", 180.0, 20.0, 18.0),          # ~7.5 days
    ("Mumbai", "Singapore", "Air", 18.0, 10.0, 10.0),             # Regional Air
    ("Los Angeles", "Shanghai", "Ocean", 384.0, 24.0, 24.0),      # Transpacific Westbound
    ("Los Angeles", "Shanghai", "Air", 30.0, 14.0, 12.0),         # Air Express
    ("Rotterdam", "Mumbai", "Ocean", 550.0, 26.0, 24.0),          # Europe-India Maritime
    ("Rotterdam", "Mumbai", "Air", 28.0, 14.0, 12.0),             # Air Freight
    ("Rotterdam", "Shanghai", "Ocean", 740.0, 28.0, 24.0),        # Europe-East Asia Maritime
    ("Rotterdam", "Shanghai", "Multimodal", 400.0, 24.0, 24.0),   # Rail-Sea Intermodal
]


# ==============================================================================
# SIMULATION FUNCTIONS (DELAY GENERATION & STAGE PROPAGATION)
# ==============================================================================

def sample_scheduled_departure(start_dt: datetime, end_dt: datetime) -> datetime:
    """Sample a random scheduled departure timestamp rounded to a 15-minute slot."""
    total_seconds = int((end_dt - start_dt).total_seconds())
    offset_seconds = random.randint(0, total_seconds)
    sampled = start_dt + timedelta(seconds=offset_seconds)
    minute_rounded = (sampled.minute // 15) * 15
    return sampled.replace(minute=minute_rounded, second=0, microsecond=0)


def simulate_departure_delay() -> float:
    """
    Simulates departure delay in hours.
    Reflects realistic container terminal and airport gate operations:
      - 20% Early Departure: -2.5 to -0.2 hours (early vessel loading / expedited gate)
      - 50% On-Time / Minor Variance: -0.2 to 1.5 hours
      - 20% Moderate Delay: 2.0 to 10.0 hours (berth queue, paperwork hold, crane maintenance)
      - 10% Severe Delay: 12.0 to 36.0 hours (vessel rescheduling, blank sailings, mechanical failure)
    """
    category = np.random.choice(["early", "on_time", "moderate", "severe"], p=[0.20, 0.50, 0.20, 0.10])
    if category == "early":
        return round(float(np.random.uniform(-2.5, -0.2)), 2)
    elif category == "on_time":
        return round(float(np.random.uniform(-0.2, 1.5)), 2)
    elif category == "moderate":
        return round(float(np.random.uniform(2.0, 10.0)), 2)
    else:
        return round(float(np.random.uniform(12.0, 36.0)), 2)


def simulate_in_transit_duration(
    scheduled_transit_hours: float,
    transport_mode: str,
    weather_severity: str
) -> float:
    """
    Simulates actual transit duration from departure to destination port arrival.
    
    Delay Propagation Logic:
      - Mild Weather: Vessels operate normally, with minor speed adjustments (-1.5% to +0.8%).
      - Moderate Weather: Adverse currents, rough seas, or head-winds impose speed reductions (+1% to +3%)
        plus direct weather delays (Ocean: +5 to +16h, Air: +1.5 to +4h).
      - Severe Weather: Typhoons, tropical storms, or winter blizzards necessitate detour routing or
        protective anchoring (+3% to +8% speed loss + Ocean: +24 to +64h, Air: +6 to +18h).
    """
    if weather_severity == "Mild":
        speed_factor = float(np.random.uniform(-0.015, 0.008))
        weather_delay = 0.0
    elif weather_severity == "Moderate":
        speed_factor = float(np.random.uniform(0.01, 0.03))
        weather_delay = float(np.random.uniform(5.0, 16.0)) if transport_mode != "Air" else float(np.random.uniform(1.5, 4.0))
    else:  # Severe
        speed_factor = float(np.random.uniform(0.03, 0.08))
        weather_delay = float(np.random.uniform(24.0, 64.0)) if transport_mode != "Air" else float(np.random.uniform(6.0, 18.0))

    actual_duration = scheduled_transit_hours * (1.0 + speed_factor) + weather_delay
    min_transit = scheduled_transit_hours * 0.85  # Physical speed limit constraint
    return round(float(max(min_transit, actual_duration)), 2)


def simulate_port_and_customs_duration(
    scheduled_customs_hours: float,
    transport_mode: str,
    port_congestion: str,
    customs_workload: str
) -> float:
    """
    Simulates elapsed time between port arrival and customs clearance completion.
    
    Delay Propagation Drivers:
      1. Port Congestion Level: Dictates outer anchorage waiting time and berth queueing.
         - Low: Swift docking and unlading (2-5h ocean, 1-2.5h air)
         - Medium: Congested terminals and container yard stacking (7-16h ocean, 2.5-6h air)
         - High: Severe port gridlock / vessel queues (20-48h ocean, 6-14h air)
      2. Customs Workload: Dictates declaration clearance and physical inspection latency.
         - Low: Green channel rapid clearance (4-10h ocean, 3-6h air)
         - Medium: Document audits and scanning (10-18h ocean, 6-11h air)
         - High: Red channel physical inspections, manual tariff audits (20-40h ocean, 12-22h air)
    """
    # 1. Berthing and terminal handling duration
    if transport_mode == "Ocean":
        if port_congestion == "Low":
            port_wait = float(np.random.uniform(2.0, 5.0))
        elif port_congestion == "Medium":
            port_wait = float(np.random.uniform(7.0, 16.0))
        else:  # High
            port_wait = float(np.random.uniform(20.0, 48.0))
    elif transport_mode == "Multimodal":
        if port_congestion == "Low":
            port_wait = float(np.random.uniform(2.0, 4.0))
        elif port_congestion == "Medium":
            port_wait = float(np.random.uniform(5.0, 12.0))
        else:
            port_wait = float(np.random.uniform(15.0, 30.0))
    else:  # Air
        if port_congestion == "Low":
            port_wait = float(np.random.uniform(1.0, 2.5))
        elif port_congestion == "Medium":
            port_wait = float(np.random.uniform(2.5, 6.0))
        else:
            port_wait = float(np.random.uniform(6.0, 14.0))

    # 2. Customs agency processing duration
    if transport_mode == "Air":
        if customs_workload == "Low":
            customs_proc = float(np.random.uniform(3.0, 6.0))
        elif customs_workload == "Medium":
            customs_proc = float(np.random.uniform(6.0, 11.0))
        else:
            customs_proc = float(np.random.uniform(12.0, 22.0))
    else:  # Ocean & Multimodal
        if customs_workload == "Low":
            customs_proc = float(np.random.uniform(4.0, 10.0))
        elif customs_workload == "Medium":
            customs_proc = float(np.random.uniform(10.0, 18.0))
        else:
            customs_proc = float(np.random.uniform(20.0, 40.0))

    total_duration = port_wait + customs_proc
    return round(float(max(3.0, total_duration)), 2)


def simulate_last_mile_delivery_duration(
    scheduled_delivery_hours: float,
    transport_mode: str
) -> float:
    """
    Simulates final drayage / inland freight transit from customs release to customer delivery.
    Models normal operational dispatch variations, driver scheduling, and gate appointments.
    """
    variance = float(np.random.normal(loc=0.0, scale=2.0))
    actual_hours = scheduled_delivery_hours + variance
    min_delivery = max(3.0, scheduled_delivery_hours * 0.5)
    return round(float(max(min_delivery, actual_hours)), 2)


# ==============================================================================
# CORE GENERATION PIPELINE
# ==============================================================================

def generate_raw_shipment_records(num_records: int = 300) -> list[dict]:
    """
    Generates 300 shipment journeys with full end-to-end delay propagation.
    
    Stage-by-Stage Propagation:
      1. actual_departure = scheduled_departure + departure_delay
      2. actual_port_arrival = actual_departure + actual_transit_duration
      3. actual_customs_completion = actual_port_arrival + actual_customs_duration
      4. actual_delivery = actual_customs_completion + actual_delivery_duration
      
    This guarantees that every stage's delay carries forward into all subsequent
    milestones, realistically modeling cumulative supply chain delay propagation.
    """
    records = []

    for i in range(1, num_records + 1):
        shipment_id = f"SHP-{i:04d}"

        # 1. Route & Transport Mode Selection
        route = random.choice(ROUTE_CATALOG)
        origin, destination, mode, sched_transit_h, sched_customs_h, sched_delivery_h = route

        # 2. Operational Conditions Sampling
        # Realistic distribution: mild weather common, high congestion and severe weather infrequent
        weather_sev = np.random.choice(["Mild", "Moderate", "Severe"], p=[0.68, 0.22, 0.10])
        port_cong = np.random.choice(["Low", "Medium", "High"], p=[0.58, 0.30, 0.12])
        customs_work = np.random.choice(["Low", "Medium", "High"], p=[0.55, 0.32, 0.13])

        # 3. Scheduled Timestamps (Strict Chronological Ordering)
        sched_dep = sample_scheduled_departure(START_DATE_SIMULATION, END_DATE_SIMULATION)
        sched_arr = sched_dep + timedelta(hours=sched_transit_h)
        sched_cust = sched_arr + timedelta(hours=sched_customs_h)
        sched_deliv = sched_cust + timedelta(hours=sched_delivery_h)

        # 4. Actual Journey Progression (Stage-to-Stage Delay Propagation)
        # Stage 1: Departure
        dep_delay_h = simulate_departure_delay()
        act_dep = sched_dep + timedelta(hours=dep_delay_h)

        # Stage 2: Ocean/Air Transit to Port
        act_transit_h = simulate_in_transit_duration(sched_transit_h, mode, weather_sev)
        act_arr = act_dep + timedelta(hours=act_transit_h)

        # Stage 3: Port Berthing & Customs Processing
        act_customs_h = simulate_port_and_customs_duration(sched_customs_h, mode, port_cong, customs_work)
        act_cust = act_arr + timedelta(hours=act_customs_h)

        # Stage 4: Last-Mile Delivery
        act_deliv_h = simulate_last_mile_delivery_duration(sched_delivery_h, mode)
        act_deliv = act_cust + timedelta(hours=act_deliv_h)

        # 5. Ground Truth Delay Calculations
        # Consistent formula: actual minus scheduled (in hours)
        calc_dep_delay = round((act_dep - sched_dep).total_seconds() / 3600.0, 2)
        calc_arr_delay = round((act_arr - sched_arr).total_seconds() / 3600.0, 2)
        calc_cust_delay = round((act_cust - sched_cust).total_seconds() / 3600.0, 2)
        calc_final_delay = round((act_deliv - sched_deliv).total_seconds() / 3600.0, 2)

        # Delay status classification: Delayed if final delay > 6.0 hours, otherwise On-Time
        delay_status = "Delayed" if calc_final_delay > 6.0 else "On-Time"

        records.append({
            "shipment_id": shipment_id,
            "origin": origin,
            "destination": destination,
            "transport_mode": mode,
            "scheduled_departure": sched_dep,
            "actual_departure": act_dep,
            "scheduled_port_arrival": sched_arr,
            "actual_port_arrival": act_arr,
            "scheduled_customs_completion": sched_cust,
            "actual_customs_completion": act_cust,
            "scheduled_delivery": sched_deliv,
            "actual_delivery": act_deliv,
            "port_congestion_level": port_cong,
            "weather_severity": weather_sev,
            "customs_workload": customs_work,
            "departure_delay_hours": calc_dep_delay,
            "port_arrival_delay_hours": calc_arr_delay,
            "customs_delay_hours": calc_cust_delay,
            "final_delay_hours": calc_final_delay,
            "delay_status": delay_status,
        })

    return records


def inject_missing_tracking_telemetry(df: pd.DataFrame, missing_ratio: float = 0.07) -> pd.DataFrame:
    """
    Simulates real-world tracking dropouts (e.g., AIS blackout, EDI integration drops, missed scans).
    
    Applies ~5-10% missing values specifically to intermediate tracking stages:
      - actual_departure and departure_delay_hours
      - actual_port_arrival and port_arrival_delay_hours
      - actual_customs_completion and customs_delay_hours
      
    Maintains 100% completeness for:
      - All scheduled timestamps (baseline schedule plan)
      - Environmental and operational factors (weather, congestion, workload)
      - actual_delivery, final_delay_hours, and delay_status (ground truth targets for ML)
    """
    df_masked = df.copy()
    n = len(df_masked)

    intermediate_pairs = [
        ("actual_departure", "departure_delay_hours"),
        ("actual_port_arrival", "port_arrival_delay_hours"),
        ("actual_customs_completion", "customs_delay_hours"),
    ]

    for ts_col, delay_col in intermediate_pairs:
        # Sample ~7% missing rows independently for each intermediate milestone
        mask_count = int(round(n * missing_ratio))
        missing_indices = np.random.choice(n, size=mask_count, replace=False)
        df_masked.loc[missing_indices, ts_col] = pd.NaT
        df_masked.loc[missing_indices, delay_col] = np.nan

    return df_masked


def format_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Formats datetime columns into standardized ISO-8601 strings (YYYY-MM-DD HH:MM:SS)."""
    df_formatted = df.copy()
    ts_cols = [
        "scheduled_departure", "actual_departure",
        "scheduled_port_arrival", "actual_port_arrival",
        "scheduled_customs_completion", "actual_customs_completion",
        "scheduled_delivery", "actual_delivery",
    ]
    for col in ts_cols:
        df_formatted[col] = df_formatted[col].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(x) else ""
        )
    return df_formatted


# ==============================================================================
# DATA QUALITY VALIDATION & REPORTING
# ==============================================================================

def run_data_quality_checks(df_raw: pd.DataFrame, df_final: pd.DataFrame) -> None:
    """
    Executes comprehensive data quality validation:
      1. Verifies 300 unique shipments with no duplicate IDs.
      2. Validates timestamp ordering and positive durations across stages.
      3. Audits missing value percentages per column.
      4. Prints summary statistics for all delay columns.
      5. Reports the class distribution for On-Time vs Delayed shipments.
    """
    print("=" * 78)
    print("                DATA QUALITY AND VALIDATION REPORT                ")
    print("=" * 78)

    # 1. Uniqueness check
    num_rows = len(df_final)
    unique_ids = df_final["shipment_id"].nunique()
    print(f"\n[1] SHIPMENT COUNT & UNIQUENESS")
    print(f"  - Total records: {num_rows} (Expected: 300)")
    print(f"  - Unique shipment IDs: {unique_ids}")
    assert num_rows == 300, f"Expected 300 records, got {num_rows}"
    assert unique_ids == 300, f"Duplicate shipment_ids detected! Count: {unique_ids}"
    print("  --> PASS: Exactly 300 unique shipments verified with zero duplicates.")

    # 2. Chronological sequence and processing duration checks
    print(f"\n[2] TIMELINE ORDERING & STAGE DURATION VALIDATION")
    sched_order_ok = (
        (df_raw["scheduled_departure"] < df_raw["scheduled_port_arrival"]) &
        (df_raw["scheduled_port_arrival"] < df_raw["scheduled_customs_completion"]) &
        (df_raw["scheduled_customs_completion"] < df_raw["scheduled_delivery"])
    ).all()

    actual_order_ok = (
        (df_raw["actual_departure"] < df_raw["actual_port_arrival"]) &
        (df_raw["actual_port_arrival"] < df_raw["actual_customs_completion"]) &
        (df_raw["actual_customs_completion"] < df_raw["actual_delivery"])
    ).all()

    print(f"  - Scheduled timestamps strictly chronological: {sched_order_ok}")
    print(f"  - Actual operational timestamps strictly chronological: {actual_order_ok}")
    assert sched_order_ok, "FAIL: Inverted scheduled timestamps detected!"
    assert actual_order_ok, "FAIL: Inverted actual timestamps detected!"

    transit_durs = (df_raw["actual_port_arrival"] - df_raw["actual_departure"]).dt.total_seconds() / 3600.0
    customs_durs = (df_raw["actual_customs_completion"] - df_raw["actual_port_arrival"]).dt.total_seconds() / 3600.0
    delivery_durs = (df_raw["actual_delivery"] - df_raw["actual_customs_completion"]).dt.total_seconds() / 3600.0

    assert (transit_durs > 0).all(), "FAIL: Negative or zero transit duration detected!"
    assert (customs_durs > 0).all(), "FAIL: Negative or zero customs clearance duration detected!"
    assert (delivery_durs > 0).all(), "FAIL: Negative or zero delivery duration detected!"

    print(f"  - Transit durations:  min={transit_durs.min():.2f}h, max={transit_durs.max():.2f}h (All > 0)")
    print(f"  - Customs durations:  min={customs_durs.min():.2f}h, max={customs_durs.max():.2f}h (All > 0)")
    print(f"  - Delivery durations: min={delivery_durs.min():.2f}h, max={delivery_durs.max():.2f}h (All > 0)")
    print("  --> PASS: All physical durations are strictly positive and logically sound.")

    # 3. Missing Value Audit
    print(f"\n[3] MISSING VALUE AUDIT")
    missing_summary = []
    for col in df_final.columns:
        n_miss = df_final[col].apply(lambda x: pd.isna(x) or str(x).strip() == "").sum()
        pct_miss = (n_miss / num_rows) * 100.0
        missing_summary.append({"Column": col, "Missing Count": n_miss, "Missing (%)": f"{pct_miss:.1f}%"})
    missing_df = pd.DataFrame(missing_summary)
    print(missing_df.to_string(index=False))

    # Validate that actual_delivery and final ground-truth columns have 0 missing values
    assert df_final["actual_delivery"].replace("", np.nan).notnull().all(), "FAIL: actual_delivery has missing values!"
    assert df_final["final_delay_hours"].notnull().all(), "FAIL: final_delay_hours has missing values!"
    assert df_final["delay_status"].notnull().all(), "FAIL: delay_status has missing values!"
    print("  --> PASS: Missingness constrained to intermediate milestones; target fields 100% complete.")

    # 4. Summary Statistics for Delay Metrics
    print(f"\n[4] DELAY COLUMN DESCRIPTIVE STATISTICS (HOURS)")
    delay_cols = ["departure_delay_hours", "port_arrival_delay_hours", "customs_delay_hours", "final_delay_hours"]
    numeric_delays = df_final[delay_cols].apply(pd.to_numeric, errors="coerce")
    stats_df = numeric_delays.describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    print(stats_df.round(2).to_string())

    # 5. Delay Status Breakdown
    print(f"\n[5] DELAY STATUS CLASSIFICATION BREAKDOWN")
    status_counts = df_final["delay_status"].value_counts()
    status_pcts = df_final["delay_status"].value_counts(normalize=True) * 100.0
    for status, count in status_counts.items():
        print(f"  - {status:<8}: {count:3d} shipments ({status_pcts[status]:.2f}%)")
    print("=" * 78)


# ==============================================================================
# SCRIPT ENTRYPOINT
# ==============================================================================

def main():
    print("Initializing Synthetic Shipment Dataset Generator...")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Destination folder ready: '{OUTPUT_DIR}'")

    # Generate 300 realistic shipment lifecycles
    raw_records = generate_raw_shipment_records(num_records=NUM_SHIPMENTS)
    df_raw = pd.DataFrame(raw_records)

    # Inject realistic 7% missing tracking telemetry in intermediate actual timestamps
    df_with_missing = inject_missing_tracking_telemetry(df_raw, missing_ratio=0.07)

    # Execute comprehensive validation suite
    run_data_quality_checks(df_raw, df_with_missing)

    # Format datetime objects as clean ISO-8601 strings
    df_export = format_timestamps(df_with_missing)

    # Save to data/shipments.csv
    df_export.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDataset successfully generated and saved to: {OUTPUT_FILE}")
    print(f"Dataset shape: {df_export.shape[0]} rows x {df_export.shape[1]} columns\n")


if __name__ == "__main__":
    main()
