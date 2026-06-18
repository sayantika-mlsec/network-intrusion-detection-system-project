import pandas as pd
import numpy as np
from pathlib import Path

from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset

def evaluate_nids_drift(df_2017: pd.DataFrame, df_2018: pd.DataFrame, output_file: str = "nids_drift_report.html"):
    """
    Executes a strict schema alignment between CICIDS 2017 and 2018 datasets,
    enforces data contracts, and runs an Evidently AI Data Drift evaluation.
    """
    
    print("Step 1: Translating 2018 schema to 2017 baseline...")
    rename_dict = {
        'Tot Fwd Pkts': 'Total Fwd Packet', 'Tot Bwd Pkts': 'Total Bwd packets',
        'TotLen Fwd Pkts': 'Total Length of Fwd Packet', 'TotLen Bwd Pkts': 'Total Length of Bwd Packet',
        'Fwd Pkt Len Max': 'Fwd Packet Length Max', 'Fwd Pkt Len Min': 'Fwd Packet Length Min',
        'Fwd Pkt Len Mean': 'Fwd Packet Length Mean', 'Fwd Pkt Len Std': 'Fwd Packet Length Std',
        'Bwd Pkt Len Max': 'Bwd Packet Length Max', 'Bwd Pkt Len Min': 'Bwd Packet Length Min',
        'Bwd Pkt Len Mean': 'Bwd Packet Length Mean', 'Bwd Pkt Len Std': 'Bwd Packet Length Std',
        'Flow Byts/s': 'Flow Bytes/s', 'Flow Pkts/s': 'Flow Packets/s',
        'Fwd IAT Tot': 'Fwd IAT Total', 'Bwd IAT Tot': 'Bwd IAT Total',
        'Fwd Header Len': 'Fwd Header Length', 'Bwd Header Len': 'Bwd Header Length',
        'Fwd Pkts/s': 'Fwd Packets/s', 'Bwd Pkts/s': 'Bwd Packets/s',
        'Pkt Len Min': 'Packet Length Min', 'Pkt Len Max': 'Packet Length Max',
        'Pkt Len Mean': 'Packet Length Mean', 'Pkt Len Std': 'Packet Length Std',
        'Pkt Len Var': 'Packet Length Variance', 'FIN Flag Cnt': 'FIN Flag Count',
        'SYN Flag Cnt': 'SYN Flag Count', 'RST Flag Cnt': 'RST Flag Count',
        'PSH Flag Cnt': 'PSH Flag Count', 'ACK Flag Cnt': 'ACK Flag Count',
        'URG Flag Cnt': 'URG Flag Count', 'CWE Flag Count': 'CWR Flag Count',
        'ECE Flag Cnt': 'ECE Flag Count', 'Pkt Size Avg': 'Average Packet Size',
        'Fwd Seg Size Avg': 'Fwd Segment Size Avg', 'Bwd Seg Size Avg': 'Bwd Segment Size Avg',
        'Subflow Fwd Pkts': 'Subflow Fwd Packets', 'Subflow Fwd Byts': 'Subflow Fwd Bytes',
        'Subflow Bwd Pkts': 'Subflow Bwd Packets', 'Subflow Bwd Byts': 'Subflow Bwd Bytes',
        'Init Fwd Win Byts': 'FWD Init Win Bytes', 'Init Bwd Win Byts': 'Bwd Init Win Bytes',
        'Fwd Byts/b Avg': 'Fwd Bytes/Bulk Avg', 'Fwd Pkts/b Avg': 'Fwd Packet/Bulk Avg',
        'Fwd Blk Rate Avg': 'Fwd Bulk Rate Avg', 'Bwd Byts/b Avg': 'Bwd Bytes/Bulk Avg',
        'Bwd Pkts/b Avg': 'Bwd Packet/Bulk Avg', 'Bwd Blk Rate Avg': 'Bwd Bulk Rate Avg'
    }
    df_2018_renamed = df_2018.rename(columns=rename_dict)

    print("Step 2: Computing feature sets and enforcing Data Contracts...")
    cols_2017 = set(df_2017.columns)
    cols_2018 = set(df_2018_renamed.columns)

    dropped_from_2017 = cols_2017 - cols_2018
    dropped_from_2018 = cols_2018 - cols_2017

    # Strict Data Contract: These are the exact 9 columns CICFlowMeter abandoned
    expected_orphans_2017 = {
        'ICMP Code', 'ICMP Type', 'Total TCP Flow Time',
        'Attempted Category', 'Src IP dec', 'Dst IP dec',
        'Fwd RST Flags', 'Bwd RST Flags', 'Src Port'
    }

    # Pipeline halting assertions
    assert dropped_from_2017 == expected_orphans_2017, \
        f"DATA CONTRACT BREACH: Unexpected orphans from 2017 schema: {dropped_from_2017 - expected_orphans_2017}"
    
    assert len(dropped_from_2018) == 0, \
        f"DATA CONTRACT BREACH: 2018 dataset introduced unexpected new columns: {dropped_from_2018}"

    print("Step 3: Intersecting schemas for absolute feature parity...")
    shared_columns = sorted(list(cols_2017.intersection(cols_2018)))
    columns_to_drop = ['Label', 'Timestamp']
    df_2017_pure = df_2017[shared_columns]
    df_2018_pure = df_2018_renamed[shared_columns]
    df_2017_pure = df_2017_pure.drop(columns=columns_to_drop, errors='ignore')
    df_2018_pure = df_2018_pure.drop(columns=columns_to_drop, errors='ignore')
    df_2017_pure = df_2017_pure.replace([float('inf'), float('-inf')], np.nan)
    df_2018_pure = df_2018_pure.replace([float('inf'), float('-inf')], np.nan)
    df_2018_pure = df_2018_pure[df_2018_pure['Subflow Bwd Bytes'] < 50000000]    # Excludes 1 row with internally inconsistent values (98.6MB transfer, 0 ACK flags, 0 active time) — data artifact, verified manually

    print("Step 4: Constructing explicit Data Definition blueprint...")
    # Explicitly mapping categorical flags to prevent Evidently from using continuous math (e.g. Wasserstein)
    known_categorical_flags = [
        "Protocol", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", 
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count", 
        "CWR Flag Count", "ECE Flag Count"
    ]
    
    # Dynamically routing all remaining intersecting features to the numerical bucket
    # Note: If 'Label' survived the intersection, drop it from these lists so it isn't evaluated for drift.
    numerical_measurements = [col for col in df_2017_pure.columns if col not in known_categorical_flags]
    known_broken_columns = ['Subflow Bwd Packets', 'Bwd URG Flags', 'Fwd URG Flags']  
    # Confirmed constant 0 across full datasets (both years, including attack/Bot traffic) — verified, not assumed

    numerical_measurements = [col for col in numerical_measurements if col not in known_broken_columns]

    drift_definition = DataDefinition(
        categorical_columns=known_categorical_flags,
        numerical_columns=numerical_measurements
    )

    print("Step 5: Instantiating Evidently Datasets...")
    ref_dataset = Dataset.from_pandas(df_2017_pure, data_definition=drift_definition)
    cur_dataset = Dataset.from_pandas(df_2018_pure, data_definition=drift_definition)

    print("Step 6: Executing Data Drift Report...")
    # By default, this will flag 'Dataset Drift' if >= 50% of the columns drift
    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_data=ref_dataset, current_data=cur_dataset)
    
    print(f"Step 7: Exporting dashboard to {output_file}...")
    my_eval.save_html(output_file)

    print("Pipeline execution complete.")


# Execution Block

if __name__ == "__main__":
    
    # --- Load 2017 Reference Data ---
    script_dir = Path(__file__).parent.absolute()
    data_2017 = (script_dir.parent / 'data' / 'reference_2017_baseline.csv')
    
    print("--- Loading 2017 Reference Data ---")
    if not data_2017.exists():
        raise FileNotFoundError(f"No 2017 CSV files found. Checked here: {data_2017}")
    
    df_2017_raw = pd.read_csv(data_2017)
    
    df_2017_raw = df_2017_raw[df_2017_raw['Label'] == 'BENIGN']
    print(len(df_2017_raw))
    
    print(f"Successfully loaded {len(df_2017_raw)} rows for 2017.\n")

    # --- Load 2018 Current Data (Sampled) ---
    print("--- Loading 2018 Current Data ---")
    data_2018 = (script_dir.parent / 'cicids_2018' / 'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv')
    if not data_2018.exists():
        raise FileNotFoundError(f"Could not find the external 2018 file at: {data_2018}")

    print(f"Loading and sampling 50,000 rows from {data_2018}...")
    
    # Load the specific file and extract exactly 8000 Benign random rows.
    # 'random_state=42' guarantees you get the exact same 8k rows every time you run it.

    df_2018_raw = pd.read_csv(data_2018)

    df_2018_raw = df_2018_raw[df_2018_raw['Label'] == 'Benign']
    print(len(df_2018_raw))

    df_2018_raw = df_2018_raw.sample(n=8000, random_state=42)
    
    print(f"Successfully loaded {len(df_2018_raw)} rows for 2018.\n")


    # --- Execute the Pipeline ---
    print("--- Initiating Drift Evaluation Pipeline ---")
    evaluate_nids_drift(df_2017_raw, df_2018_raw, output_file="nids_drift_report_benign_only.html")

