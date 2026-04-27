# network-intrusion-detection-system-project
# NIDS Threat Detection API 🛡️

## 1. The Problem Statement
Security Operations Center (SOC) analysts are currently drowning in a sea of network alerts, making it nearly impossible to triage critical incidents efficiently. The core of this pain point stems from traditional rule-based firewalls, which trigger thousands of false positives daily and cause severe "alert fatigue." 

This machine learning API solves that bottleneck by analyzing the hidden, multidimensional statistical patterns of network traffic to classify threats with high precision. By significantly reducing false alarms, it empowers analysts to focus their limited time and resources on neutralizing real, verified attacks.

## 2. The Dataset
This project was built and trained using the benchmark **CICIDS 2017** dataset, which contains highly realistic captures of network traffic. 

The primary engineering challenge was the massive class imbalance inherent to real security data, where the vast majority of the traffic is purely benign. To prevent the model from blindly predicting "Benign" to achieve a superficially high accuracy, I implemented **SMOTE** (Synthetic Minority Over-sampling Technique) alongside optimized **XGBoost class weights** to heavily penalize the system for missing rare, critical attacks.

## 3. The Methodology
The architecture of this system is designed for speed, security, and explainability. Here is the lifecycle of a single network packet passing through the API:

* **Step 1: Raw Feature Extraction:** The system ingests structured network flow features (such as `Flow_Duration`, `Total_Fwd_Packets`, and `Destination_Port`) representing the metadata of raw PCAP network captures.
* **Step 2: Preprocessing Pipeline:** Incoming data passes through a Scikit-Learn `ColumnTransformer` that robustly handles missing values and scales numerical features using `StandardScaler` to ensure the model interprets all data points evenly. 
* **Step 3: The Inference Engine & Optuna Tuning:** An **XGBoost** classifier acts as the core engine. Because a False Negative (missing a real attack) is vastly more dangerous in cybersecurity than a False Positive, I utilized **Optuna** to automate hyperparameter tuning. Instead of optimizing for standard accuracy, the Optuna trials were explicitly instructed to maximize the **F2-Score**, which mathematically weights Recall higher than Precision. 
* **Step 4: The Deployment:** The model is served via a high-performance **FastAPI** endpoint, which is protected by a strict **Pydantic** `BaseModel` "bouncer." This ensures that incoming JSON payloads mathematically match the model's required schema before any compute power is spent on inference.
* **Step 5: Conditional Explainability:** If an attack is detected, the API dynamically wakes up a **SHAP** `TreeExplainer` to extract the top 3 most impactful features triggering the alert. This translates the model's mathematical decision into plain English, providing the SOC analyst with immediate, actionable context without slowing down the processing of benign traffic.

## 4. Real-World Limitations & Considerations
While CICIDS 2017 is a standard benchmark, deploying a model trained exclusively on this data into a modern production environment requires acknowledging several critical limitations:

* **Concept Drift & Age:** The dataset captures 2017 traffic patterns. It lacks visibility into modern zero-day vulnerabilities, recent ransomware behaviors, and the massive architectural shift to remote-work VPNs and encrypted TLS 1.3 traffic that defines networks today.
* **Simulation Artifacts & Overfitting:** Because the data was generated in a controlled lab, ML algorithms are highly prone to overfitting on meaningless simulation artifacts (such as specific hardcoded MAC addresses, predictable TTL values, or localized TCP window sizes) rather than learning the actual underlying malicious behavior.
* **Labeling Inconsistencies:** Independent academic reviews have highlighted flaws in the dataset's original flow construction, such as TCP timeout misconfigurations that occasionally cause subsequent malicious flows to adopt the wrong direction and be mislabeled as benign traffic. In production, this model would require continuous monitoring and retraining on live, localized SOC data.

