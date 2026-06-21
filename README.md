# NIDS Threat Detection API 🛡️

## 1. The Problem Statement
Security Operations Center (SOC) analysts are currently drowning in a sea of network alerts, making it nearly impossible to triage critical incidents efficiently. The core of this pain point stems from traditional rule-based firewalls, which trigger thousands of false positives daily and cause severe "alert fatigue."

This machine learning API solves that bottleneck by analyzing the hidden, multidimensional statistical patterns of network traffic to classify threats with high precision. By significantly reducing false alarms, it empowers analysts to focus their limited time and resources on neutralizing real, verified attacks.

## 2. The Dataset
This project was built and trained using the benchmark **CICIDS 2017** dataset, which contains highly realistic captures of network traffic, pre-processed into structured flow-level features rather than raw packet captures.

The primary engineering challenge was the massive class imbalance inherent to real security data, where the vast majority of the traffic is purely benign. To prevent the model from blindly predicting "Benign" to achieve a superficially high accuracy, I implemented **SMOTE** (Synthetic Minority Over-sampling Technique) alongside optimized **XGBoost class weights** to heavily penalize the system for missing rare, critical attacks.

## 3. The Methodology
The architecture of this system is designed for speed, security, and explainability. Here is the lifecycle of a single network flow passing through the API:

* **Step 1: Flow Feature Ingestion:** The system ingests pre-extracted network flow features (such as `Flow_Duration`, `Total_Fwd_Packets`, and `Destination_Port`) — the statistical summary of a network conversation, not raw packet bytes. Raw PCAP parsing is explicitly out of scope for this project; CICIDS 2017's flow-level features are the input contract.
* **Step 2: Preprocessing Pipeline:** Incoming data passes through an **imbalanced-learn `ImbPipeline`** — used instead of a vanilla Scikit-Learn pipeline so that SMOTE resampling is correctly scoped to training folds only, with no leakage into validation data. Numerical features are scaled with **`RobustScaler`**, chosen over `StandardScaler` for its resilience to the extreme outliers common in network flow data (e.g. a single flow with an abnormally large packet count shouldn't distort the scale for every other feature).
* **Step 3: The Inference Engine & Optuna Tuning:** An **XGBoost** multi-class classifier acts as the core engine, predicting across 26 distinct traffic classes (specific attack types plus benign) rather than a simple binary attack/no-attack flag. Because a False Negative (missing a real attack) is vastly more dangerous in cybersecurity than a False Positive, I used **Optuna** to automate hyperparameter tuning. Instead of optimizing for standard accuracy, the Optuna trials were explicitly instructed to maximize the **F2-Score**, which mathematically weights Recall higher than Precision.
* **Step 4: The Deployment:** The model is served via a high-performance **FastAPI** endpoint, protected by a strict **Pydantic** `BaseModel` schema check. This ensures incoming JSON payloads mathematically match the model's required schema before any compute power is spent on inference.
* **Step 5: Conditional Explainability:** If an attack is detected, the API dynamically wakes up a **SHAP** `TreeExplainer` to extract the top 3 most impactful features triggering the alert. This translates the model's mathematical decision into plain English, providing the SOC analyst with immediate, actionable context without slowing down the processing of benign traffic.

## 4. MLOps & Production Engineering
Beyond the model itself, this project is tracked and validated the way a production ML service would be:

* **Experiment Tracking (MLflow):** Every Optuna trial — parameters, metrics, and the final tuned model — is logged to MLflow, so the tuning process and the final model selection are both reproducible and auditable, not just a notebook output.
* **Drift Monitoring (Evidently AI):** A drift report compares the training distribution against CICIDS 2018 as a stand-in production sample. A subset of bulk-transfer features were flagged as flat-zero in the 2018 data — investigated and documented as a likely CICFlowMeter tooling artifact rather than silently dismissed.
* **CI/CD (GitHub Actions):** Every push runs both a unit test suite (mocked model artifacts, fast schema validation) and an integration test suite that triggers the real FastAPI lifespan and exercises the full train -> serialize -> load -> predict round-trip — catching deserialization failures that mocked tests can't.

## 5. Real-World Limitations & Considerations
While CICIDS 2017 is a standard benchmark, deploying a model trained exclusively on this data into a modern production environment requires acknowledging several critical limitations:

* **Concept Drift & Age:** The dataset captures 2017 traffic patterns. It lacks visibility into modern zero-day vulnerabilities, recent ransomware behaviors, and the massive architectural shift to remote-work VPNs and encrypted TLS 1.3 traffic that defines networks today.
* **Simulation Artifacts & Overfitting:** Because the data was generated in a controlled lab, ML algorithms are highly prone to overfitting on meaningless simulation artifacts (such as specific hardcoded MAC addresses, predictable TTL values, or localized TCP window sizes) rather than learning the actual underlying malicious behavior.
* **Labeling Inconsistencies:** Independent academic reviews have highlighted flaws in the dataset's original flow construction, such as TCP timeout misconfigurations that occasionally cause subsequent malicious flows to adopt the wrong direction and be mislabeled as benign traffic. In production, this model would require continuous monitoring and retraining on live, localized SOC data.

## 6. Prerequisites
To build and run this service locally, you strictly need:
* **Git:** Required to clone the repository.
* **Docker (v20.10.0+):** Required to build and spin up the API container.

## 7. Quickstart: Build & Run

* **1. Clone the repository:**

```bash
git clone https://github.com/sayantika-mlsec/network-intrusion-detection-system-project.git
cd network-intrusion-detection-system-project
```

* **2. Build the Docker image:**

```bash
docker build -t nids-api:latest .
```

* **3. Run the container:**

```bash
docker run -d -p 8000:8000 --name nids-service nids-api:latest
```
The API is now live and listening on http://localhost:8000.

## 8. Interacting with the API
The Interactive Dashboard (Swagger UI)
The easiest way to visually test the API is via the auto-generated Swagger UI.
Open your browser and navigate to: http://localhost:8000/docs

## 9. Programmatic Testing
You can send POST requests directly to the `/predict` endpoint. A sample Python test script (`docker_tests/test_dockerized_endpoint.py`) is included in the repository to fire real test data (`X_test.csv`) at the container.

Important: To run the test script locally, you must have Python installed along with the `requests` and `pandas` libraries.

```bash
# 1. Install the required local testing libraries
pip install requests pandas

# 2. Run the test script against the live container
python docker_tests/test_dockerized_endpoint.py
```