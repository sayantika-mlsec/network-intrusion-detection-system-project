# %%
import duckdb
import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import fbeta_score, make_scorer, precision_recall_fscore_support
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import mlflow

# --- ENVIRONMENT CONFIGURATION ---
# Set to False for the full production tuning run.
DEV_MODE = False
N_TRIALS = 2 if DEV_MODE else 20
# ---------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("NIDS_XGBoost")

# %%
# Define the production-grade query
query = """
WITH BenignSample AS (
    SELECT * FROM 'data/*.csv' 
    WHERE Label = 'BENIGN'
    ORDER BY RANDOM() 
    LIMIT 8000
),
RankedAttacks AS (
    SELECT *,
           ROW_NUMBER() OVER(PARTITION BY Label ORDER BY RANDOM()) as rn
    FROM 'data/*.csv'
    WHERE Label != 'BENIGN'
)
SELECT * FROM BenignSample
UNION ALL
-- Grab up to 200 samples of EVERY specific attack type
SELECT * EXCLUDE(rn) FROM RankedAttacks 
WHERE rn <= 200;
"""

# Execute directly into a Pandas DataFrame
con = duckdb.connect()
con.execute("SET threads TO 1;")
con.execute("SELECT setseed(0.42);")
df_sample = con.execute(query).df()

# Checking how many rows exist for each attack type
class_counts = df_sample['Label'].value_counts()
print("Original Class Counts:\n", class_counts)

# Identify classes with at least 15 rows (the minimum safe threshold for SMOTE + CV)
valid_classes = class_counts[class_counts >= 15].index

# Filter the dataframe to only keep rows belonging to valid classes
df_sample = df_sample[df_sample['Label'].isin(valid_classes)]
print("\nPruned Class Counts:\n", df_sample['Label'].value_counts())

# %%
# Define columns
metadata_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'] 
port_cols = ['Src Port', 'Dst Port']

# Drop metadata and target label from X
X_raw = df_sample.drop(columns=metadata_cols + ['Label'], errors='ignore')

# --- IN-PLACE PANDAS PREPROCESSING (DECOUPLED FROM PICKLE) ---
# 1. Clean Infinities directly on the dataframe copy
X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

# 2. Encode well-known ports directly on the dataframe (< 1024)
for col in port_cols:
    if col in X_raw.columns:
        X_raw[col] = (X_raw[col] < 1024).astype(int)
# -------------------------------------------------------------

# Separate continuous numeric columns from categorical numeric columns (Ports)
continuous_numeric_cols = [col for col in X_raw.columns if col not in port_cols and pd.api.types.is_numeric_dtype(X_raw[col])]

# Build the simplified numeric pipeline (No inf_cleaner)
numeric_pipeline = Pipeline([
    ('fill_missing', SimpleImputer(strategy='median')),
    ('smart_scaling', RobustScaler())  
])

# Build the simplified preprocessor (No port_engineer custom function)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, continuous_numeric_cols),
        ('ports', 'passthrough', port_cols)  # Safely pass through pre-engineered columns
    ],
    remainder='drop'
)

# %%
# 1. Define y_raw
y_raw = df_sample["Label"]

# 2. Split the preprocessed data (No leakage!)
X_train, X_cv, y_train_raw, y_cv_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

# 3. Encode the target labels (XGBoost needs integers)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_cv = label_encoder.transform(y_cv_raw)

# %%
# 1. PREP: Fix the multi-class scorer
f2_scorer = make_scorer(fbeta_score, beta=2, average='macro')
num_classes = len(np.unique(y_train))

# Define SMOTE balancing strategy
SYNTHETIC_CAP = 500
benign_encoded_value = label_encoder.transform(['BENIGN'])[0]
unique_classes = np.unique(y_train)
smote_strategy = {}

for cls in unique_classes:
    if cls == benign_encoded_value:
        continue 
    else:
        smote_strategy[cls] = SYNTHETIC_CAP

# %%
# Defining the objective function for optuna
def objective(trial, X, y):
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1.0, 5.0)
        }

        mlflow.log_params(param)

        nids_pipeline = ImbPipeline([
            ('preprocessing', preprocessor),
            ('smote', SMOTE(sampling_strategy=smote_strategy, random_state=42)),
            ('classifier', XGBClassifier(**param, random_state=42, eval_metric='mlogloss', objective='multi:softprob', num_class=num_classes, n_jobs=-1))
        ])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(nids_pipeline, X, y, cv=cv, scoring=f2_scorer)
        mean_score = scores.mean()
        mlflow.log_metric("cv_f2_score", mean_score)

        return mean_score

# EXECUTE THE STUDY
study = optuna.create_study(direction='maximize', study_name="CICIDS_XGBoost_Tuning", sampler=optuna.samplers.TPESampler(seed=42))

print(f"Starting Optuna Hyperparameter Tuning... (DEV_MODE: {DEV_MODE})")

with mlflow.start_run(run_name="Optuna_Study_Parent"):
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_TRIALS)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_cv_f2_score", study.best_value)

print(f"Best F2-Score: {study.best_value:.4f}")
print("Best Params:", study.best_params)

# %%
with mlflow.start_run(run_name="production_model"):
    mlflow.set_tag("stage", "production")
    
    named_strategy = {
        label_encoder.inverse_transform([k])[0]: int(v)
        for k, v in smote_strategy.items()
    }

    mlflow.log_param("smote_strategy", str(named_strategy))
    mlflow.log_param("decision_rule", "argmax")
    mlflow.log_params(study.best_params)

    best_pipeline = ImbPipeline([
        ('preprocessing', preprocessor),
        ('smote', SMOTE(sampling_strategy=smote_strategy, random_state=42)),
        ('classifier', XGBClassifier(**study.best_params, random_state=42, eval_metric='mlogloss', objective="multi:softprob", n_jobs=-1, num_class=num_classes))
    ])
    
    best_pipeline.fit(X_train, y_train)
    predicted = best_pipeline.predict(X_cv)

    labels = sorted(set(y_cv))
    class_names = label_encoder.inverse_transform(labels)

    precision, recall, _, _ = precision_recall_fscore_support(y_cv, predicted, labels=labels, zero_division=0)
    per_class_f2 = fbeta_score(y_cv, predicted, beta=2, average=None, labels=labels)

    print("\n--- Final Per-Class Metrics ---")
    for name, p, r, f2 in zip(class_names, precision, recall, per_class_f2):
        print(f"{name} -> Precision: {p:.4f} | Recall: {r:.4f} | F2: {f2:.4f}")
        clean_name = str(name).replace(" ", "_").replace("-", "_").replace("/", "_").replace("\\", "_")
        mlflow.log_metric(f"class_{clean_name}_precision", p)
        mlflow.log_metric(f"class_{clean_name}_recall", r)
        mlflow.log_metric(f"class_{clean_name}_f2", f2)

    mlflow.sklearn.log_model(best_pipeline, name="production_pipeline")
    print("\nProduction model, parameters, and per-class metrics successfully logged to MLflow.")
    
# %%
X_cv.head(5).to_csv("X_test.csv", index=False)

# %%
joblib.dump(best_pipeline, 'nids_pipeline.pkl')
joblib.dump(label_encoder, 'nids_label_encoder.pkl')

print("Artifacts successfully serialized to disk without custom function dependencies.")
