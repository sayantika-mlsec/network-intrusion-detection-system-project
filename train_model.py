# %%
import duckdb
import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# --- ENVIRONMENT CONFIGURATION ---
# Set to True for a quick syntax/logic check.
# Set to False for the full production tuning run.
DEV_MODE = True
N_TRIALS = 2 if DEV_MODE else 20
# ---------------------------------

# %%
import mlflow

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

def replace_inf_with_nan(X):
    # Safer implementation ensuring it operates on a copy
    return X.replace([np.inf, -np.inf], np.nan)

inf_cleaner = FunctionTransformer(replace_inf_with_nan, validate=False)

# Define columns (You will need to adjust these based on the exact CICIDS 2017 headers)
metadata_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'] 
port_cols = ['Src Port', 'Dst Port']

# Drop metadata and target label from X
X_raw = df_sample.drop(columns=metadata_cols + ['Label'], errors='ignore')

# Separate continuous numeric columns from categorical numeric columns (Ports)
continuous_numeric_cols = [col for col in X_raw.columns if col not in port_cols and pd.api.types.is_numeric_dtype(X_raw[col])]

# Build the numeric pipeline
numeric_pipeline = Pipeline([
    ('remove_infinity', inf_cleaner),             
    ('fill_missing', SimpleImputer(strategy='median')),
    ('smart_scaling', RobustScaler())  
])


def encode_well_known_ports(X):
    # X is the DataFrame containing just your port columns
    # We check if values are < 1024, which returns True/False.
    # .astype(int) converts True to 1 and False to 0.
    return (X < 1024).astype(int)

# Create the transformer. 
# feature_names_out='one-to-one' is CRITICAL here so we don't lose the column names for SHAP later.
port_engineer = FunctionTransformer(encode_well_known_ports, feature_names_out='one-to-one')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, continuous_numeric_cols),
        ('ports',port_engineer,port_cols)
    ],
    remainder='drop'
)

# %%
# 1. Define X_raw(defined in the previous cell) and y_raw FIRST
y_raw = df_sample["Label"]

# 2. Split the RAW data (No leakage!)
X_train, X_cv, y_train_raw, y_cv_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

# 3. Encode the target labels (XGBoost needs integers)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_cv = label_encoder.transform(y_cv_raw)

# %%

print("class NetworkPacket(BaseModel):")
print('    """')
print('    Pydantic model automatically generated from X_train columns with ML-compatible aliases.')
print('    """')

for col in X_train.columns:
    # Get the pandas data type
    dtype_str = str(X_train[col].dtype)
    
    # Map Pandas types to standard Python types
    if 'int' in dtype_str:
        py_type = 'int'
    elif 'float' in dtype_str:
        py_type = 'float'
    else:
        py_type = 'str'
        
    # Clean the column names (Python variables cannot have spaces, hyphens, or slashes)
    clean_col_name = col.replace(" ", "_").replace("-", "_").replace("/", "_")
    
    # Print the perfectly formatted Pydantic field with the alias
    print(f'    {clean_col_name}: {py_type} = Field(alias="{col}")')

print("\n    class Config:")
print("        populate_by_name = True")

# %%
# 1. PREP: Fix the multi-class scorer
#Set average='macro' to explicitly tell the model to treat every anomaly class equally
f2_scorer = make_scorer(fbeta_score, beta=2, average='macro')
num_classes = len(np.unique(y_train))

# 1. Define the cap
SYNTHETIC_CAP = 500

# 2. Find the integer encoding for 'BENIGN'
benign_encoded_value = label_encoder.transform(['BENIGN'])[0]

# 3. Dynamically build the dictionary based on whatever is actually in y_train
unique_classes = np.unique(y_train)
smote_strategy = {}

for cls in unique_classes:
    if cls == benign_encoded_value:
        # Ignore the majority class
        continue 
    else:
        # Tell SMOTE to upsample this specific attack class to exactly 500 rows
        smote_strategy[cls] = SYNTHETIC_CAP
#The sampling strategy will be used later for SMOTE to generate 500 synthetic examples of the anomaly classes

# %%

# Defining the objective function for optuna
def objective(trial, X, y):

    # Start a nested MLflow run for each trial
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):

        # Defining the parameters
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1.0, 5.0)
        }

        # Log parameters for this specific trial
        mlflow.log_params(param)

        # Building the Pipeline
        nids_pipeline = ImbPipeline([
            ('preprocessing', preprocessor),
            ('smote', SMOTE(sampling_strategy=smote_strategy, random_state=42)),
            ('classifier', XGBClassifier(**param, random_state=42, eval_metric='mlogloss', objective='multi:softprob', num_class=num_classes, n_jobs=-1))
        ])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(nids_pipeline, X, y, cv=cv, scoring=f2_scorer)
        mean_score = scores.mean()
        # Log the resulting cross-validation score for this trial
        mlflow.log_metric("cv_f2_score", mean_score)

        return mean_score

# EXECUTE THE STUDY
study = optuna.create_study(direction='maximize', study_name="CICIDS_XGBoost_Tuning", sampler=optuna.samplers.TPESampler(seed=42))

print(f"Starting Optuna Hyperparameter Tuning... (DEV_MODE: {DEV_MODE})")

# Open the parent MLflow run
with mlflow.start_run(run_name="Optuna_Study_Parent"):
    # Wrap in a lambda so Optuna can inject the 'trial' object dynamically
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_TRIALS)

    # Log the best parameters and score from the entire study to the parent run
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_cv_f2_score", study.best_value)

print(f"Best F2-Score: {study.best_value:.4f}")
print("Best Params:", study.best_params)

# %%
best_pipeline = ImbPipeline([('preprocessing', preprocessor),
                             ('smote', SMOTE(sampling_strategy=smote_strategy, random_state=42)),
                             ('classifier', XGBClassifier(**study.best_params, random_state=42, eval_metric='mlogloss',objective="multi:softprob",n_jobs=-1, num_class=num_classes))
    ])
best_pipeline.fit(X_train,y_train)
predicted = best_pipeline.predict(X_cv)

labels = sorted(set(y_cv))  # the integer-encoded classes present
per_class_f2 = fbeta_score(y_cv, predicted, beta=2, average=None, labels=labels)
class_names = label_encoder.inverse_transform(labels)

for name, score in zip(class_names, per_class_f2):
    print(f"{name}: {score:.4f}")


print(classification_report(y_cv_raw,label_encoder.inverse_transform(predicted)))


# %%
# Export first 5 rows for robust testing
X_cv.head(5).to_csv("X_test.csv", index=False)

# %%

joblib.dump(best_pipeline, 'nids_pipeline.pkl')
joblib.dump(label_encoder, 'nids_label_encoder.pkl')

# Let's export just the pipeline and encoder for now.
print("Artifacts successfully serialized to disk.")

