import pytest
import os
import pandas as pd
from fastapi.testclient import TestClient
from app import app, ml_models

# ------------------------------------------
# ARCHITECTURAL DECISION: MODULE-SCOPED FIXTURE
# ------------------------------------------
# Why use `@pytest.fixture(scope="module")` instead of putting 
# `with TestClient(app) as client:` inside every single test function?
#
# 1. The "Startup Tax": Our FastAPI 'lifespan' event loads heavy Machine Learning 
#    models (Scikit-Learn pipeline, SHAP explainer) from disk into memory. 
#    This takes about 6 seconds.
# 2. The Bottleneck: If we initialize the client inside each test function, 
#    the server boots up and shuts down for EVERY test. Two tests would take 
#    12 seconds. Ten tests would take 60 seconds.
# 3. The Fix: Setting `scope="module"` tells Pytest to boot the server exactly ONCE 
#    when this file runs. The heavy ML models stay hot in memory, and all tests 
#    share that single running server. This drops total execution time drastically.
# -------------------------------------------

# --- Boot the server exactly ONE time ---
@pytest.fixture(scope="module")
def client():
    print("\nBooting up the ML Server for testing...")

    # The 'with' block triggers the FastAPI lifespan event (loading models)
    with TestClient(app) as c:
        yield c  
    
    # Once all tests finish, it resumes and triggers the shutdown event
    print("\nShutting down the ML Server...")


# --- TEST 1: BENIGN ---

def test_load_and_predict_benign(client):
    
    csv_path = os.path.join(os.path.dirname(__file__), "X_test_labeled.csv")
    df = pd.read_csv(csv_path)

    benign_row = df[df['Label'] == 'BENIGN'].iloc[0]
    payload = benign_row.drop('Label').to_dict()

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    known_classes = set(ml_models["label_encoder"].classes_)
    assert data["threat_classification"] in known_classes

#--- TEST 2: ATTACK ---

def test_load_and_predict_attack(client):
    csv_path = os.path.join(os.path.dirname(__file__), "X_test_labeled.csv")
    df = pd.read_csv(csv_path)

    attack_row = df[df['Label'] != 'BENIGN'].iloc[0]
    payload = attack_row.drop('Label').to_dict()

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    known_classes = set(ml_models["label_encoder"].classes_)
    assert data["threat_classification"] in known_classes
    assert data["threat_classification"] != "BENIGN"
    assert len(data["top_3_features"]) > 0