import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app import app, ml_models  

client = TestClient(app)

# Load a REAL row of data to satisfy the Pydantic Bouncer
X_test_sample = pd.read_csv("X_test.csv", nrows=1)
valid_payload = X_test_sample.to_dict(orient="records")[0]

# We create our mock objects first
mock_preprocessor = MagicMock()
mock_model = MagicMock()
mock_label_encoder = MagicMock()
mock_explainer = MagicMock()

# --- TEST 1: BENIGN TRAFFIC ---
# Patch the dictionary, not global variables
@patch.dict('app.ml_models', {
    'preprocessor': mock_preprocessor, 
    'model': mock_model, 
    'label_encoder': mock_label_encoder, 
    'explainer': mock_explainer
})
def test_benign_traffic_skips_shap():
    mock_model.predict.return_value = [0]  
    mock_model.predict_proba.return_value = [[0.99, 0.01]] 
    mock_label_encoder.inverse_transform.return_value = ["BENIGN"] 

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["threat_classification"] == "BENIGN"
    assert data["top_3_features"] == [] 
    
    mock_explainer.assert_not_called()

# --- TEST 2: ATTACK TRAFFIC ---

@patch('app.get_top_3_shap_features') 
@patch.dict('app.ml_models', {
    'preprocessor': mock_preprocessor,
    'model': mock_model, 
    'label_encoder': mock_label_encoder, 
    'explainer': mock_explainer
})
def test_attack_traffic_triggers_shap(mock_get_top_3):
    
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.05, 0.95]]
    mock_label_encoder.inverse_transform.return_value = ["DDoS"]
    
    mock_get_top_3.return_value = [{"feature": "Dst_port", "impact_score": 4.5, "plain_english": "..."}]

    response = client.post("/predict", json=valid_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["threat_classification"] == "DDoS"
    assert len(data["top_3_features"]) > 0
    
    assert mock_explainer.called
    assert mock_get_top_3.called