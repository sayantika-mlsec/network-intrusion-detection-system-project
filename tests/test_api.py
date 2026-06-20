import pytest
import os
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app import app

client = TestClient(app)


@pytest.fixture                            # CSV dependency moved into a fixture — not module level
def valid_payload():
    csv_path = os.path.join(os.path.dirname(__file__), "X_test.csv")
    if not os.path.exists(csv_path):
        pytest.skip("X_test.csv not available")
    sample = pd.read_csv(csv_path, nrows=1)
    return sample.to_dict(orient="records")[0]

@pytest.fixture
def mock_ml_models():
    return {
        'preprocessor': MagicMock(),
        'model': MagicMock(),
        'label_encoder': MagicMock(),
        'explainer': MagicMock()
    }

# ---TEST 1: BENIGN TRAFFIC ---
def test_benign_traffic_skips_shap(valid_payload, mock_ml_models):
    
    # mock_ml_models is a dict with 'model', 'label_encoder', etc.
    # Set return values on them HERE
    mock_ml_models['model'].predict.return_value = [0]
    mock_ml_models['model'].predict_proba.return_value = [[0.99, 0.01]] 
    mock_ml_models['label_encoder'].inverse_transform.return_value = ["BENIGN"] 

    # Patch the dictionary at TEST TIME using context manager
    # Not as a decorator — as a context manager inside the function
    with patch.dict('app.ml_models', mock_ml_models):  
        response = client.post("/predict", json= valid_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["threat_classification"] == "BENIGN"
    assert data["top_3_features"] == []


    mock_ml_models["explainer"].assert_not_called()

# --- TEST 2: ATTACK TRAFFIC ---

def test_attack_traffic_triggers_shap(valid_payload, mock_ml_models):
    # Configure mocks for ATTACK prediction
    mock_ml_models['model'].predict.return_value = [1]
    mock_ml_models['model'].predict_proba.return_value = [[0.05, 0.95]]
    mock_ml_models['label_encoder'].inverse_transform.return_value = ['DDoS']

    # Patch BOTH the dict AND the shap function
    # Both as context managers — nested
    with patch.dict('app.ml_models', mock_ml_models):
        with patch('app.get_top_3_shap_features') as mock_get_top_3:
            
            # STEP 3: Configure the shap mock return value
            mock_get_top_3.return_value =  [{"feature": "Dst_port", "impact_score": 4.5, "plain_english": "..."}]

            response = client.post("/predict", json=valid_payload)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["threat_classification"] == "DDoS"
    assert len(data["top_3_features"]) > 0

    # Fixed behavioral assertions — specific, not vague
    mock_get_top_3.assert_called_once()  # specific
    mock_ml_models['explainer'].assert_called() 


# --- TEST 3: EMPTY ml_models DICTIONARY ---
def test_predict_fails_gracefully_when_models_not_loaded(valid_payload):

    # Patch ml_models to be completely empty
    # This simulates a cold start or a failed model load
    with patch.dict('app.ml_models', {}, clear=True):  # clear=True empties the dict
        
        response = client.post("/predict", json=valid_payload)


    assert response.status_code == 503  

    # Assert the response body gives a useful message
    data = response.json()
    assert "detail" in data 

# --- TEST 4: MISSING REQUIRED FIELD ---
def test_missing_required_field(valid_payload):
    
    # Removing one critical field e.g. "Src_IP_dec"
    incomplete_payload = valid_payload.copy()
    incomplete_payload.pop("Src IP dec")  

    # POST the broken payload
    response = client.post("/predict", json=incomplete_payload)

    # Asserting error 422 to denote to denote a required field is absent 
    assert response.status_code == 422  

    # Asserting the response body contains the RIGHT error structure
    data = response.json()
    assert "detail" in data  

# --- TEST 5: WRONG DATA TYPE ---
def test_wrong_data_type_for_float_field(valid_payload):

    # Identify a field in  Pydantic model that expects float
    # Replace its value with a non-coercible string
    bad_type_payload = valid_payload.copy()
    bad_type_payload["Packet Length Mean"] = "NIL"  

    # POST it
    response = client.post("/predict", json=bad_type_payload)

    # Assert status code
    assert response.status_code == 422 

    # Verify the error points to the RIGHT field
    data = response.json()
    error_detail = data.get("detail", [])
    
    # The error detail is a list of dicts — each has a 'loc' key showing which field failed
    # Assert that the error is about YOUR specific field, not a generic crash
    assert any("Packet Length Mean" in str(err.get("loc", "")) 
               for err in error_detail)  
    

# --- TEST 6: EXTRA UNKNOWN FIELDS ---
def test_extra_unknown_fields_in_payload(valid_payload, mock_ml_models):

    # Adding fields that do NOT exist in the Pydantic model
    polluted_payload = valid_payload.copy()
    polluted_payload["malicious_extra_key"] = "unexpected_value"
    polluted_payload["another_unknown_field"] = 99999

    # the model needs to respond — patch ml_models so it doesn't fail
    with patch.dict('app.ml_models', mock_ml_models): 
        
        # Configure mock return values for a BENIGN prediction
        mock_ml_models['model'].predict.return_value = [0]        
        mock_ml_models['model'].predict_proba.return_value = [[0.99, 0.01]]   
        mock_ml_models['label_encoder'].inverse_transform.return_value = ["BENIGN"]  

        response = client.post("/predict", json=polluted_payload)

    # Reasoning: In Pydantic, the default configuration for extra fields is 'ignore'. 
    # Because the NetworkPacket Config class only defines 'populate_by_name = True'
    # and does NOT define 'extra = 'forbid', Pydantic will silently strip out the 
    # extra malicious keys, validate the rest of the payload, and process the request normally.
    assert response.status_code == 200

# --- TEST 7: MALFORMED JSON BODY ---
def test_malformed_json_body():

    # Send raw non-JSON content
    # Use client.post with 'content' (raw bytes) instead of 'json'
    # and set the Content-Type header manually
    response = client.post(
        "/predict",
        content=b"this is not json at all {{{",  # raw malformed bytes
        headers={"Content-Type": "application/json"}
    )

    
    assert response.status_code == 422  

    # Verify it's a framework-level rejection — not a model-level crash
    # The response should come back fast — the model was never touched
    data = response.json()
    assert "detail" in data  

    

                
    

   
   