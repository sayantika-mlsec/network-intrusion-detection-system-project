import os
import logging
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from typing import List
import joblib
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- DYNAMIC CONFIGURATION ---
MODEL_PATH = os.getenv("MODEL_PATH", "nids_pipeline.pkl")
ENCODER_PATH = os.getenv("ENCODER_PATH", "nids_label_encoder.pkl")


# Global dictionary
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting up ML Engine...")
    try:
        pipeline = joblib.load(MODEL_PATH)
        ml_models["model"] = pipeline.named_steps['classifier']
        ml_models["preprocessor"] = pipeline.named_steps['preprocessing']
        ml_models["label_encoder"] = joblib.load(ENCODER_PATH)
        ml_models["explainer"] = shap.TreeExplainer(ml_models["model"])
        logger.info("ML Engine loaded successfully.")
    
    except Exception as e:
        logger.critical(f"FATAL: Failed to load ML artifacts: {e}", exc_info=True)
        raise RuntimeError(f"ML artifact loading failed: {e}") from e

        
    yield 
    
    logger.info("Shutting down ML Engine. Clearing memory...")
    ml_models.clear()

app = FastAPI(title="NIDS Threat Detection API", lifespan=lifespan)

# --- SOC TRANSLATION DICTIONARY ---
SOC_TRANSLATIONS = {
    "Protocol": "Abnormal protocol detected for this traffic profile.",
    "Src_port": "Suspicious source port usage, potential reconnaissance.",
    "Dst_port": "Anomalous destination port targeted.",
    "Flow_Duration": "Unusually long connection time, indicating potential data exfiltration.",
    # Note: Expand this based on top 10-15 highest-impact SHAP features
}

# --- SCHEMA DEFINITIONS ---
class NetworkPacket(BaseModel):
    # Using aliases ensures pythonic variables while accepting Scikit-Learn's exact strings
    Src_IP_dec: int = Field(alias="Src IP dec")
    Src_Port: int = Field(alias="Src Port")
    Dst_IP_dec: int = Field(alias="Dst IP dec")
    Dst_Port: int = Field(alias="Dst Port")
    Protocol: int = Field(alias="Protocol")
    Flow_Duration: int = Field(alias="Flow Duration")
    Total_Fwd_Packet: int = Field(alias="Total Fwd Packet")
    Total_Bwd_packets: int = Field(alias="Total Bwd packets")
    Total_Length_of_Fwd_Packet: int = Field(alias="Total Length of Fwd Packet")
    Total_Length_of_Bwd_Packet: int = Field(alias="Total Length of Bwd Packet")
    Fwd_Packet_Length_Max: int = Field(alias="Fwd Packet Length Max")
    Fwd_Packet_Length_Min: int = Field(alias="Fwd Packet Length Min")
    Fwd_Packet_Length_Mean: float = Field(alias="Fwd Packet Length Mean")
    Fwd_Packet_Length_Std: float = Field(alias="Fwd Packet Length Std")
    Bwd_Packet_Length_Max: int = Field(alias="Bwd Packet Length Max")
    Bwd_Packet_Length_Min: int = Field(alias="Bwd Packet Length Min")
    Bwd_Packet_Length_Mean: float = Field(alias="Bwd Packet Length Mean")
    Bwd_Packet_Length_Std: float = Field(alias="Bwd Packet Length Std")
    Flow_Bytes_s: float = Field(alias="Flow Bytes/s")
    Flow_Packets_s: float = Field(alias="Flow Packets/s")
    Flow_IAT_Mean: float = Field(alias="Flow IAT Mean")
    Flow_IAT_Std: float = Field(alias="Flow IAT Std")
    Flow_IAT_Max: int = Field(alias="Flow IAT Max")
    Flow_IAT_Min: int = Field(alias="Flow IAT Min")
    Fwd_IAT_Total: int = Field(alias="Fwd IAT Total")
    Fwd_IAT_Mean: float = Field(alias="Fwd IAT Mean")
    Fwd_IAT_Std: float = Field(alias="Fwd IAT Std")
    Fwd_IAT_Max: int = Field(alias="Fwd IAT Max")
    Fwd_IAT_Min: int = Field(alias="Fwd IAT Min")
    Bwd_IAT_Total: int = Field(alias="Bwd IAT Total")
    Bwd_IAT_Mean: float = Field(alias="Bwd IAT Mean")
    Bwd_IAT_Std: float = Field(alias="Bwd IAT Std")
    Bwd_IAT_Max: int = Field(alias="Bwd IAT Max")
    Bwd_IAT_Min: int = Field(alias="Bwd IAT Min")
    Fwd_PSH_Flags: int = Field(alias="Fwd PSH Flags")
    Bwd_PSH_Flags: int = Field(alias="Bwd PSH Flags")
    Fwd_URG_Flags: int = Field(alias="Fwd URG Flags")
    Bwd_URG_Flags: int = Field(alias="Bwd URG Flags")
    Fwd_RST_Flags: int = Field(alias="Fwd RST Flags")
    Bwd_RST_Flags: int = Field(alias="Bwd RST Flags")
    Fwd_Header_Length: int = Field(alias="Fwd Header Length")
    Bwd_Header_Length: int = Field(alias="Bwd Header Length")
    Fwd_Packets_s: float = Field(alias="Fwd Packets/s")
    Bwd_Packets_s: float = Field(alias="Bwd Packets/s")
    Packet_Length_Min: int = Field(alias="Packet Length Min")
    Packet_Length_Max: int = Field(alias="Packet Length Max")
    Packet_Length_Mean: float = Field(alias="Packet Length Mean")
    Packet_Length_Std: float = Field(alias="Packet Length Std")
    Packet_Length_Variance: float = Field(alias="Packet Length Variance")
    FIN_Flag_Count: int = Field(alias="FIN Flag Count")
    SYN_Flag_Count: int = Field(alias="SYN Flag Count")
    RST_Flag_Count: int = Field(alias="RST Flag Count")
    PSH_Flag_Count: int = Field(alias="PSH Flag Count")
    ACK_Flag_Count: int = Field(alias="ACK Flag Count")
    URG_Flag_Count: int = Field(alias="URG Flag Count")
    CWR_Flag_Count: int = Field(alias="CWR Flag Count")
    ECE_Flag_Count: int = Field(alias="ECE Flag Count")
    Down_Up_Ratio: float = Field(alias="Down/Up Ratio")
    Average_Packet_Size: float = Field(alias="Average Packet Size")
    Fwd_Segment_Size_Avg: float = Field(alias="Fwd Segment Size Avg")
    Bwd_Segment_Size_Avg: float = Field(alias="Bwd Segment Size Avg")
    Fwd_Bytes_Bulk_Avg: int = Field(alias="Fwd Bytes/Bulk Avg")
    Fwd_Packet_Bulk_Avg: int = Field(alias="Fwd Packet/Bulk Avg")
    Fwd_Bulk_Rate_Avg: int = Field(alias="Fwd Bulk Rate Avg")
    Bwd_Bytes_Bulk_Avg: int = Field(alias="Bwd Bytes/Bulk Avg")
    Bwd_Packet_Bulk_Avg: int = Field(alias="Bwd Packet/Bulk Avg")
    Bwd_Bulk_Rate_Avg: int = Field(alias="Bwd Bulk Rate Avg")
    Subflow_Fwd_Packets: int = Field(alias="Subflow Fwd Packets")
    Subflow_Fwd_Bytes: int = Field(alias="Subflow Fwd Bytes")
    Subflow_Bwd_Packets: int = Field(alias="Subflow Bwd Packets")
    Subflow_Bwd_Bytes: int = Field(alias="Subflow Bwd Bytes")
    FWD_Init_Win_Bytes: int = Field(alias="FWD Init Win Bytes")
    Bwd_Init_Win_Bytes: int = Field(alias="Bwd Init Win Bytes")
    Fwd_Act_Data_Pkts: int = Field(alias="Fwd Act Data Pkts")
    Fwd_Seg_Size_Min: int = Field(alias="Fwd Seg Size Min")
    Active_Mean: float = Field(alias="Active Mean")
    Active_Std: float = Field(alias="Active Std")
    Active_Max: int = Field(alias="Active Max")
    Active_Min: int = Field(alias="Active Min")
    Idle_Mean: float = Field(alias="Idle Mean")
    Idle_Std: float = Field(alias="Idle Std")
    Idle_Max: int = Field(alias="Idle Max")
    Idle_Min: int = Field(alias="Idle Min")
    ICMP_Code: int = Field(alias="ICMP Code")
    ICMP_Type: int = Field(alias="ICMP Type")
    Total_TCP_Flow_Time: int = Field(alias="Total TCP Flow Time")
    Attempted_Category: int = Field(alias="Attempted Category")

    
    class Config:
        populate_by_name = True # Allows to pass either the alias or the variable name

class ShapFeature(BaseModel):
    feature: str
    impact_score: float
    plain_english: str  

class PredictionResponse(BaseModel):
    threat_classification: str
    confidence_score: float
    top_3_features: List[ShapFeature]  

def get_top_3_shap_features(shap_explanation):
    # Ensure it's a numpy array to easily check its shape
    shap_values = np.array(shap_explanation.values)
    
    # --- DEFENSIVE PROGRAMMING GUARDRAIL ---
    # SHAP values must be 1D (n_features,) for a single instance and single class.
    # If it's 2D or more, np.argsort will sort row-by-row and return garbage indices.
    if shap_values.ndim != 1:
        raise ValueError(
            f"SHAP Explainer Error: Expected 1D array, got shape {shap_values.shape}. "
            "You must slice the SHAP explanation for a specific instance and target class "
            "before passing it to this function."
        )
    # ----------------------------------------

    abs_shap_values = np.abs(shap_values)

    top_n = min(3, len(shap_values))
    # Now it is perfectly safe to use np.argsort
    top_indices = np.argsort(abs_shap_values)[-top_n:][::-1]

    feature_names = shap_explanation.feature_names
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(shap_values))]

    top_3_results = []
    for idx in top_indices:
        feat_name = str(feature_names[idx])
        score = float(shap_values[idx])
        
        # Map to SOC plain English, with a fallback
        english_translation = SOC_TRANSLATIONS.get(
            feat_name, 
            f"Anomalous metric detected in {feat_name}."
        )

        top_3_results.append({
            "feature": feat_name,
            "impact_score": score,
            "plain_english": english_translation
        })

    return top_3_results

# --- ENDPOINTS ---

@app.post("/predict", response_model=PredictionResponse)
async def predict_threat(packet: NetworkPacket):  

    if not ml_models.get('model'):
        raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service unavailable."
    )
    
    try:
        # FAST I/O: Pydantic validation and dict dumping happen on the main event loop
        input_data = pd.DataFrame([packet.model_dump(by_alias=True)]) 
        
        # ISOLATE THE HEAVY LIFTING: Wrap the Scikit-Learn & SHAP math in a normal function
        def compute_ml_inference():
            preprocessor = ml_models["preprocessor"].named_steps['preprocessing']
            model = ml_models["model"].named_steps['classifier']
            
            transformed_data = preprocessor.transform(input_data)
            target_class_idx = int(model.predict(transformed_data)[0])
            probabilities = model.predict_proba(transformed_data)[0]
            confidence_float = float(probabilities[target_class_idx])
            
            prediction_string = str(ml_models["label_encoder"].inverse_transform([target_class_idx])[0])

            if prediction_string == "BENIGN":
            # Explicitly return the Pydantic object, not a raw dictionary
                return PredictionResponse(
                threat_classification=prediction_string,
                confidence_score=confidence_float,
                top_3_features=[] 
            )
            
            else:
                logger.info(f"Threat detected! Classification: {prediction_string} | Confidence: {confidence_float:.2f}")
            
                # Calculate the raw SHAP values
                shap_explanation = ml_models["explainer"](transformed_data)[0, :, target_class_idx]
            
            # --- DEFENSIVE PROGRAMMING FIX ---
            # Extract the CORRECT post-transformation feature names from the preprocessor
            try:
                # This gets the exact names created by OneHotEncoders, Scalers, etc.
                correct_feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                # Fallback safeguard in case a custom transformer lacks this method
                logger.warning("Preprocessor missing get_feature_names_out(). Using generic indices.")
                correct_feature_names = [f"transformed_feat_{i}" for i in range(transformed_data.shape[1])]
            
            # Assign the correct names to the SHAP explanation
            shap_explanation.feature_names = list(correct_feature_names)
            # ---------------------------------

            top_3_reasons = get_top_3_shap_features(shap_explanation)
            
            return PredictionResponse(
                threat_classification=prediction_string,
                confidence_score=confidence_float,
                top_3_features=top_3_reasons
            )
        # DELEGATE: Send the heavy math to the background thread pool
        # It will return the fully-formed PredictionResponse object!
        final_response = await run_in_threadpool(compute_ml_inference)
        
        # RETURN RESULT directly to the user
        return final_response
        
    except ValueError as ve:
        logger.error(f"Data Validation Error: {str(ve)}")
        raise HTTPException(status_code=422, detail=f"Data Validation Error: {str(ve)}")
        
    except Exception as e:
        logger.error(f"Prediction endpoint crashed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during prediction.")