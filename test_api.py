import requests
import json

# The local URL where Uvicorn is running
URL = "http://127.0.0.1:8000/predict"

# Create a dummy payload using the ALIAS strings your API expects
# Replace these values with a row from your actual X_test dataset
payload = {
    "Protocol": 6,
    "Source Port": 443,
    "Destination Port": 8080,
    "Flow Duration": 1500.5
    # Add 2-3 more of your actual columns here just to test the mapping
}

print("Sending POST request to NIDS API...")

try:
    response = requests.post(URL, json=payload)
    
    # Print the raw status code and response
    print(f"Status Code: {response.status_code}")
    print(f"Response Body:\n{json.dumps(response.json(), indent=2)}")
    
except Exception as e:
    print(f"Failed to connect: {e}")