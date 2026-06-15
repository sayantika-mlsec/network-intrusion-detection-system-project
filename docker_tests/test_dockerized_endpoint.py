import requests
import pandas as pd
import json

# Load the test data exported from your training run
df_test = pd.read_csv("X_test.csv")

# Grab the first row and convert it to a dictionary
payload = df_test.iloc[3].to_dict()

# Define your local Docker endpoint
url = "http://localhost:8000/predict" 

print("Sending network packet to NIDS Container...")

# Send the POST request
response = requests.post(url, json=payload)

# Print the beautifully formatted JSON response
if response.status_code == 200:
    print("\n--- NIDS PREDICTION RESPONSE ---")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"\nFailed with Status Code: {response.status_code}")
    print(response.text)