# client.py

import requests
import json
import time

# Base URL of the FastAPI application
# If running locally, it's typically http://127.0.0.1:8000
# If deployed, replace with your deployment URL
BASE_URL = "http://127.0.0.1:8000"

def test_health_check():
    """Tests the health check endpoint."""
    print(f"Testing health check at {BASE_URL}/")
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        print("Health Check Response:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the API at {BASE_URL}. Is the FastAPI server running?")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during health check: {e}")

def test_prediction(feature_1: float, feature_2: float):
    """Tests the prediction endpoint with given features."""
    print(f"\nTesting prediction for feature_1={feature_1}, feature_2={feature_2} at {BASE_URL}/predict")
    url = f"{BASE_URL}/predict"
    headers = {"Content-Type": "application/json"}
    data = {"feature_1": feature_1, "feature_2": feature_2}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        prediction_result = response.json()
        print("Prediction Result:")
        print(json.dumps(prediction_result, indent=2))
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the API at {BASE_URL}. Is the FastAPI server running?")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    print("Starting client tests...")
    # Give the server a moment to start if running immediately after uvicorn command
    # time.sleep(1) # Uncomment if you're running this script right after starting uvicorn in another terminal

    test_health_check()

    # Test with some sample data
    test_prediction(feature_1=7.5, feature_2=2.1)
    test_prediction(feature_1=1.0, feature_2=9.0)
    test_prediction(feature_1=5.0, feature_2=5.0)

    print("\nClient tests finished.")
