import requests
import json

def send_request():
    """
    Reads request.json and sends a POST request to the running FastAPI server.
    """
    try:
        with open('request.json', 'r') as f:
            payload = json.load(f)
    except FileNotFoundError:
        print("Error: request.json not found.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode request.json.")
        return

    url = "http://127.0.0.1:8000/generate-next-episode"
    headers = {"Content-Type": "application/json"}

    try:
        print("Sending POST request to:", url)
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes

        print("Response status code:", response.status_code)
        print("Response body:")
        print(response.json())

    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")

if __name__ == "__main__":
    send_request()
