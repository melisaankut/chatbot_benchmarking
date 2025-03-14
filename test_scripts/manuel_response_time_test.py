import requests
import time

import requests

login_url = "https://prodbot-backend.aperion-analytics.eu/login"

login_payload = {
    "email": "-----",
    "password": "-----"
}

headers = {"Content-Type": "application/json"}

login_response = requests.post(login_url, json=login_payload, headers=headers)

if login_response.status_code == 200:
    access_token = login_response.json().get("access_token")
    headers["Authorization"] = f"Bearer {access_token}"

    prompt_url = "https://prodbot-backend.aperion-analytics.eu/new-prompt"
    prompt_payload = {"request": "Retrieve article from Articles table which productid is 9517"}

    start_time = time.time()
    prompt_response = requests.post(prompt_url, json=prompt_payload, headers=headers)
    end_time = time.time()

    print(f"Response Time: {end_time - start_time:.4f} seconds")
    print("Response:", prompt_response.json())
else:
    print(f"Login failed: {login_response.status_code}, {login_response.text}")