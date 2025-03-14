from locust import HttpUser, task, between

class ChatbotUser(HttpUser):
    wait_time = between(3, 5)

    def on_start(self):
        login_data = {
            "email": "-----",
            "password": "-----"
        }
        response = self.client.post("/login", json=login_data)
        if response.status_code == 200:
            self.access_token = response.json().get("access_token")
        else:
            self.access_token = None

    @task
    def send_prompt(self):
        if self.access_token:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            prompt_data = {"request": "Retrieve article from Articles table which productid is 9517"}
            self.client.post("/new-prompt", json=prompt_data, headers=headers)
