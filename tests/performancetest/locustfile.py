from locust import HttpUser, between, task


class DisasterTweetUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        self.client.post(
            "/predict",
            json={"text": "There is a huge earthquake and people need help"},
        )
