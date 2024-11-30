from locust import HttpUser, task, between
import random

class MLModelUser(HttpUser):
    # The wait time between tasks (in seconds)
    wait_time = between(1, 3)

    @task
    @task
    def upload_and_predict(self):
        # Simulate the image upload process
        with self.client.get("/upload", name="Upload page", catch_response=True) as response:
            if response.status_code == 200:
                # Correct the file path to point to a specific image file
                files = {'file': ('test_3.png', open('./test_images/test_3.png', 'rb'), 'image/png')}
                upload_response = self.client.post("/upload", files=files)

                if upload_response.status_code == 200:
                    # Simulate accessing the prediction result after file upload
                    filename = 'test_3.png'
                    self.client.get(f"/predict/{filename}", name="Prediction result")


    @task
    def evaluate_model(self):
        # This simulates evaluating the model after making predictions
        self.client.post("/evaluate", name="Model evaluation")
