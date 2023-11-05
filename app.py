from rag_engine import ElliotEngine
import requests
from fastapi import FastAPI
from ray import serve

app = FastAPI()

#Class copy
@serve.deployment
@serve.ingress(app)
class DeploymentEngine(ElliotEngine):
    def __init__(self):
        super().__init__()

    @app.get("/hello")
    def query_something(self, query_str: str):
        return self.query(query_str)



#Class wrappers from Ray
# 2: Deploy the deployment.
app = DeploymentEngine.bind()

# 3: Query the deployment and print the result.
if __name__ == "__main__":
    print(requests.get("http://localhost:8080/hello", params={"query_str": "Hello!"}).json())