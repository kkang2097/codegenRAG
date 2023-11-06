from rag_engine import ElliotEngine
import requests
from fastapi import FastAPI
from ray import serve

app = FastAPI()

#Class copy, use inheritance to not duplicate code.
#It's hacky, but it works.
@serve.deployment(num_replicas=1)
@serve.ingress(app)
class DeploymentEngine(ElliotEngine):
    def __init__(self):
        super().__init__()

    @app.get("/hello")
    def query_something(self, query_str: str):
        return self.query(query_str)



#Class wrappers from Ray
# 2: Deploy the deployment.
entrypoint = DeploymentEngine.bind()

#
#Command to deploy: serve run app:entrypoint --port 8080
#


# 3: Query the deployment and print the result.
#TODO: Make a script that can run this automatically later... (final thing to do)