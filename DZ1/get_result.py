import requests
import json

def get_result(body):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    response = requests.post("http://127.0.0.1:8000/predict", headers=headers, data=json.dumps(body))
    return response