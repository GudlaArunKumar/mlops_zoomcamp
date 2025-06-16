from urllib import response
from flask import request_tearing_down
import predict 
import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}


# this is for testing manually without flask

# features = predict.prepare_features(ride)
# pred = predict.predict(features)
# print(pred[0])


# testigng using flask

url = "http://127.0.0.1:5000/predict"
response = requests.post(url, json=ride)
print(response.json())


