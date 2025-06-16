import pickle 

from flask import Flask, request, jsonify


with open("lin_reg.bin", "rb") as f_in:
    (dv, lr) = pickle.load(f_in)  # this returns both dict vectorizer and Linear Reg model 


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])  # to use this feature as string
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    preds = lr.predict(X)
    return preds 


# initializing Flask app 
app = Flask("trip-duration-prediction")


@app.route("/predict", methods=["POST"])  # to post the output in to endpoint
def predict_endpoint():

    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        "duration": float(pred[0])
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)