"""
This script is an example of a workflow orchestration where each function / component is executed in 
a sequence as they define whereas it works same with orchestration tool

Top execute this script,
1) first run the server - mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 8080
2) then run the script so mlflow tracks the exp in the same server.
"""

import pickle 
from pathlib import Path

import pandas as pd
import xgboost as xgb 
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error 


import mlflow 

mlflow.set_tracking_uri("http://localhost:8080") 
mlflow.set_experiment("nyc_taxi_orchestraction_exp")


# creating models folder for saving the model 
models_folder = Path("models")
models_folder.mkdir(exist_ok=True)


# reading data by passing taxi records data with year and month as arguments 

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'  # copy address from wesbite
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_features(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:  # if dv is not passed then train on training set
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):

    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        # best params from previous week
        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=20,
            evals=[(valid, 'validation')],
            early_stopping_rounds=10
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor") # logging dictVectorizer

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")  # logging the xgboost model

        # to get some information about the mlflow run
        return run.info.run_id


def run(year, month):

    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1

    df_val = read_dataframe(year=next_year, month=next_month) # next month's data 

    X_train, dv = create_features(df_train, dv=None)
    X_val, _ = create_features(df_val, dv)

    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv) 

    return run_id




if __name__ == "__main__":

    # paramterize the arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    # just saving the run id so we can check in mlflow ui later
    with open("run_id.txt", "w") as f:
        f.write(run_id)

    






