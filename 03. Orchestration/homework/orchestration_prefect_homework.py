"""
This script is for Orchestration homework using prefect tool

Top execute this script,
1) first run the prefect server in a separate terminal
2) Open another terminal, configure the prefect server and run the actual pythin script.
"""

import pickle 
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error 
from sklearn.linear_model import LinearRegression


import mlflow 
from prefect import task, flow



@task(retries=3, retry_delay_seconds=2)
def read_dataframe(filename: str) -> pd.DataFrame:
    """
    Read the train and validation dataset and filters the dataset by duration less than or equal to 60 minutes.
    Returns the filtered dataframe
    """

    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    print(f"Dataset shape before filtering: {df.shape}")

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    # to check the data shape
    print(f"Dataset shape after filtering: {df.shape}")
    
    return df

@task()
def create_features(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')

    if dv is None:  # if dv is not passed then train on training set
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


@task(log_prints=True)  # it logs / prints all the print statement
def train_model(X_train, y_train, X_val, y_val, dv) -> None:

    with mlflow.start_run() as run:
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # to check the intercept of the model
        print("Linear regression model Intercept: ", lr.intercept_)
        
        y_pred = lr.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor") # logging dictVectorizer

        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")  # logging the xgboost model

    return


@flow()
def main_flow(
        train_path: str = r"/mnt/e/Machine_Learning_Projects/mlops_zoomcamp/03. Orchestration/homework/yellow_tripdata_2023-03.parquet",
        val_path: str = r"/mnt/e/Machine_Learning_Projects/mlops_zoomcamp/03. Orchestration/homework/yellow_tripdata_2023-03.parquet"
    ) -> None:

    # setting mlflow tracking uri 
    mlflow.set_tracking_uri("sqlite:///mlflow.db") # runs stored in local db 
    mlflow.set_experiment("nyc_taxi_orchestraction_homework_exp")

    # load the data and train the model
    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)

    X_train, dv = create_features(df_train, dv=None)
    X_val, _ = create_features(df_val, dv)

    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values

    train_model(X_train, y_train, X_val, y_val, dv) 

    return




if __name__ == "__main__":

    import prefect
    print(prefect.__version__)

    # creating models folder for saving the model 
    models_folder = Path("models")
    models_folder.mkdir(exist_ok=True)

    # run the main code
    main_flow()

    






