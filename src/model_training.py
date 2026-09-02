import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import  ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score

from xgboost import XGBClassifier

# ensure that an mlflow server is running before executing this script
# mlflow server -p 5001
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("flight_delay_base")

df = pd.read_csv("data/processed/flight_weather_data_2025.csv")

X = df.drop("DELAY", axis=1)
y = df["DELAY"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)

categorical_columns = X_train.select_dtypes(include=["object"]).columns.to_list()
numeric_columns = X_train.select_dtypes(include=["int","float"]).columns.to_list()


cat_pipeline = Pipeline([("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=True))]) 
num_pipeline = Pipeline([("scale", StandardScaler())])

preprocessing = ColumnTransformer([
    ("num", num_pipeline, numeric_columns),
    ("cat", cat_pipeline, categorical_columns),
])

# hyperparameters selected via random search CV
# see the notebook Weather Models for full results 
model = make_pipeline(
    preprocessing,
    XGBClassifier(
        random_state=42,
        n_jobs=1,
        eval_metric="logloss",
        n_estimators=876,
        max_depth=7,
        learning_rate=0.246599,
        subsample=0.6063865,
        colsample_bytree=0.60530598,
        min_child_weight=2
    )
)

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="xgboost_model"):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
    }

    mlflow.log_metrics(metrics)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")


joblib.dump(model, "models/xgboost.joblib")
