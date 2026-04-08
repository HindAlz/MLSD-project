from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

def log_model_run(model_name, model, results, X, y):
    mlflow.utils.logging_utils.disable_logging()
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_class", model.__class__.__name__)

        for k, v in model.get_params().items():
            mlflow.log_param(k, v)

        mlflow.log_metric("cv_accuracy", float(results["accuracy"]))
        mlflow.log_metric("cv_f1", float(results["f1"]))
        mlflow.log_metric("cv_recall", float(results["recall"]))
        mlflow.log_metric("cv_precision", float(results["precision"]))

        model.fit(X, y)

        if isinstance(model, XGBClassifier):
            mlflow.xgboost.log_model(model, name="model")
        elif isinstance(model, LGBMClassifier):
            mlflow.lightgbm.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(model, name="model")


#import mlflow
#from mlflowFunction import log_model_run

#mlflow.set_tracking_uri("http://127.0.0.1:5000")
#print("Tracking URI:", mlflow.get_tracking_uri())
#mlflow.set_experiment("Model Development")
#import mlflow
#mlflow.utils.logging_utils.disable_logging()