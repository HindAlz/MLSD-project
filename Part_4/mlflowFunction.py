from sklearn.model_selection import StratifiedKFold
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import numpy as np


def log_model_run(model_name, model, results, method_name=None):
    mlflow.utils.logging_utils.disable_logging()

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_class", model.__class__.__name__)

        if method_name:
            mlflow.log_param("method", method_name)

        for k, v in model.get_params().items():
            mlflow.log_param(k, v)

        mlflow.log_metric("cv_accuracy", float(results["accuracy"]))
        mlflow.log_metric("cv_f1", float(results["f1"]))
        mlflow.log_metric("cv_recall", float(results["recall"]))
        mlflow.log_metric("cv_precision", float(results["precision"]))



def evaluate_model(X, y, name, model, preprocess_fn):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

    acc_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_train_fold = y.iloc[train_idx]
        y_valid_fold = y.iloc[valid_idx]

        X_train_used, X_valid_used, y_train_used, class_weight = preprocess_fn(
            X_train_fold, X_valid_fold, y_train_fold
        )

        fold_model = clone(model)

        if class_weight is not None and "class_weight" in fold_model.get_params():
            fold_model.set_params(class_weight=class_weight)

        fold_model.fit(X_train_used, y_train_used)
        y_pred = fold_model.predict(X_valid_used)

        acc_scores.append(accuracy_score(y_valid_fold, y_pred))
        f1_scores.append(f1_score(y_valid_fold, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_valid_fold, y_pred, zero_division=0))
        precision_scores.append(precision_score(y_valid_fold, y_pred, zero_division=0))

    results = {
        "accuracy": np.mean(acc_scores),
        "f1": np.mean(f1_scores),
        "recall": np.mean(recall_scores),
        "precision": np.mean(precision_scores),
    }

    print(name)
    print(
        f"Accuracy: {results['accuracy']:.4f}, "
        f"F1: {results['f1']:.4f}, "
        f"Recall: {results['recall']:.4f}, "
        f"Precision: {results['precision']:.4f}"
    )

    return results

# Load the dataset
train = pd.read_parquet('C:\\Users\\Hzaab\\Desktop\\MLSD project\\data\\preprocessed\\train.parquet')

categorical_cols = ["profile pic", "name==username", "external URL", "private"]
numeric_columns = [col for col in train.columns if col not in categorical_cols and col != "fake"]



# Preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_columns),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="drop",
)

def apply_no_sampling(X_train_fold, X_valid_fold, y_train_fold):
    X_train_processed = preprocessor.fit_transform(X_train_fold)
    X_valid_processed = preprocessor.transform(X_valid_fold)
    return X_train_processed, X_valid_processed, y_train_fold, None

def apply_class_weight_balanced(X_train_fold, X_valid_fold, y_train_fold):
    X_train_processed = preprocessor.fit_transform(X_train_fold)
    X_valid_processed = preprocessor.transform(X_valid_fold)
    return X_train_processed, X_valid_processed, y_train_fold, "balanced"


def apply_random_oversample(X_train_fold, X_valid_fold, y_train_fold):
    X_train_processed = preprocessor.fit_transform(X_train_fold)
    X_valid_processed = preprocessor.transform(X_valid_fold)

    sampler = RandomOverSampler(random_state=10)
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_processed, y_train_fold)

    return X_train_resampled, X_valid_processed, y_train_resampled, None


def apply_smote(X_train_fold, X_valid_fold, y_train_fold):
    X_train_processed = preprocessor.fit_transform(X_train_fold)
    X_valid_processed = preprocessor.transform(X_valid_fold)

    sampler = SMOTE(random_state=10)
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_processed, y_train_fold)

    return X_train_resampled, X_valid_processed, y_train_resampled, None


def apply_random_undersample(X_train_fold, X_valid_fold, y_train_fold):
    X_train_processed = preprocessor.fit_transform(X_train_fold)
    X_valid_processed = preprocessor.transform(X_valid_fold)

    sampler = RandomUnderSampler(random_state=10)
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_processed, y_train_fold)

    return X_train_resampled, X_valid_processed, y_train_resampled, None



def make_preprocessor(X):
    categorical_base = ["profile pic", "name==username", "external URL", "private"]

    categorical_cols = [col for col in categorical_base if col in X.columns]
    numeric_columns = [col for col in X.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

def apply_no_sampling2(X_train_fold, X_valid_fold, y_train_fold):
    preprocessor = make_preprocessor(X_train_fold)

    X_train_processed = preprocessor.fit_transform(X_train_fold)
    X_valid_processed = preprocessor.transform(X_valid_fold)

    return X_train_processed, X_valid_processed, y_train_fold, None