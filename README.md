## How to Read the Project

This project is organized as a step by step machine learning pipeline. Each part builds on the previous one, so it is best read in order from Part_1 to Part_5.

### Part 1 — Exploratory data analysis

This section focuses on understanding the dataset before any modeling:

* **Data Exploration:** examines distributions, feature types, and general structure of the data.
* **Null Values:** identifies missing data and explores how it is distributed.
* **Class Imbalance:** analyzes whether the target classes are balanced and how that might affect modeling.

---

### Part 2 — Preprocessing

This section prepares the data for modeling:

* handles missing values
* applies transformations (e.g., scaling, encoding)
* constructs a clean dataset suitable for machine learning

---

### Part 3 — Feature Selection

This part determines which features are most useful:

* evaluates feature importance
* removes irrelevant or redundant features
* produces a refined feature set for modeling

---

### Part 4 — Modeling

This is the main experimental section:

* **Initial Models:** builds baseline models
* **Hyperparameter Tuning:** optimizes model parameters
* **Ensembles:** combines multiple models for improved performance
* **AutoML:** explores automated model selection approaches
* **Final Selection:** compares all models and selects the best one

---

### Part 5 — Evaluation and Explainability

This section validates and interprets the final model:

* **Explainability:** analyzes which features influence predictions (e.g., SHAP)
* **Evaluation:** reports final performance metrics and model behavior

---

### Additional Notes

* MLflow is used for experiment tracking in the modeling phase.
* mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri "sqlite:///C:/Users/Hzaab/Desktop/MLSD%20project/scratch/mlflow.db"
