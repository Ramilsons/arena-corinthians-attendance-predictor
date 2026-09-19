import os
import pickle
import joblib
import mlflow

from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from utils import loadProcessedData, evaluateModel

def main():
    XTrain, yTrain, XTest, yTest = loadProcessedData()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("arena-corinthians-attendance-baseline")

    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    }

    with mlflow.start_run():
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("learning_rate", params["learning_rate"])
        mlflow.log_param("random_state", params["random_state"])
        mlflow.log_param("nax_depth", params["max_depth"])

        # Training Model
        model = GradientBoostingRegressor(**params)
        model.fit(XTrain, yTrain)

        predictions = model.predict(XTest)
        metrics = evaluateModel(yTest, predictions)

        print("Gradient Boosting Metrics:")
        for metricName, value in metrics.items():
            print(f"{metricName.upper()}: {value:.2f}")
            mlflow.log_metric(metricName, value)

        # Saving model as Artifact
        mlflow.sklearn.log_model(model, name = "gradient-boosting-regressor", skops_trusted_types=["sklearn.tree._tree.Tree"])
        print("Training has been saved into MLFow")

        rootDir = Path(__file__).resolve().parent.parent.parent
        artifactsPath = rootDir / "artifacts" / "model.pkl"

        os.makedirs('artifacts', exist_ok = True)
        joblib.dump(model, artifactsPath)
        print("Model as PKL has been saved into Artifacts folder")

if __name__ == "__main__":
    main()
