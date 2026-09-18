import mlflow
from sklearn.ensemble import RandomForestRegressor
from utils import loadProcessedData, evaluateModel

def main():
    XTrain, yTrain, XTest, yTest = loadProcessedData()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("arena-corinthians-attendance-baseline")

    n_estimators = 100
    max_depth = 5

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("nax_depth", max_depth)

        # Training Model
        model = RandomForestRegressor(
            n_estimators = n_estimators,
            max_depth = max_depth,
            random_state = 42
        )

        model.fit(XTrain, yTrain)

        predictions = model.predict(XTest)
        metrics = evaluateModel(yTest, predictions)

        print("Random Forest Metrics:")
        for metricName, value in metrics.items():
            print(f"{metricName.upper()}: {value:.2f}")
            mlflow.log_metric(metricName, value)

        # Saving model as Artifact
        mlflow.sklearn.log_model(model, name = "random-forest-model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        print("Training has been saved into MLFow")

if __name__ == "__main__":
    main()
