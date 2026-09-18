import mlflow
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from utils import loadProcessedData, evaluateModel

def main():
    XTrain, yTrain, XTest, yTest = loadProcessedData()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("arena-corinthians-attendance-baseline")

    baseModel = XGBRegressor(random_state = 42, n_jobs = -1)

    paramsDistributions = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 8],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    randomSearch = RandomizedSearchCV(
        estimator = baseModel,
        param_distributions = paramsDistributions,
        n_iter = 20,
        cv = 5,
        scoring = 'r2',
        random_state = 42,
        n_jobs = -1
    )

    randomSearch.fit(XTrain, yTrain)

    bestModel = randomSearch.best_estimator_
    bestParams = randomSearch.best_params_

    print("Best Params: ", bestParams)


    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBRegressor-Tuned")
        for paramName, paramValue in bestParams.items():
            mlflow.log_param(paramName, paramValue)

        predictions = bestModel.predict(XTest)
        metrics = evaluateModel(yTest, predictions)

        print("XGBoost Tuned Metrics:")
        for metricName, value in metrics.items():
            print(f"{metricName.upper()}: {value:.2f}")
            mlflow.log_metric(metricName, value)

        # Saving model as Artifact
        mlflow.sklearn.log_model(
            bestModel, 
            artifact_path="xgboost-regressor-tuned", 
            skops_trusted_types=[
                "xgboost.sklearn.XGBRegressor", 
                "xgboost.core.Booster"
            ]
        )        
        print("Training has been saved into MLFow")

if __name__ == "__main__":
    main()
