import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

targetName = "PUBLICO PAGANTE"

def loadProcessedData():
    rootDir = Path(__file__).resolve().parent.parent.parent
    trainPath = rootDir / "data" / "processed" / "train.csv"
    testPath = rootDir / "data" / "processed" / "test.csv"

    trainDf = pd.read_csv(trainPath)
    testDf = pd.read_csv(testPath)

    XTrain = trainDf.drop(targetName, axis = 1)
    yTrain = trainDf[targetName]

    XTest = testDf.drop(targetName, axis = 1)
    yTest = testDf[targetName]

    return XTrain, yTrain, XTest, yTest

def evaluateModel(yTrue, yPred):
    mae = mean_absolute_error(yTrue, yPred)
    rmse = root_mean_squared_error(yTrue, yPred)
    r2 = r2_score(yTrue, yPred)

    print("Primeiras previsões:", yPred[:5])
    print("Primeiros valores reais:", yTrue.values[:5])

    return { "mae": mae, "rmse": rmse, "r2": r2 }