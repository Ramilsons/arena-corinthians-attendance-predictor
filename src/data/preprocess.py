import pandas as pd;
from pathlib import Path

rootDir = Path(__file__).resolve().parent.parent.parent

def loadData():
    dataPath = rootDir / "data" / "raw" / "dataset.csv"

    return pd.read_csv(dataPath)


def saveData(trainDf, testDf):
    rootDir = Path(__file__).resolve().parent.parent.parent
    processedDataPath = rootDir / "data" / "processed"

    processedDataPath.mkdir(parents=True, exist_ok=True)

    trainPath = processedDataPath / "train.csv"
    testPath = processedDataPath / "test.csv"
    
    trainDf.to_csv(trainPath, index=False)
    testDf.to_csv(testPath, index=False)
    
    print("The data has been saved.")


# Data Cleanning
def removeUnusedColumns(dataframe):
    df = dataframe.drop(columns=['JOGO','RESULTADO', 'CORINTHIANS', 'GOL COR', 'GOL VIS', 'CAPITÃO', 'TÉCNICO', 'TÉCNICO - VISITANTE', 'RENDA', 'CAMISA', 'SAÍDA DE JOGO', '1 TEMPO', '2 TEMPO', 'ARTILHEIROS','ARTILHEIROS - VISITANTE', 'NUM-GOLS', 'PRIMEIRO GOL', 'GOL-SUL', 'GOL-NORTE', 'GOL VIS-SUL', 'GOL VIS-NORTE', 'GOL-1T', 'GOL-2T', 'GOL VIS-1T', 'GOL VIS-2T', 'ÁRBITRO', 'UF-ÁRBITRO', 'VAR'])
    return df


def cleanData(dataframe):
    cleaned = removeUnusedColumns(dataframe)
    return cleaned


# Feature Engineering 
def createIsClassicFeature(dataframe):
    rivals = ["SÃO PAULO", "SANTOS", "PALMEIRAS"]
    dataframe['IS_CLASSIC'] = dataframe['VISITANTE'].isin(rivals)

    return dataframe


def createIsWeekendFeature(dataframe):
    weekendDays = ["DOM", "SÁB"]
    dataframe['IS_WEEKEND'] = dataframe['DIA-SEMANA'].isin(weekendDays)

    return dataframe


def createFullDateFeature(dataframe):
    years = dataframe['ANO'].astype(str).apply(lambda x: '20' + x if len(x) == 2 else x)
    dataframe['FULL_DATE'] = pd.to_datetime(
        years + '-' + 
        dataframe['MES'].astype(str).str.zfill(2) + '-' + 
        dataframe['DIA'].astype(str).str.zfill(2), 
        errors='coerce'
    )

    # Order by chronological
    dataframe = dataframe.sort_values('FULL_DATE').reset_index(drop=True)
    return dataframe


def createIsPandemicPeriodFeature(dataframe):
    startPandemicLimit = pd.to_datetime('2020-02-26')
    endPandemicLimit = pd.to_datetime('2021-10-30')

    dataframe['IS_PANDEMIC_PERIOD'] = (dataframe['FULL_DATE'] >= startPandemicLimit) & (dataframe['FULL_DATE'] <= endPandemicLimit)
    return dataframe


def createSplit(dataframe):
    totalRows = len(dataframe)
    splitIndex = int(totalRows * 0.8) # Equals 80%

    trainDf = dataframe.iloc[:splitIndex].copy()
    testDf = dataframe.iloc[splitIndex:].copy()

    return trainDf, testDf


def createTargetEncondingOnTeamsNamesFeature(trainDf, testDf):
    targetEncondingMap = trainDf.groupby('VISITANTE')['PUBLICO PAGANTE'].mean().to_dict()
    global_mean = trainDf['PUBLICO PAGANTE'].mean()

    trainDf['VISITANTE_ENCODED'] = trainDf['VISITANTE'].map(targetEncondingMap).fillna(global_mean)
    testDf['VISITANTE_ENCODED'] = testDf['VISITANTE'].map(targetEncondingMap).fillna(global_mean)

    return trainDf, testDf

def featEng(dataframe):
    featEngApplied = createIsClassicFeature(dataframe)
    featEngApplied = createIsWeekendFeature(featEngApplied)
    featEngApplied = createFullDateFeature(featEngApplied)
    featEngApplied = createIsPandemicPeriodFeature(featEngApplied)

    trainDf, testDF = createSplit(featEngApplied)

    return trainDf, testDF



df = loadData()
dfCleaned = cleanData(df)
trainDfAfterFeatEng, testDfAfterFeatEng = featEng(dfCleaned)

saveData(trainDfAfterFeatEng, testDfAfterFeatEng)

