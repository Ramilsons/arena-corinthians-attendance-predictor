import pandas as pd;
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split;

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
def createHourFeature(dataframe):
    if 'HORA' in dataframe.columns:
        dataframe['HORA_NUM'] = pd.to_datetime(dataframe['HORA'], format='%H:%M', errors='coerce').dt.hour
        
        medianHour = dataframe['HORA_NUM'].median()
        dataframe['HORA_NUM'] = dataframe['HORA_NUM'].fillna(medianHour)
        
        dataframe = dataframe.drop(columns=['HORA'])
        
    return dataframe

def createIsClassicFeature(dataframe):
    rivals = ["SÃO PAULO", "SANTOS", "PALMEIRAS"]
    dataframe['IS_CLASSIC'] = dataframe['VISITANTE'].isin(rivals).astype(int)

    return dataframe


def createIsWeekendFeature(dataframe):
    weekendDays = ["DOM", "SÁB"]
    dataframe['IS_WEEKEND'] = dataframe['DIA-SEMANA'].isin(weekendDays).astype(int)

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
    dataframe = dataframe.drop(columns="FULL_DATE")
    trainDf, testDf = train_test_split(dataframe, test_size=0.2, random_state=42, shuffle=True)
    
    return trainDf, testDf


def applyTargetEncoding(trainDf, testDf, targetCols, targetColName = 'PUBLICO PAGANTE'):
    globalMean = trainDf[targetColName].mean()

    for col in targetCols:
        if col in trainDf.columns:
            encodingMap = trainDf.groupby(col)[targetColName].mean().to_dict()

            trainDf[f'{col}_ENCODED'] = trainDf[col].map(encodingMap).fillna(globalMean)
            testDf[f'{col}_ENCODED'] = testDf[col].map(encodingMap).fillna(globalMean)

    return trainDf, testDf


def applyOneHotEncoding(trainDf, testDf, oheCols):
    existingCols = [col for col in oheCols if col in trainDf.columns]

    if existingCols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        trainOheArray = encoder.fit_transform(trainDf[existingCols])
        trainOheDf = pd.DataFrame(
            trainOheArray, 
            columns = encoder.get_feature_names_out(existingCols), 
            index = trainDf.index
        )

        testOheArray = encoder.transform(testDf[existingCols])
        testOheDf = pd.DataFrame(
            testOheArray, 
            columns=encoder.get_feature_names_out(existingCols), 
            index=testDf.index
        )

        trainDf = pd.concat([trainDf.drop(columns=existingCols), trainOheDf], axis=1)
        testDf = pd.concat([testDf.drop(columns=existingCols), testOheDf], axis=1)

    return trainDf, testDf


def createTargetEncodingOnTeamsNamesFeature(trainDf, testDf):
    targetEncondingMap = trainDf.groupby('VISITANTE')['PUBLICO PAGANTE'].mean().to_dict()
    global_mean = trainDf['PUBLICO PAGANTE'].mean()

    trainDf['VISITANTE_ENCODED'] = trainDf['VISITANTE'].map(targetEncondingMap).fillna(global_mean)
    testDf['VISITANTE_ENCODED'] = testDf['VISITANTE'].map(targetEncondingMap).fillna(global_mean)

    return trainDf, testDf


def featEng(dataframe):
    featEngApplied = createHourFeature(dataframe)
    featEngApplied = createIsClassicFeature(featEngApplied)
    featEngApplied = createIsWeekendFeature(featEngApplied)
    featEngApplied = createFullDateFeature(featEngApplied)
    featEngApplied = createIsPandemicPeriodFeature(featEngApplied)

    trainDf, testDf = createSplit(featEngApplied)

    columnsToTargetEncoding = ['CIDADE - VISITANTE', 'CAMPEONATO', 'VISITANTE']
    trainDf, testDf = applyTargetEncoding(trainDf, testDf, columnsToTargetEncoding)

    columnsToOneHotEncoding = ['UF', 'PAIS', 'DIA-SEMANA']
    trainDf, testDf = applyOneHotEncoding(trainDf, testDf, columnsToOneHotEncoding)

    return trainDf, testDf



df = loadData()
dfCleaned = cleanData(df)
trainDfAfterFeatEng, testDfAfterFeatEng = featEng(dfCleaned)

saveData(trainDfAfterFeatEng, testDfAfterFeatEng)

