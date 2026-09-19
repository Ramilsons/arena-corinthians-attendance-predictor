import joblib
import pandas as pd

from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

app = FastAPI()
templates = Jinja2Templates(directory="api/templates")

def loadModel():
    rootDir = Path(__file__).resolve().parent.parent
    modelPath = rootDir / "artifacts" / "model.pkl"

    model = None

    if modelPath.exists():
        try:
            model = joblib.load(modelPath)
            print('Model has been loaded')
        except Exception as e:
            print('Error trying to load model: ', e)
    
    return model

def loadDataProcessed():
    rootDir = Path(__file__).resolve().parent.parent
    dataProcessedPath = rootDir / "data" / "processed" /"test.csv"

    df = pd.read_csv(dataProcessedPath)
    return df

def executeModel(request, model, dayForm, monthForm, yearForm, dayOfWeekForm, hourForm, visitantNameForm, visitantStateForm):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found.")

    try:
        if hasattr(model, "feature_names_in_"):
            # All columns initialize with default value 0
            inputDf = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)

            if "DIA" in inputDf.columns: inputDf["DIA"] = dayForm
            if "MES" in inputDf.columns: inputDf["MES"] = monthForm
            if "ANO" in inputDf.columns: inputDf["ANO"] = yearForm
            if "HORA_NUM" in inputDf.columns: inputDf["HORA_NUM"] = hourForm

            if "IS_PANDEMIC_PERIOD" in inputDf.columns: inputDf["IS_PANDEMIC_PERIOD"] = 0

            dataProcessedReference = loadDataProcessed()
            targetEncodingColumns = ["VISITANTE_ENCODED", "CIDADE - VISITANTE_ENCODED"]

            result = (dataProcessedReference.loc[dataProcessedReference["VISITANTE"] == visitantNameForm.upper(), targetEncodingColumns].head(1))

            if "VISITANTE_ENCODED" in inputDf.columns: inputDf["VISITANTE_ENCODED"] = result["VISITANTE_ENCODED"].item()
            if "CIDADE - VISITANTE_ENCODED" in inputDf.columns: inputDf["CIDADE - VISITANTE_ENCODED"] = result["CIDADE - VISITANTE_ENCODED"].item()
            
            if "CAMPEONATO_ENCODED" in inputDf.columns: inputDf["CAMPEONATO_ENCODED"] = 30.27025287356322
            if "PAIS_BRA" in inputDf.columns: inputDf["PAIS_BRA"] = 1

            stateColumn = f"UF_{visitantStateForm.upper()}"
            if stateColumn in inputDf.columns:
                inputDf[stateColumn] = 1

            dayOfWeekColumn = f"DIA-SEMANA_{dayOfWeekForm.upper()}"
            if dayOfWeekColumn in inputDf.columns:
                inputDf[dayOfWeekColumn] = 1

            isWeekend = 0
            if dayOfWeekForm == "SÁB" or dayOfWeekForm == "DOM":
                isWeekend = 1

            isClassic = 0
            if  visitantNameForm == "SÃO PAULO" or visitantNameForm == "SANTOS" or visitantNameForm == "PALMEIRAS":
                isClassic = 1

            if "IS_WEEKEND" in inputDf.columns: inputDf["IS_WEEKEND"] = isWeekend
            if "IS_CLASSIC" in inputDf.columns: inputDf["IS_CLASSIC"] = isClassic
    

            prediction = model.predict(inputDf)[0]

            comparison_df = pd.DataFrame({
                'Column_Expected': model.feature_names_in_,
                'Valor_Shipped': [inputDf[col].iloc[0] if col in inputDf.columns else 'AUSENTE' for col in model.feature_names_in_]
            })

            print("\n--- SIDE-BY-SIDE COMPARISON: EXPECTED VS. SHIPPED ---")
            print(comparison_df.to_string())
            print("---------------------------------------------------\n")

            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "message": "Previsão de Público Neo Química Arena",
                    "prediction": f"{float(prediction):.3f}"
                }
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error executing prediction: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={"message": "Previsão de Público Neo Química Arena"}
    )

@app.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "message": "Service is up and running"}
    )

@app.post("/predict")
async def predict(
    request: Request,
    day: int = Form(...),
    month: int = Form(...),
    year: int = Form(...),
    day_of_week: str = Form(...),
    hour_num: int = Form(...),
    visitant: str = Form(...),
    visit_state: str = Form(0),
):
    modelLoaded = loadModel()
    return executeModel(request, modelLoaded, day, month, year, day_of_week, hour_num, visitant, visit_state)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)