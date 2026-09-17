# Arena Corinthians Match Attendance Predictor 🏟️⚽
Machine learning model and data pipeline for forecasting football match attendance at Corinthians' Neo Química Arena.

## 🎯 Business Objective
Accurately predicting match attendance enables stadium management and operations to:
* **Optimize Services:** Ensure appropriate staffing for food, beverage, and ticketing operations.
* **Enhance Operations & Security:** Improve crowd control, logistics, and stadium safety.
* **Cost Efficiency:** Reduce operational waste by aligning resources directly with expected crowd sizes.

---

## 📊 Dataset & Credits
* **Source:** [Kaggle - Arena Corinthians Dataset](https://www.kaggle.com/datasets/danilosoares/arena-corinthians)
* **Creator:** Originally compiled and curated by **Timão Dados @TimaoDados** (Twitter/X profile).
* **Data Version Control (DVC):** Raw and processed datasets are version-controlled using DVC to ensure reproducibility.

---

## 🛠️ Tech Stack & Tools
* **Programming Language:** Python
* **Data Manipulation & Modeling:** Pandas, Scikit-Learn, NumPy
* **Data Versioning:** DVC (Data Version Control)
* **Experiment Tracking:** MLflow
* **Containerization & Deployment:** Docker, AWS EC2, FastAPI

---

## 🏗️ Project Architecture
```text
arena-corinthians-attendance-predictor/
│
├── data/
│   ├── raw/                 # Raw dataset from Kaggle (tracked by DVC)
│   └── processed/           # Cleaned data and feature engineered sets
│
├── notebooks/               # Exploratory Data Analysis (EDA)
│
├── src/                     # Modularized production source code
│   ├── preprocessing.py
│   ├── train.py
│   └── main.py
│
├── artifacts/               # Saved model binaries (.pkl) and encoders
├── mlruns/                  # Local MLflow experiment tracking logs
├── requirements.txt         # Project dependencies
└── README.md
```

---

### Exploratory Data Analysis (EDA)

In `notebooks/01-eda.ipynb`, we analyze the relationship and temporal evolution of the mean attendance by year at the Neo Química Arena. 

A sharp drop in attendance is clearly visible during **2020 and 2021**, directly resulting from stadium access restrictions and public health protocols during the COVID-19 pandemic (spanning from the match on February 26, 2020, up to the return of fans on November 1, 2021).

| Mean Attendance by Year |
| :----------------------: |
| ![Mean Attendance by Year Graphic](./notebooks/outputs/mean-attendance-by-year.png) |

[Reference](https://www.tudotimao.com.br/noticia/159661/veja-como-a-torcida-viveu-o-corinthians-no-periodo-de-portoes-fechados)
[Reference](https://www.meutimao.com.br/jogo/5930/brasileirao-2021/corinthians-1-x-0-chapecoense)