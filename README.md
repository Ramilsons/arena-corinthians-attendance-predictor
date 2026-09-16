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