# Customer Churn Prediction & Retention Analytics (Machine Learning Lab)

A B.Tech IT machine learning case study: predict which telecom customers are likely to leave (Churn = Yes),
compare six ML models, and give data-driven retention advice through a Streamlit dashboard.

> **Dataset note:** `telco_customer_churn_synthetic.csv` is a **synthetic educational dataset** (1,200 customers,
> customerID + 19 features + the Churn target). Results demonstrate the method; they do not describe a real telecom company.

## 1. Project structure

```
customer_churn_project/
├── app.py                  Streamlit dashboard (6 pages)
├── train_models.py         Trains, evaluates and saves everything (run first)
├── requirements.txt
├── README.md
├── telco_customer_churn_synthetic.csv
├── .streamlit/config.toml  Theme settings
├── src/
│   ├── preprocessing.py    Load, inspect, clean, encode, scale, split
│   ├── model_training.py   6 model pipelines + GridSearchCV tuning
│   ├── evaluation.py       Metrics, best-model selection, model charts
│   ├── eda.py              Exploratory data analysis charts
│   └── prediction.py       Single-customer prediction + rule-based recommendations
├── models/                 Saved models (created by train_models.py)
├── outputs/graphs/         15 PNG charts (created by train_models.py)
├── outputs/reports/        Metrics CSV, classification reports, run summary
├── docs/                   Project report, viva Q&A, presentation content
└── screenshots/            Put your dashboard screenshots here
```

## 2. Run on Windows (PowerShell, copy-paste)

**Step 1 - Install Python.** Download Python 3.11 or 3.12 from https://www.python.org/downloads/ .
During installation **tick "Add python.exe to PATH"**. Check it:

```powershell
python --version
```

**Step 2 - Open the project in VS Code.** Unzip the project, then in VS Code use *File > Open Folder* and choose
`customer_churn_project`. Open the terminal with *Terminal > New Terminal* (make sure it is **PowerShell**).

**Step 3 - Create a virtual environment.**

```powershell
python -m venv venv
```

**Step 4 - Activate it.** You should see `(venv)` at the start of the prompt.

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell says *running scripts is disabled*, run this once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Step 5 - Install dependencies.**

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Step 6 - Train the models** (takes about 30-60 seconds; creates `models/` and `outputs/`).

```powershell
python train_models.py
```

**Step 7 - Start the dashboard.**

```powershell
streamlit run app.py
```

**Step 8 - Open the browser.** Streamlit opens it automatically. If not, go to http://localhost:8501 .
Press `Ctrl + C` in the terminal to stop the app.

## 3. Troubleshooting

| Problem | Fix |
|---|---|
| `python is not recognized` | Reinstall Python and tick *Add to PATH*, or use `py -m venv venv` instead of `python -m venv venv`. |
| `...Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`. |
| `ModuleNotFoundError: No module named 'src'` | Run commands from inside the `customer_churn_project` folder (where `app.py` is). |
| `ModuleNotFoundError: streamlit / sklearn ...` | The venv is not active or `pip install -r requirements.txt` was not run. |
| App says *Trained models not found* | Run `python train_models.py` first, then refresh the page. |
| Error loading a `.joblib` file after changing the Python / scikit-learn version | Run `python train_models.py` again to rebuild the models. |
| `Port 8501 is already in use` | `streamlit run app.py --server.port 8502` |
| `Dataset not found` | Keep `telco_customer_churn_synthetic.csv` next to `app.py`. |
| pip install fails (network) | Check internet / proxy, then re-run `pip install -r requirements.txt`. |

## 4. How the code works (simple language, useful for the viva)

1. **Load and check** (`preprocessing.py`): read the CSV, report shape, data types, missing values, duplicates and the
   churn class balance. This data has 1,200 rows, **0 missing values and 0 duplicates**, and 29.25% churners.
2. **Features and target**: `Churn` is the target (Yes = 1). `customerID` is only a label, so it is dropped.
3. **Split**: `train_test_split` with `stratify=y` (80% train = 960 rows, 20% test = 240 rows, `random_state=42`) keeps
   the same churn share in both parts. All six models use this same split.
4. **Preprocessing pipeline**: numbers (`tenure`, `MonthlyCharges`, `TotalCharges`) are scaled with `StandardScaler`;
   text categories are converted to 0/1 columns with `OneHotEncoder`; `SeniorCitizen` is already 0/1.
   This `ColumnTransformer` sits **inside** each model's `Pipeline`, so it is fitted on training data only
   (**no data leakage**) and the dashboard transforms a new customer in exactly the same way.
5. **Training** (`model_training.py`): Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest and SVM.
   `GridSearchCV` (5-fold, scoring = F1) picks hyper-parameters using the training set only.
   `class_weight="balanced"` is used where available so churners (the minority) are not ignored.
6. **Evaluation** (`evaluation.py`): Accuracy, Precision, Recall, F1 and ROC-AUC on the untouched test set
   (churn = positive class). The **best model is chosen by F1-score** (ROC-AUC as tie-breaker), not by accuracy.
7. **Saving**: the best model, all models, test predictions, metrics and charts are saved by `train_models.py`.
   The dashboard only reads these files - no numbers are typed in by hand.
8. **Dashboard** (`app.py`): Dashboard, Dataset Explorer, Churn Analysis, Model Comparison, Predict Churn and
   Retention Recommendations pages.
9. **Prediction** (`prediction.py`): validates the form input, builds a one-row table, calls `predict` and
   `predict_proba`, and maps the probability to a risk level (Low < 30%, Medium 30-60%, High >= 60%).
10. **Rule-based recommendations** are plain if-then business rules and are labelled separately from the
    ML prediction on every page.

## 5. Results from the actual run (test set of 240 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (best by F1)** | 0.704 | 0.496 | 0.857 | 0.628 | 0.773 |
| Decision Tree | 0.679 | 0.472 | 0.843 | 0.605 | 0.764 |
| SVM | 0.688 | 0.479 | 0.814 | 0.603 | 0.779 |
| Naive Bayes | 0.671 | 0.464 | 0.829 | 0.595 | 0.753 |
| Logistic Regression | 0.688 | 0.478 | 0.786 | 0.595 | 0.781 |
| KNN | 0.658 | 0.364 | 0.229 | 0.281 | 0.632 |

Random Forest catches 60 of the 70 churners in the test set (recall 0.857) but also flags 61 loyal customers
(precision about 0.50). Differences between the top five models are small on a 240-row test set; KNN is clearly the weakest.
If you re-run `train_models.py` on another machine you should get the same numbers for the same library versions.

## 6. Limitations (be ready to say these in the viva)

- The dataset is synthetic; real churn behaviour is more complex.
- Only 240 test customers, so metrics have noticeable uncertainty.
- Accuracy around 70% and precision around 0.50 mean many false alarms; the model is a screening aid, not a decision maker.
- Probabilities are estimates, not guarantees. Naive Bayes in particular tends to give over-confident probabilities.
- Rule-based recommendations R2 (high charges) and R7 (fiber + high charges) are only weakly linked to churn in this data (see the Recommendations page).
