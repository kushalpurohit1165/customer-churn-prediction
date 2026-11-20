# Presentation Content (12 slides)

*(Numbers are from the actual run. Add screenshots from `screenshots/` and charts from `outputs/graphs/`.)*

## Slide 1 - Title
**Customer Churn Prediction & Retention Analytics Using Machine Learning**
Machine Learning Lab · B.Tech Information Technology
Name · Roll No. · Guide · Date
*(Note: dataset is synthetic and educational.)*

## Slide 2 - Introduction
- Churn = customers leaving the telecom service
- Keeping customers is cheaper than finding new ones
- Machine Learning can estimate who is likely to leave
- This project: supervised binary classification + Streamlit dashboard

## Slide 3 - Industry Problem
- Retention budgets are limited - you cannot target everyone
- Need to know: who is at risk, why, and what to offer
- Goal: predict churn, estimate risk (Low / Medium / High), recommend actions

## Slide 4 - Objectives
1. Analyse customer behaviour and churn patterns
2. Build a leak-free preprocessing pipeline
3. Train and compare 6 ML models fairly
4. Evaluate using accuracy, precision, recall, F1, ROC-AUC
5. Build an interactive dashboard with prediction
6. Provide rule-based retention recommendations

## Slide 5 - Dataset
- `telco_customer_churn_synthetic.csv` - **synthetic, educational**
- 1,200 customers · 19 features + customerID + target `Churn`
- 0 missing values, 0 duplicates
- Churn: Yes 351 (29.25%) · No 849 (70.75%) -> imbalanced
- customerID excluded from features
- Visual: churn distribution chart (`01_churn_distribution.png`)

## Slide 6 - Methodology
Load -> Inspect -> Clean -> Stratified 80/20 split -> Pipeline (scale + one-hot + model)
-> GridSearchCV (5-fold, F1) on training data -> Evaluate on 240 unseen customers -> Save -> Dashboard
- Key point: preprocessing fitted on training data only (no leakage)

## Slide 7 - ML Algorithms
Logistic Regression · KNN · Naive Bayes · Decision Tree · Random Forest · SVM
- One line each: linear probability model / nearest-neighbour vote / Bayes with independence / if-then splits / many trees voting / maximum-margin boundary
- `class_weight="balanced"` where supported

## Slide 8 - Dashboard Features
- Dashboard: KPI cards and trend charts
- Dataset Explorer: search, filters, missing values, dtypes, statistics, CSV download
- Churn Analysis: contract, tenure, charges, payment, internet, demographics with filters
- Model Comparison: metrics, confusion matrix, ROC, classification report
- Predict Churn: form -> Yes/No + probability + risk + recommendations
- Retention Recommendations: rule-based, checked against data
- Visual: dashboard screenshots

## Slide 9 - Results and Comparison
| Model | Acc | Prec | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Random Forest | 0.704 | 0.496 | 0.857 | 0.628 | 0.773 |
| Decision Tree | 0.679 | 0.472 | 0.843 | 0.605 | 0.764 |
| SVM | 0.688 | 0.479 | 0.814 | 0.603 | 0.779 |
| Naive Bayes | 0.671 | 0.464 | 0.829 | 0.595 | 0.753 |
| Logistic Reg. | 0.688 | 0.478 | 0.786 | 0.595 | 0.781 |
| KNN | 0.658 | 0.364 | 0.229 | 0.281 | 0.632 |
- Best by F1: **Random Forest** (catches 60 of 70 churners; 61 false alarms)
- Accuracy alone is misleading (always-"No" gives ~71%)
- Top features: Month-to-month contract, tenure, Two-year contract, TotalCharges

## Slide 10 - Sample Prediction
- Use a screenshot of the Predict page. Example idea: month-to-month contract, 2 months tenure, electronic check
- Show: predicted label, churn probability, risk category
- Show rule-based recommendations (month-to-month, low tenure, electronic check) labelled "rule-based"
- Say: probability is an estimate, not a guarantee

## Slide 11 - Limitations and Future Scope
**Limitations:** synthetic data · only 240 test customers · precision about 0.50 · default 0.5 threshold · association, not causation
**Future scope:** real/larger data · threshold tuning by business cost · gradient boosting · SMOTE · SHAP explanations · A/B test retention offers · cloud deployment

## Slide 12 - Conclusion
- Contract type, tenure and charges are the main churn indicators in this data
- Random Forest gave the best F1 (0.628) and recall (0.857)
- Dashboard turns the model into a usable tool for retention teams
- Results demonstrate the method on a synthetic dataset
- Thank you / Questions
