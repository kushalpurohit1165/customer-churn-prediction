# Viva Questions and Answers - Customer Churn Prediction Project

(Numbers quoted are from the actual run on the 240-customer test set. If you re-train, check `outputs/reports/model_metrics.csv`.)

## A. Machine Learning basics

**1. What is Machine Learning?**
It is a way of letting a computer learn patterns from data instead of writing fixed rules. The program improves its predictions by learning from examples.

**2. What is supervised learning?**
Learning from data that has the correct answer (label) for every example. Here every customer has a `Churn` label (Yes/No), and the model learns the link between customer features and that label.

**3. What is classification? How is it different from regression?**
Classification predicts a category (Churn = Yes or No); regression predicts a number (for example monthly charges). Our problem is binary classification.

**4. What are features and the target variable in your project?**
Features are the input columns (tenure, Contract, MonthlyCharges, etc.). The target is `Churn`. `customerID` is neither - it is only an identifier.

**5. What is overfitting and underfitting?**
Overfitting: the model memorises the training data and performs poorly on new data (high training score, low test score). Underfitting: the model is too simple and performs poorly on both. We reduce overfitting with a held-out test set, cross-validation, and limits such as `max_depth` and `min_samples_leaf`.

## B. Customer churn and the dataset

**6. What is customer churn and why does it matter?**
Churn means a customer stops using the company's service. Winning a new customer usually costs more than keeping an existing one, so predicting churn lets the company act early.

**7. Describe your dataset.**
`telco_customer_churn_synthetic.csv`: 1,200 customers, 21 columns (customerID + 19 features + Churn). There are 0 missing values and 0 duplicates. Churn = Yes for 351 customers (29.25%) and No for 849 (70.75%).

**8. Is the dataset real?**
No, it is a synthetic educational dataset, and I state this in the report and the dashboard. The results show the method works; they do not describe a real company.

**9. What is class imbalance and is it present here?**
When one class is much rarer than the other. Churners are about 29%, so it is moderately imbalanced. That is why accuracy alone is misleading, and why I used a stratified split and `class_weight="balanced"` where supported.

**10. Which features showed the strongest link with churn?**
Contract type (month-to-month 44.9% churn vs 9.2% for two-year), tenure (about 50% churn in the first 12 months) and monthly charges. The Random Forest importance ranking agrees: Contract = Month-to-month, tenure, Contract = Two year, TotalCharges.

## C. Data preprocessing

**11. Why did you remove customerID?**
It is a unique label with no real meaning. If used, models could memorise it and overfit, and it would not help for new customers.

**12. What preprocessing steps did you apply?**
Inspection, missing/duplicate checks, removing customerID, one-hot encoding of categorical columns, scaling of numeric columns, stratified train-test split, all inside a Scikit-learn Pipeline.

**13. What is encoding? Why one-hot encoding?**
Models need numbers, so text categories are converted. One-hot creates one 0/1 column per category (e.g. `Contract_Two year`). It avoids implying a false order such as Month-to-month < One year < Two year.

**14. What is scaling and why is it needed?**
Scaling puts numeric features on a similar range (StandardScaler: mean 0, standard deviation 1). Tenure (0-72) and TotalCharges (0-6,886) have very different ranges, which would dominate distance-based models like KNN and SVM.

**15. Do tree models need scaling?**
No, trees split on thresholds and are not affected by scale. Scaling is applied to all models in one pipeline for consistency, and it does not harm trees.

**16. What is data leakage and how did you prevent it?**
Leakage is when information from the test set influences training (for example, computing the scaler mean on the full dataset). I put the ColumnTransformer inside the Pipeline, so it is fitted only on the training data, and the test data is only transformed.

**17. How did you handle missing values?**
The dataset has none. The pipeline still contains imputers (median for numbers, most-frequent for categories) fitted on training data, so the app will not crash if a blank value appears later.

## D. Train-test split and tuning

**18. What is train-test split and why stratify?**
We train on one part (80% = 960 rows) and test on unseen data (20% = 240 rows). Stratify keeps the same churn share in both (29.27% train, 29.17% test), which matters for an imbalanced target.

**19. Why `random_state = 42`?**
To make the split reproducible, so everyone gets the same results. 42 is only a conventional number.

**20. What is cross-validation and GridSearchCV?**
Cross-validation splits the training data into folds (5 here), trains on 4 and validates on 1, and repeats. GridSearchCV tries every combination in a small parameter grid and keeps the best by average F1. The test set is never used for this.

## E. Algorithms

**21. How does Logistic Regression work?**
It computes a weighted sum of the features and passes it through the sigmoid function to give a probability between 0 and 1. If the probability is 0.5 or more, the prediction is Yes. Despite the name, it is a classification algorithm. Our run: F1 0.595, ROC-AUC 0.781 (the highest AUC).

**22. How does KNN work and why did it perform worst?**
It finds the k most similar customers and takes a majority vote. It struggles with many one-hot columns (distances become less meaningful) and with imbalance, because neighbours are mostly non-churners. Result: recall only 0.229 and F1 0.281.

**23. How does Naive Bayes work?**
It uses Bayes' theorem to compute the probability of each class given the features, assuming features are independent. It is fast but the independence assumption is unrealistic (e.g. tenure and TotalCharges are correlated), and its probabilities can be over-confident.

**24. How does a Decision Tree work?**
It repeatedly splits the data by the feature question that best separates the classes (using Gini impurity), forming if-then rules. It is easy to explain but can overfit, so I limited the depth (max_depth = 3).

**25. How does Random Forest work and why is it usually better than one tree?**
It trains many trees on random samples of rows and features and averages their votes. Averaging reduces variance and overfitting. It was my best model by F1 (0.628) with recall 0.857.

**26. How does SVM work?**
It finds the boundary (hyperplane) with the maximum margin between the two classes; with the RBF kernel it can form curved boundaries. It needs scaled data. I set `probability=True` so it can output churn probabilities.

**27. What are hyper-parameters? Give examples.**
Settings chosen before training, not learned from data: `C` in Logistic Regression/SVM, `n_neighbors` in KNN, `max_depth` in trees.

**28. What does `class_weight="balanced"` do?**
It gives more weight to mistakes on the minority class (churners), so the model does not just favour "No". It is used for Logistic Regression, Decision Tree, Random Forest and SVM; KNN and Naive Bayes do not support it.

## F. Evaluation metrics

**29. What is a confusion matrix?**
A table of actual vs predicted classes: True Negatives, False Positives, False Negatives, True Positives. For Random Forest: TN = 109, FP = 61, FN = 10, TP = 60.

**30. Define accuracy. Why is it not enough here?**
Accuracy = correct predictions / all predictions (0.704 for Random Forest). With 71% non-churners, predicting "No" for everyone gives about 71% accuracy and finds zero churners.

**31. What is precision?**
Of the customers predicted to churn, the share who really churn: TP / (TP + FP) = 60 / 121 = 0.496. Low precision means many wasted retention offers.

**32. What is recall?**
Of the customers who really churn, the share we catch: TP / (TP + FN) = 60 / 70 = 0.857. High recall means few missed churners.

**33. What is F1-score?**
The harmonic mean of precision and recall: 2PR / (P + R). It is high only when both are reasonably high. I used it to choose the best model.

**34. Which matters more in churn prediction - precision or recall?**
Usually recall, because losing a customer costs more than a discount offered by mistake. But too many false alarms waste money, so F1 (balance) is used, and the business can adjust the threshold according to the cost of an offer.

**35. What is ROC-AUC?**
The ROC curve plots recall (true positive rate) against false positive rate at all thresholds. AUC is the area under it: 0.5 = random guessing, 1.0 = perfect. Random Forest got 0.773. It measures ranking quality independent of one threshold.

**36. Why did you choose Random Forest as best if Logistic Regression has a higher ROC-AUC?**
My rule was the highest F1 for the churn class (ROC-AUC as tie-breaker). Random Forest had the best F1 (0.628) and the best recall. I also say honestly that the top models are close and the test set is small (240), so the difference may be chance.

**37. What would you do if the company wants fewer false alarms?**
Raise the decision threshold above 0.5 (or choose the model with higher precision). This lowers recall, so the threshold should be chosen using the cost of an offer versus the value of a retained customer.

## G. Dashboard and prediction

**38. What is Streamlit?**
An open-source Python library for building web apps for data projects using only Python, without writing HTML/JavaScript. Every time the user interacts, the script reruns.

**39. How does the prediction page work?**
The form values are validated and put into a one-row table with the same columns as training. The saved Pipeline applies the same encoding and scaling, then `predict` gives Yes/No and `predict_proba` gives the churn probability, which is mapped to Low (< 30%), Medium (30-60%) or High (>= 60%).

**40. Why save the whole Pipeline instead of just the model?**
So that new data gets exactly the same preprocessing as the training data. Saving only the model would risk mismatched encoding or scaling.

**41. Is the predicted probability a guarantee?**
No. It is a statistical estimate learned from past (here synthetic) data, and the dashboard says so.

**42. How are rule-based recommendations different from ML predictions?**
Rules are fixed if-then statements written by people (e.g. month-to-month contract -> offer a long-term discount). ML predictions are learned from data. The dashboard labels them separately, and the Recommendations page shows the actual churn rate in each rule group to check them against the data.

**43. Which rule is best supported by the data?**
Low tenure (51.0% churn) and month-to-month contracts (44.9%), compared with 29.2% overall. The rules for high charges (31.0%) and fibre with high charges (31.2%) are only weakly supported.

**44. How does the app handle invalid input?**
The `prepare_input` function checks for negative or non-numeric values, unknown categories and impossible combinations (for example, no internet but online security = Yes is corrected with a notice). Errors are shown as friendly messages instead of crashing.

## H. Limitations and extensions

**45. What are the limitations of your project?**
Synthetic dataset; small test set (240); precision about 0.50; default threshold 0.5; associations are not causes; no automatic retraining.

**46. How can you improve the model?**
Use real and larger data, try gradient boosting (XGBoost/LightGBM), calibrate probabilities, tune the threshold using business cost, use SMOTE or cost-sensitive learning, and explain predictions with SHAP.

**47. Why compare several models instead of just using Random Forest?**
No model is best for every dataset. Comparing under the same split and tuning gives evidence for the choice, and simple models (Logistic Regression) can be preferred when interpretability matters.

**48. Was the test set used to tune anything?**
No. Hyper-parameters were chosen with cross-validation on the training data only; the test set was used once for the final evaluation.
