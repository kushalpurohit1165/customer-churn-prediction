"""
model_training.py
-----------------
Builds the six classification pipelines and trains them.

Each model is a Pipeline:   preprocessor (impute + scale + one-hot)  ->  classifier
Hyper-parameters are chosen with GridSearchCV (5-fold stratified CV, scoring = F1)
on the TRAINING data only. The test set is never touched during training/tuning.
"""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor


def get_model_configs(random_state: int = 42) -> dict:
    """
    Return {model name: (estimator, parameter grid)}.
    class_weight="balanced" is used where supported so the minority class
    (churners, ~29%) is not ignored. KNN and Naive Bayes have no such option.
    """
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [5, 11, 21, 31], "model__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"model__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
            {"model__max_depth": [3, 5, 7, 10], "model__min_samples_leaf": [1, 5, 10, 20]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=random_state, n_jobs=-1),
            {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 3, 5]},
        ),
        "SVM": (
            # probability=True lets SVM output churn probabilities (needed for ROC-AUC / app)
            SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_state),
            {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01]},
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    """Preprocessing + classifier in ONE object -> training and prediction stay consistent."""
    return Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])


def train_all_models(X_train, y_train, cv_folds: int = 5, random_state: int = 42, verbose: bool = True):
    """
    Tune and train every model on the training data.
    Returns {name: {"model": fitted pipeline, "best_params": dict, "cv_f1": float}}.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = {}
    for name, (estimator, grid) in get_model_configs(random_state).items():
        if verbose:
            print(f"  Training {name} ...")
        search = GridSearchCV(build_pipeline(estimator), grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)          # refit=True -> best model refitted on full training set
        results[name] = {
            "model": search.best_estimator_,
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "cv_f1": float(search.best_score_),
        }
    return results


def save_artifact(obj, path):
    joblib.dump(obj, path)


def load_artifact(path):
    return joblib.load(path)
