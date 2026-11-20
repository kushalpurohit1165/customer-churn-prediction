"""
prediction.py
-------------
(1) Predict churn for ONE new customer using a saved pipeline.
(2) Rule-based retention recommendations (business rules - NOT machine learning).

The saved model is a full Pipeline (preprocessing + classifier), so a new customer is
transformed in exactly the same way as the training data.
"""
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS

# Probability cut-offs used for the risk category
LOW_RISK_MAX = 0.30
HIGH_RISK_MIN = 0.60

HIGH_CHARGE = 80.0      # monthly charges considered "high"
LOW_TENURE = 12         # months considered "new customer"

INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]

VALID_VALUES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}


# ----------------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------------
def prepare_input(inputs: dict):
    """
    Validate and normalise the form values. Returns (clean_dict, list_of_notes).
    Raises ValueError for invalid values.
    """
    data = dict(inputs)
    notes = []

    missing = [c for c in FEATURE_COLUMNS if c not in data]
    if missing:
        raise ValueError(f"Missing input fields: {', '.join(missing)}")

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        try:
            data[col] = float(data[col])
        except (TypeError, ValueError):
            raise ValueError(f"{col} must be a number.")
        if data[col] < 0:
            raise ValueError(f"{col} cannot be negative.")
    data["tenure"] = int(round(data["tenure"]))
    if data["tenure"] > 120:
        raise ValueError("Tenure above 120 months looks unrealistic.")
    if data["MonthlyCharges"] == 0:
        raise ValueError("Monthly charges must be greater than 0.")

    for col, allowed in VALID_VALUES.items():
        if data[col] not in allowed:
            raise ValueError(f"Invalid value '{data[col]}' for {col}.")

    # Keep dependent fields consistent with how the dataset is structured
    if data["PhoneService"] == "No" and data["MultipleLines"] != "No phone service":
        data["MultipleLines"] = "No phone service"
        notes.append("Multiple lines was set to 'No phone service' because the customer has no phone service.")
    if data["PhoneService"] == "Yes" and data["MultipleLines"] == "No phone service":
        data["MultipleLines"] = "No"
        notes.append("Multiple lines was set to 'No' because the customer has phone service.")
    if data["InternetService"] == "No":
        changed = [c for c in INTERNET_ADDONS if data[c] != "No internet service"]
        for c in INTERNET_ADDONS:
            data[c] = "No internet service"
        if changed:
            notes.append("Internet add-on services were set to 'No internet service' because the customer has no internet.")
    else:
        for c in INTERNET_ADDONS:
            if data[c] == "No internet service":
                data[c] = "No"
                notes.append(f"{c} was set to 'No' because the customer has internet service.")

    if data["tenure"] == 0 and data["TotalCharges"] > 0:
        notes.append("Tenure is 0 months but total charges are above 0 - please double-check the values.")
    elif data["tenure"] > 0:
        expected = data["tenure"] * data["MonthlyCharges"]
        if data["TotalCharges"] > 1.6 * expected + 100 or data["TotalCharges"] < 0.5 * expected - 100:
            notes.append(f"Total charges ({data['TotalCharges']:.0f}) differ a lot from tenure x monthly charges "
                         f"(about {expected:.0f}). The prediction is still made, but check the inputs.")
    return data, notes


# ----------------------------------------------------------------------------
# ML prediction
# ----------------------------------------------------------------------------
def risk_category(prob: float) -> str:
    if prob < LOW_RISK_MAX:
        return "Low"
    if prob < HIGH_RISK_MIN:
        return "Medium"
    return "High"


def predict_customer(model, inputs: dict) -> dict:
    """Predict churn for one customer. Returns label, probability, risk and notes."""
    data, notes = prepare_input(inputs)
    row = pd.DataFrame([data])[FEATURE_COLUMNS]
    label = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0, 1])
    return {"churn": "Yes" if label == 1 else "No", "probability": prob,
            "risk": risk_category(prob), "notes": notes, "clean_input": data}


# ----------------------------------------------------------------------------
# Rule-based recommendations (business rules, NOT model output)
# ----------------------------------------------------------------------------
RULES = [
    {"id": "R1", "name": "Month-to-month contract",
     "condition": lambda r: r["Contract"] == "Month-to-month",
     "action": "Offer a discounted 1-year or 2-year contract (e.g. a price lock or a free add-on)."},
    {"id": "R2", "name": f"High monthly charges (>= {HIGH_CHARGE:.0f})",
     "condition": lambda r: r["MonthlyCharges"] >= HIGH_CHARGE,
     "action": "Review the plan and suggest a suitable lower-cost bundle or a loyalty discount."},
    {"id": "R3", "name": "No tech support (internet customer)",
     "condition": lambda r: r["InternetService"] != "No" and r["TechSupport"] == "No",
     "action": "Offer free or discounted tech support for a trial period."},
    {"id": "R4", "name": "No online security (internet customer)",
     "condition": lambda r: r["InternetService"] != "No" and r["OnlineSecurity"] == "No",
     "action": "Offer an online security add-on trial to increase service value."},
    {"id": "R5", "name": f"Low tenure (< {LOW_TENURE} months)",
     "condition": lambda r: r["tenure"] < LOW_TENURE,
     "action": "Run an onboarding programme and give early-loyalty rewards."},
    {"id": "R6", "name": "Electronic check payment",
     "condition": lambda r: r["PaymentMethod"] == "Electronic check",
     "action": "Encourage automatic payment (bank transfer / credit card) with a small incentive."},
    {"id": "R7", "name": "Fiber optic with high charges",
     "condition": lambda r: r["InternetService"] == "Fiber optic" and r["MonthlyCharges"] >= HIGH_CHARGE,
     "action": "Check service quality and offer a speed/price review for this premium customer."},
]


def get_recommendations(inputs: dict) -> list:
    """Return the list of rules (id, name, action) that match this customer."""
    return [{"id": r["id"], "rule": r["name"], "action": r["action"]}
            for r in RULES if r["condition"](inputs)]


def rule_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every rule: how many customers in the dataset match it and what their ACTUAL churn
    rate is. This shows the rules are backed by the data.
    """
    records = df.to_dict("records")
    overall = (df["Churn"] == "Yes").mean() * 100
    rows = []
    for rule in RULES:
        mask = pd.Series([bool(rule["condition"](r)) for r in records], index=df.index)
        n = int(mask.sum())
        rate = (df.loc[mask, "Churn"] == "Yes").mean() * 100 if n else float("nan")
        rows.append({"ID": rule["id"], "Rule": rule["name"], "Customers matching": n,
                     "Actual churn rate (%)": round(rate, 1), "Overall churn rate (%)": round(overall, 1),
                     "Recommended action": rule["action"]})
    return pd.DataFrame(rows)
