"""
app.py  -  Customer Churn Prediction & Retention Analytics (Streamlit dashboard)

Run with:   streamlit run app.py
(Run  python train_models.py  first so that the trained models exist.)

The app only DISPLAYS results saved by train_models.py. No metric is typed in by hand.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import eda, evaluation, prediction
from src.model_training import load_artifact
from src.preprocessing import FEATURE_COLUMNS, ID_COL, clean_data, inspect_data, load_data

BASE = Path(__file__).parent
CSV_PATH = BASE / "telco_customer_churn_synthetic.csv"
MODELS_DIR = BASE / "models"
REPORTS_DIR = BASE / "outputs" / "reports"

st.set_page_config(page_title="Churn Analytics", page_icon="📡", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1250px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0B1B3A 0%, #12295A 100%);}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {color: #E2E8F0;}
[data-testid="stSidebar"] hr {border-color: rgba(255,255,255,0.15);}
.page-title {font-size: 1.9rem; font-weight: 800; color: #0B1B3A; margin-bottom: 0;}
.page-sub {color: #475569; margin-bottom: 1rem;}
.badge {display:inline-block; background:#FEF3C7; color:#92400E; padding:3px 10px; border-radius:999px;
        font-size:0.78rem; font-weight:600; margin-bottom: 0.6rem;}
.badge-ml {background:#DBEAFE; color:#1E40AF;}
.badge-rule {background:#DCFCE7; color:#166534;}
.kpi-card {background:#FFFFFF; border-radius:14px; padding:16px 18px; border-left:6px solid var(--c);
           box-shadow:0 2px 10px rgba(15,23,42,0.07); margin-bottom: 12px; min-height: 104px;}
.kpi-label {font-size:0.78rem; color:#64748B; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;}
.kpi-value {font-size:1.75rem; font-weight:800; color:#0B1B3A; line-height:1.25;}
.kpi-sub {font-size:0.78rem; color:#64748B;}
.result-card {border-radius:16px; padding:22px; color:white; text-align:center; margin-bottom: 10px;}
.result-card h2 {color:white; margin:0; font-size:2rem;}
.result-card p {color:rgba(255,255,255,0.92); margin:4px 0 0 0;}
@media (max-width: 700px) {.kpi-value {font-size: 1.35rem;} .page-title {font-size: 1.5rem;}}
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {"Low": "#16A34A", "Medium": "#D97706", "High": "#DC2626"}


def kpi(label, value, sub="", color="#2563EB"):
    st.markdown(f"""<div class="kpi-card" style="--c:{color}">
        <div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>',
                unsafe_allow_html=True)


def show(fig):
    """Draw a matplotlib figure in Streamlit and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    raw = load_data(CSV_PATH)
    clean, report = clean_data(raw)
    return raw, clean, inspect_data(raw), report


@st.cache_data
def load_reports():
    metrics = pd.read_csv(REPORTS_DIR / "model_metrics.csv")
    importance = pd.read_csv(REPORTS_DIR / "feature_importance_random_forest.csv")
    return metrics, importance


@st.cache_resource
def load_models():
    return {
        "all": load_artifact(MODELS_DIR / "all_models.joblib"),
        "tests": load_artifact(MODELS_DIR / "test_results.joblib"),
        "summary": load_artifact(MODELS_DIR / "run_summary.joblib"),
    }


@st.cache_data
def score_customers(_model, df):
    """Churn probability of every customer in the dataset (used on the recommendations page)."""
    return _model.predict_proba(df[FEATURE_COLUMNS])[:, 1]


try:
    raw_df, df, info, clean_report = load_dataset()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

if not (MODELS_DIR / "all_models.joblib").exists() or not (REPORTS_DIR / "model_metrics.csv").exists():
    st.error("Trained models not found. Open a terminal in the project folder and run:  "
             "`python train_models.py`  - then refresh this page.")
    st.stop()

metrics, importance = load_reports()
art = load_models()
best_name = art["summary"]["best_model"]
best_row = metrics[metrics["Model"] == best_name].iloc[0]
df["SeniorLabel"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
PAGES = ["📊 Dashboard", "🗂️ Dataset Explorer", "🔍 Churn Analysis", "🤖 Model Comparison",
         "🎯 Predict Churn", "💡 Retention Recommendations"]
with st.sidebar:
    st.markdown("## 📡 Churn Analytics")
    st.caption("Customer Churn Prediction & Retention")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("⚠️ **Synthetic educational dataset.** Results show the method, "
               "not real telecom customers.")
    st.caption(f"Dataset: {len(df):,} customers · {len(FEATURE_COLUMNS)} features")


# ----------------------------------------------------------------------------
# A. Dashboard
# ----------------------------------------------------------------------------
def page_dashboard():
    page_header("Customer Churn Dashboard", "Overview of telecom customers and the best-performing model")
    st.markdown('<span class="badge">Synthetic educational dataset</span>', unsafe_allow_html=True)

    total = len(df)
    churned = int((df["Churn"] == "Yes").sum())
    active = total - churned
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Total customers", f"{total:,}", "Rows in dataset", "#2563EB")
    with c2:
        kpi("Churned customers", f"{churned:,}", "Churn = Yes", "#F97316")
    with c3:
        kpi("Active customers", f"{active:,}", "Churn = No", "#10B981")
    c4, c5, c6 = st.columns(3)
    with c4:
        kpi("Churn rate", f"{churned / total * 100:.1f}%", "Share of customers who left", "#EF4444")
    with c5:
        kpi("Best ML model", best_name, "Chosen by highest churn F1-score", "#8B5CF6")
    with c6:
        kpi("Best model F1-score", f"{best_row['F1-Score']:.3f}",
            f"Recall {best_row['Recall']:.2f} · Precision {best_row['Precision']:.2f} · "
            f"ROC-AUC {best_row['ROC-AUC']:.2f}", "#0EA5E9")

    st.markdown("### Churn overview and customer trends")
    a, b = st.columns(2)
    with a:
        show(eda.plot_churn_percentage(df))
    with b:
        show(eda.plot_churn_rate_by(df, "Contract", order=["Month-to-month", "One year", "Two year"]))
    a, b = st.columns(2)
    with a:
        show(eda.plot_churn_by_tenure(df))
    with b:
        show(eda.plot_monthly_charges_vs_churn(df))
    st.info("Model metrics come from the 20% test set (240 customers) that the models never saw during training.")


# ----------------------------------------------------------------------------
# B. Dataset Explorer
# ----------------------------------------------------------------------------
def page_explorer():
    page_header("Dataset Explorer", "Browse, search, filter and download the customer data")
    st.markdown('<span class="badge">Synthetic educational dataset</span>', unsafe_allow_html=True)

    with st.expander("🔎 Search and filters", expanded=True):
        search = st.text_input("Search by customerID (e.g. TC01005)")
        f1, f2, f3 = st.columns(3)
        churn_f = f1.multiselect("Churn", ["Yes", "No"], default=["Yes", "No"])
        contract_f = f2.multiselect("Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
        internet_f = f3.multiselect("Internet service", sorted(df["InternetService"].unique()),
                                    default=sorted(df["InternetService"].unique()))
        tmin, tmax = int(df["tenure"].min()), int(df["tenure"].max())
        tenure_f = st.slider("Tenure (months)", tmin, tmax, (tmin, tmax))

    view = df.drop(columns=["SeniorLabel"])
    view = view[view["Churn"].isin(churn_f) & view["Contract"].isin(contract_f)
                & view["InternetService"].isin(internet_f) & view["tenure"].between(*tenure_f)]
    if search:
        view = view[view[ID_COL].str.contains(search.strip(), case=False, na=False)]

    m1, m2, m3 = st.columns(3)
    m1.metric("Rows shown", f"{len(view):,}")
    m2.metric("Total rows / columns", f"{df.shape[0]:,} / {df.shape[1] - 1}")
    m3.metric("Churn rate in view", f"{(view['Churn'] == 'Yes').mean() * 100:.1f}%" if len(view) else "-")

    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Missing values", "Data types", "Summary statistics"])
    with tab1:
        st.dataframe(view, hide_index=True)
        st.download_button("⬇️ Download filtered data as CSV", view.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_customers.csv", mime="text/csv")
    with tab2:
        miss = raw_df.isna().sum().rename("Missing values").to_frame()
        miss["Missing %"] = (miss["Missing values"] / len(raw_df) * 100).round(2)
        st.dataframe(miss)
        st.success(f"Total missing values: {info['total_missing']} · Duplicate rows: {info['duplicate_rows']} · "
                   f"Duplicate customer IDs: {info['duplicate_customer_ids']}")
    with tab3:
        st.dataframe(raw_df.dtypes.astype(str).rename("Data type").to_frame())
    with tab4:
        st.markdown("**Numerical columns**")
        st.dataframe(view.describe().T.round(2))
        st.markdown("**Categorical columns**")
        st.dataframe(view.drop(columns=[ID_COL]).describe(include=["object", "string"]).T)


# ----------------------------------------------------------------------------
# C. Customer Churn Analysis
# ----------------------------------------------------------------------------
def page_analysis():
    page_header("Customer Churn Analysis", "Which customer groups leave the most? Use the filters to compare groups")
    with st.expander("🎛️ Filter customer groups", expanded=True):
        f1, f2, f3 = st.columns(3)
        contract = f1.multiselect("Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
        internet = f2.multiselect("Internet service", sorted(df["InternetService"].unique()),
                                  default=sorted(df["InternetService"].unique()))
        payment = f3.multiselect("Payment method", sorted(df["PaymentMethod"].unique()),
                                 default=sorted(df["PaymentMethod"].unique()))
        g1, g2, g3 = st.columns(3)
        gender = g1.multiselect("Gender", ["Female", "Male"], default=["Female", "Male"])
        senior = g2.multiselect("Senior citizen", ["No", "Yes"], default=["No", "Yes"])
        tenure = g3.slider("Tenure (months)", 0, 72, (0, 72))

    d = df[df["Contract"].isin(contract) & df["InternetService"].isin(internet)
           & df["PaymentMethod"].isin(payment) & df["gender"].isin(gender)
           & df["SeniorLabel"].isin(senior) & df["tenure"].between(*tenure)]
    overall = (df["Churn"] == "Yes").mean() * 100
    if d.empty:
        st.warning("No customers match these filters. Please widen the selection.")
        return
    rate = (d["Churn"] == "Yes").mean() * 100
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi("Customers in selection", f"{len(d):,}", f"of {len(df):,} total", "#2563EB")
    with k2:
        kpi("Churn rate (selection)", f"{rate:.1f}%", f"Overall: {overall:.1f}%", "#F97316")
    with k3:
        kpi("Avg monthly charges", f"{d['MonthlyCharges'].mean():.2f}", f"Avg tenure {d['tenure'].mean():.1f} months", "#10B981")
    if len(d) < 30:
        st.warning("Small group (< 30 customers): percentages can be unreliable.")

    t1, t2, t3, t4, t5, t6 = st.tabs(["Contract", "Tenure", "Monthly charges", "Payment method",
                                      "Internet service", "Demographics"])
    with t1:
        show(eda.plot_churn_rate_by(d, "Contract", order=["Month-to-month", "One year", "Two year"]))
    with t2:
        a, b = st.columns(2)
        with a:
            show(eda.plot_churn_by_tenure(d))
        with b:
            show(eda.plot_tenure_vs_churn(d))
    with t3:
        a, b = st.columns(2)
        with a:
            show(eda.plot_monthly_charges_vs_churn(d))
        with b:
            bins = pd.cut(d["MonthlyCharges"], [0, 40, 60, 80, 200],
                          labels=["< 40", "40-60", "60-80", "80+"])
            show(eda.plot_churn_rate_by(d.assign(ChargeBand=bins), "ChargeBand", "Churn Rate by Monthly Charge Band",
                                        order=["< 40", "40-60", "60-80", "80+"]))
    with t4:
        show(eda.plot_churn_rate_by(d, "PaymentMethod", rotate=15))
    with t5:
        show(eda.plot_churn_rate_by(d, "InternetService"))
    with t6:
        show(eda.plot_churn_by_group_grid(d, ["gender", "SeniorLabel", "Partner", "Dependents"]))
    st.caption("Dashed line = overall churn rate of the selected customers. n = customers in each bar.")


# ----------------------------------------------------------------------------
# D. Model Comparison
# ----------------------------------------------------------------------------
def page_models():
    page_header("Model Comparison", "Six classifiers trained on the same stratified split and tested on unseen data")
    s = art["summary"]
    st.success(f"🏆 Best model: **{best_name}** - highest F1-score for churn "
               f"({best_row['F1-Score']:.3f}) with recall {best_row['Recall']:.3f}, "
               f"precision {best_row['Precision']:.3f}, ROC-AUC {best_row['ROC-AUC']:.3f}.")
    st.caption(f"Train rows: {s['split']['train_rows']} · Test rows: {s['split']['test_rows']} · "
               f"Stratified split, random_state = {s['split']['random_state']} · "
               "Hyper-parameters tuned with 5-fold CV on training data only.")

    show_df = metrics.drop(columns=["Best"]).copy()
    show_df.insert(0, "★", ["★" if m == best_name else "" for m in show_df["Model"]])
    fmt = {c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "CV F1 (train)"]}
    styled = show_df.style.format(fmt).apply(
        lambda r: ["background-color: #FEF3C7; font-weight: 600" if r["Model"] == best_name else "" for _ in r], axis=1)
    st.dataframe(styled, hide_index=True)

    with st.expander("Why not choose the model by accuracy alone?"):
        st.markdown(
            "- Only about **29%** of customers churn. A useless model that says *'everyone stays'* would already be ~71% accurate.\n"
            "- **Recall** = of all customers who really leave, how many did we catch? Missing a churner means losing revenue.\n"
            "- **Precision** = of the customers we flag, how many really leave? Low precision wastes retention budget on happy customers.\n"
            "- **F1-score** balances precision and recall, so it is used to pick the best model (ROC-AUC breaks ties).\n"
            "- **ROC-AUC** shows how well the model ranks churners above non-churners at any threshold.")

    a, b = st.columns(2)
    with a:
        show(evaluation.plot_accuracy_comparison(metrics, best_name))
    with b:
        show(evaluation.plot_prf_comparison(metrics))

    st.markdown("### Detailed results for one model")
    chosen = st.selectbox("Choose a model", list(metrics["Model"]), index=list(metrics["Model"]).index(best_name))
    y_test = art["tests"]["y_test"]
    y_pred = art["tests"]["y_pred"][chosen]
    a, b = st.columns(2)
    with a:
        show(evaluation.plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {chosen}"))
    with b:
        show(evaluation.plot_roc_curves(y_test, art["tests"]["y_proba"], highlight=chosen))
    st.markdown("**Classification report**")
    st.dataframe(evaluation.get_classification_report_df(y_test, y_pred))

    st.markdown("### Feature importance (Random Forest)")
    show(evaluation.plot_feature_importance(importance))
    st.caption("The test set has only 240 customers, so small differences between models (about 0.02-0.03) "
               "may be due to chance. KNN is clearly weaker than the rest.")


# ----------------------------------------------------------------------------
# E. Predict churn for a new customer
# ----------------------------------------------------------------------------
def page_predict():
    page_header("Predict Churn for a New Customer", "Enter customer details and let the trained model estimate the churn risk")
    st.markdown('<span class="badge badge-ml">ML model prediction</span>', unsafe_allow_html=True)
    V = prediction.VALID_VALUES
    model_name = st.selectbox("Model used for prediction", list(art["all"].keys()),
                              index=list(art["all"].keys()).index(best_name),
                              help="The best model is selected by default.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", V["gender"])
        senior = c1.selectbox("Senior citizen", ["No", "Yes"])
        partner = c1.selectbox("Partner", V["Partner"], index=1)
        dependents = c1.selectbox("Dependents", V["Dependents"], index=1)
        tenure = c1.number_input("Tenure (months)", 0, 120, 12)
        monthly = c1.number_input("Monthly charges", 1.0, 500.0, 70.0, step=0.5)
        total = c1.number_input("Total charges", 0.0, 50000.0, 840.0, step=10.0,
                                help="Usually about tenure x monthly charges.")

        phone = c2.selectbox("Phone service", V["PhoneService"])
        lines = c2.selectbox("Multiple lines", V["MultipleLines"])
        internet = c2.selectbox("Internet service", V["InternetService"], index=1)
        security = c2.selectbox("Online security", ["Yes", "No", "No internet service"], index=1)
        backup = c2.selectbox("Online backup", ["Yes", "No", "No internet service"], index=1)
        device = c2.selectbox("Device protection", ["Yes", "No", "No internet service"], index=1)

        support = c3.selectbox("Tech support", ["Yes", "No", "No internet service"], index=1)
        tv = c3.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=1)
        movies = c3.selectbox("Streaming movies", ["Yes", "No", "No internet service"], index=1)
        contract = c3.selectbox("Contract", V["Contract"])
        paperless = c3.selectbox("Paperless billing", V["PaperlessBilling"])
        payment = c3.selectbox("Payment method", V["PaymentMethod"])
        submitted = st.form_submit_button("🔮 Predict churn", type="primary")

    if not submitted:
        st.info("Fill in the form and press **Predict churn**. If there is no internet service, the add-on fields are ignored.")
        return

    inputs = {"gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
              "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
              "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
              "DeviceProtection": device, "TechSupport": support, "StreamingTV": tv,
              "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
              "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total}
    try:
        res = prediction.predict_customer(art["all"][model_name], inputs)
    except ValueError as err:
        st.error(f"Invalid input: {err}")
        return
    except Exception as err:  # unexpected problem - show a friendly message
        st.error(f"Prediction failed: {err}")
        return

    for note in res["notes"]:
        st.warning(note)

    prob, risk = res["probability"], res["risk"]
    left, right = st.columns([1, 1])
    with left:
        verdict = "Likely to CHURN" if res["churn"] == "Yes" else "Likely to STAY"
        st.markdown(f"""<div class="result-card" style="background:{RISK_COLORS[risk]}">
            <p>Predicted churn: <b>{res['churn']}</b></p><h2>{verdict}</h2>
            <p>Risk category: <b>{risk}</b></p></div>""", unsafe_allow_html=True)
    with right:
        st.metric("Predicted probability of churn", f"{prob * 100:.1f}%")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"Risk category: Low < {prediction.LOW_RISK_MAX:.0%} · Medium {prediction.LOW_RISK_MAX:.0%}-"
                   f"{prediction.HIGH_RISK_MIN:.0%} · High ≥ {prediction.HIGH_RISK_MIN:.0%}")
    st.caption(f"Model: {model_name}. The probability is a statistical estimate learned from a synthetic dataset - "
               "it is **not a guarantee** that the customer will or will not leave.")
    if (res["churn"] == "Yes") != (prob >= 0.5):
        st.caption("Note: this model's Yes/No label and its probability use slightly different decision rules, "
                   "so they can disagree near the 50% mark.")

    st.markdown("### Retention recommendations")
    st.markdown('<span class="badge badge-rule">Rule-based - not an ML output</span>', unsafe_allow_html=True)
    recs = prediction.get_recommendations(res["clean_input"])
    if recs:
        st.table(pd.DataFrame(recs).rename(columns={"id": "Rule ID", "rule": "Triggered by", "action": "Suggested action"}))
    else:
        st.success("No retention rule is triggered for this customer. Keep delivering good service.")
    st.caption("These suggestions come from simple business rules on the inputs. They do not depend on the model's probability.")


# ----------------------------------------------------------------------------
# F. Retention recommendations
# ----------------------------------------------------------------------------
def page_recommendations():
    page_header("Customer Retention Recommendations", "Business rules to reduce churn, checked against the data")
    st.markdown("""
<span class="badge badge-rule">Rule-based</span> Fixed if-then business rules written by us. They look at one customer's
features and suggest an action.  &nbsp;&nbsp; <span class="badge badge-ml">ML prediction</span> A trained model estimates the
probability of churn from patterns in the data. The two are shown separately on purpose.
""", unsafe_allow_html=True)

    summary = prediction.rule_summary(df)
    st.markdown("### Rules and how the data supports them")
    st.dataframe(summary, hide_index=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(summary["ID"] + " " + summary["Rule"], summary["Actual churn rate (%)"], color="#F97316")
    ax.axvline(summary["Overall churn rate (%)"].iloc[0], color="#1E293B", ls="--", label="Overall churn rate")
    ax.invert_yaxis()
    ax.set_xlabel("Actual churn rate of matching customers (%)")
    ax.set_title("Churn rate of customers matching each rule", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    show(fig)
    st.caption("A rule is most useful when its bar is clearly to the right of the dashed line. "
               "Rules that are close to the line have a weaker link with churn in this dataset.")

    st.markdown("### Active customers to prioritise (ML score + matching rules)")
    st.markdown('<span class="badge badge-ml">ML probability</span> <span class="badge badge-rule">Rule IDs</span>',
                unsafe_allow_html=True)
    model = art["all"][best_name]
    scored = df.copy()
    scored["Churn probability"] = score_customers(model, df)
    scored["Risk"] = scored["Churn probability"].map(prediction.risk_category)
    records = scored.to_dict("records")
    scored["Matching rules"] = [", ".join(r["id"] for r in prediction.RULES if r["condition"](rec)) for rec in records]
    active = scored[scored["Churn"] == "No"].sort_values("Churn probability", ascending=False)
    n = st.slider("How many customers to show", 5, 50, 15)
    cols = [ID_COL, "Churn probability", "Risk", "Contract", "tenure", "MonthlyCharges", "InternetService",
            "PaymentMethod", "Matching rules"]
    table = active[cols].head(n).copy()
    table["Churn probability"] = (table["Churn probability"] * 100).round(1).astype(str) + "%"
    st.dataframe(table, hide_index=True)
    st.caption(f"Scored with the best model ({best_name}). Most of these customers were part of the training data, "
               "so their probabilities can look optimistic; use this list as a demonstration of the workflow.")


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
{
    PAGES[0]: page_dashboard,
    PAGES[1]: page_explorer,
    PAGES[2]: page_analysis,
    PAGES[3]: page_models,
    PAGES[4]: page_predict,
    PAGES[5]: page_recommendations,
}[page]()
