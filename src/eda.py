"""
eda.py
------
Exploratory Data Analysis charts (Matplotlib + Seaborn).
Every function takes a DataFrame and returns a Figure, so the same code is used
to save PNG files (train_models.py) and to draw live charts in the Streamlit app.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

COLORS = {"No": "#3B82F6", "Yes": "#F97316"}      # blue = stays, orange = churns
TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]

sns.set_theme(style="whitegrid", font_scale=0.95)


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TenureGroup"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)
    return df


def _empty_fig(msg="No data for the selected filters"):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color="gray")
    ax.axis("off")
    return fig


def plot_churn_distribution(df):
    """Bar chart with counts of churned vs. retained customers."""
    if df.empty:
        return _empty_fig()
    fig, ax = plt.subplots(figsize=(5, 4))
    order = ["No", "Yes"]
    sns.countplot(data=df, x="Churn", order=order, palette=COLORS, hue="Churn", legend=False, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, padding=2)
    ax.set_title("Churn Distribution", fontweight="bold")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of customers")
    sns.despine()
    fig.tight_layout()
    return fig


def plot_churn_percentage(df):
    """Donut chart showing the churn percentage."""
    if df.empty:
        return _empty_fig()
    counts = df["Churn"].value_counts().reindex(["No", "Yes"]).fillna(0)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.pie(counts, labels=["Stayed", "Churned"], colors=[COLORS["No"], COLORS["Yes"]],
           autopct="%1.1f%%", startangle=90, wedgeprops={"width": 0.4, "edgecolor": "white"},
           textprops={"fontsize": 10})
    ax.set_title("Churn Percentage", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_churn_rate_by(df, col, title=None, order=None, rotate=0):
    """Churn rate (%) for every value of a categorical column, with customer counts printed."""
    if df.empty:
        return _empty_fig()
    g = df.groupby(col, observed=True)["Churn"].agg(rate=lambda s: (s == "Yes").mean() * 100, n="size")
    if order:
        g = g.reindex([o for o in order if o in g.index])
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(g.index.astype(str), g["rate"], color="#F97316", width=0.6)
    overall = (df["Churn"] == "Yes").mean() * 100
    ax.axhline(overall, color="#1E293B", ls="--", lw=1, label=f"Overall: {overall:.1f}%")
    for b, (rate, n) in zip(bars, zip(g["rate"], g["n"])):
        ax.text(b.get_x() + b.get_width() / 2, rate + 0.8, f"{rate:.1f}%\n(n={n})",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(g["rate"].max() * 1.3, 10))
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel(col)
    ax.set_title(title or f"Churn Rate by {col}", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right" if rotate else "center")
    sns.despine()
    fig.tight_layout()
    return fig


def plot_churn_by_tenure(df):
    """Churn rate for each tenure group."""
    d = add_tenure_group(df)
    return plot_churn_rate_by(d, "TenureGroup", "Churn Rate by Tenure Group", order=TENURE_LABELS)


def plot_monthly_charges_vs_churn(df):
    """Box plot of MonthlyCharges for churned vs. retained customers."""
    if df.empty:
        return _empty_fig()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", order=["No", "Yes"], palette=COLORS,
                hue="Churn", legend=False, ax=ax)
    ax.set_title("Monthly Charges vs Churn", fontweight="bold")
    ax.set_ylabel("Monthly charges")
    sns.despine()
    fig.tight_layout()
    return fig


def plot_tenure_vs_churn(df):
    """Overlapping histograms of tenure for churned vs. retained customers."""
    if df.empty:
        return _empty_fig()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", hue_order=["No", "Yes"], palette=COLORS,
                 bins=24, multiple="layer", alpha=0.6, ax=ax)
    ax.set_title("Tenure vs Churn", fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Customers")
    sns.despine()
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Correlation of numerical features (Churn encoded as 1/0 so we can see its relation)."""
    d = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]].copy()
    d["Churn (1=Yes)"] = (df["Churn"] == "Yes").astype(int)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    sns.heatmap(d.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap (numerical features)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_churn_by_group_grid(df, columns):
    """Small multiples of churn rate for several demographic columns."""
    if df.empty:
        return _empty_fig()
    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    axes = [axes] if n == 1 else axes
    for ax, col in zip(axes, columns):
        rate = df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
        ax.bar(rate.index.astype(str), rate.values, color="#2563EB", width=0.55)
        for i, v in enumerate(rate.values):
            ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=8)
        ax.set_title(col, fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Churn rate (%)")
    sns.despine()
    fig.tight_layout()
    return fig


def save_all_eda_plots(df, out_dir):
    """Save every EDA figure required by the project brief as PNG files."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    d["SeniorLabel"] = d["SeniorCitizen"].map({0: "No", 1: "Yes"})
    figs = {
        "01_churn_distribution": plot_churn_distribution(d),
        "02_churn_percentage": plot_churn_percentage(d),
        "03_churn_by_gender": plot_churn_rate_by(d, "gender"),
        "04_churn_by_contract": plot_churn_rate_by(d, "Contract", order=["Month-to-month", "One year", "Two year"]),
        "05_churn_by_tenure": plot_churn_by_tenure(d),
        "06_churn_by_internet_service": plot_churn_rate_by(d, "InternetService"),
        "07_churn_by_payment_method": plot_churn_rate_by(d, "PaymentMethod", rotate=20),
        "08_monthly_charges_vs_churn": plot_monthly_charges_vs_churn(d),
        "09_tenure_vs_churn": plot_tenure_vs_churn(d),
        "10_correlation_heatmap": plot_correlation_heatmap(d),
    }
    for name, fig in figs.items():
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
