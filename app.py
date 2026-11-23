"""Explore synthetic card transactions with a small fraud-detection dashboard.

Run the app locally with ``streamlit run app.py``. The transaction data is
generated for demonstration purposes and should not be used to make real
financial decisions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree


st.set_page_config(
    page_title="SentinelGuard",
    page_icon="shield",
    layout="wide",
)

DEFAULT_SEED = 42
FRAUD_RATE = 0.05

FEATURES = [
    "Transaction_Amount",
    "Hour_Of_Day",
    "Account_Age_Days",
    "Failed_Pin_Attempts",
    "Is_International",
]


def make_transactions(count=500, seed=DEFAULT_SEED) -> pd.DataFrame:
    """Build a reproducible sample of legitimate and fraudulent transactions."""
    rng = np.random.default_rng(seed)

    fraud_count = int(round(count * FRAUD_RATE))
    legitimate_count = count - fraud_count

    legitimate = pd.DataFrame(
        {
            "Transaction_Amount": np.round(
                rng.gamma(shape=2.0, scale=75.0, size=legitimate_count).clip(
                    5.0, 1800.0
                ),
                2,
            ),
            "Hour_Of_Day": rng.integers(7, 21, size=legitimate_count),
            "Account_Age_Days": rng.integers(120, 2000, size=legitimate_count),
            "Failed_Pin_Attempts": rng.choice(
                [0, 1, 2],
                size=legitimate_count,
                p=[0.84, 0.13, 0.03],
            ),
            "Is_International": rng.choice(
                [0, 1],
                size=legitimate_count,
                p=[0.90, 0.10],
            ),
            "Is_Fraud": np.zeros(legitimate_count, dtype=int),
        }
    )

    fraudulent = pd.DataFrame(
        {
            "Transaction_Amount": np.round(
                rng.normal(loc=950.0, scale=320.0, size=fraud_count).clip(
                    150.0, 3500.0
                ),
                2,
            ),
            "Hour_Of_Day": rng.choice(
                [19, 20, 21, 22, 23, 14, 15],
                size=fraud_count,
                p=[0.12, 0.18, 0.22, 0.22, 0.16, 0.05, 0.05],
            ),
            "Account_Age_Days": rng.integers(1, 320, size=fraud_count),
            "Failed_Pin_Attempts": rng.choice(
                [0, 1, 2, 3, 4, 5],
                size=fraud_count,
                p=[0.08, 0.14, 0.24, 0.26, 0.18, 0.10],
            ),
            "Is_International": rng.choice(
                [0, 1],
                size=fraud_count,
                p=[0.28, 0.72],
            ),
            "Is_Fraud": np.ones(fraud_count, dtype=int),
        }
    )

    data = pd.concat([legitimate, fraudulent], ignore_index=True)

    risk = (
        10.0
        + 0.025 * data["Transaction_Amount"]
        + 0.9 * data["Hour_Of_Day"]
        - 0.012 * data["Account_Age_Days"]
        + 9.5 * data["Failed_Pin_Attempts"]
        + 14.0 * data["Is_International"]
        + rng.normal(0.0, 4.0, size=len(data))
    )
    data["Risk_Score"] = np.round(np.clip(risk, 0.0, 100.0), 1)

    return data.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@st.cache_resource
def train_models() -> dict:
    """Fit the models used by the dashboard and collect their evaluation data."""
    data = make_transactions(count=500, seed=DEFAULT_SEED)
    X = data[FEATURES]
    y = data["Is_Fraud"]

    # Fraud classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=DEFAULT_SEED,
        stratify=y,
    )

    classifier_scaler = StandardScaler()
    train_scaled = classifier_scaler.fit_transform(X_train)
    test_scaled = classifier_scaler.transform(X_test)

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=DEFAULT_SEED,
    )
    classifier.fit(train_scaled, y_train)

    predicted = classifier.predict(test_scaled)
    fraud_probability = classifier.predict_proba(test_scaled)[:, 1]

    class_report = classification_report(
        y_test,
        predicted,
        target_names=["Legitimate", "Fraud"],
        output_dict=True,
        zero_division=0,
    )

    test_results = X_test.copy()
    test_results["Actual_Fraud"] = y_test.values
    test_results["Predicted_Fraud"] = predicted
    test_results["Fraud_Probability_%"] = np.round(
        fraud_probability * 100, 1
    )

    coefficients = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Coefficient": classifier.coef_[0],
        }
    ).sort_values("Coefficient")

    # Transaction amount and risk score
    X_amount = data[["Transaction_Amount"]]
    y_score = data["Risk_Score"]

    amount_train, amount_test, score_train, score_test = train_test_split(
        X_amount,
        y_score,
        test_size=0.20,
        random_state=DEFAULT_SEED,
    )
    linear_model = LinearRegression()
    linear_model.fit(amount_train, score_train)
    linear_predictions = linear_model.predict(amount_test)

    # Multi-feature decision tree
    tree_train, tree_test, tree_score_train, tree_score_test = train_test_split(
        X,
        y_score,
        test_size=0.20,
        random_state=DEFAULT_SEED,
    )
    full_tree = DecisionTreeRegressor(random_state=DEFAULT_SEED)
    full_tree.fit(tree_train, tree_score_train)

    # Clustering
    cluster_scaler = StandardScaler()
    X_cluster = cluster_scaler.fit_transform(X)
    default_clusters = KMeans(
        n_clusters=3,
        random_state=8,
        n_init=10,
    )
    default_clusters.fit(X_cluster)

    return {
        "data": data,
        "classifier": classifier,
        "classifier_scaler": classifier_scaler,
        "accuracy": accuracy_score(y_test, predicted),
        "confusion_matrix": confusion_matrix(y_test, predicted),
        "class_report": class_report,
        "test_results": test_results,
        "coefficients": coefficients,
        "linear_model": linear_model,
        "amount_test": amount_test,
        "score_test": score_test,
        "linear_predictions": linear_predictions,
        "linear_mae": mean_absolute_error(score_test, linear_predictions),
        "linear_r2": r2_score(score_test, linear_predictions),
        "tree_train": tree_train,
        "tree_test": tree_test,
        "tree_score_train": tree_score_train,
        "tree_score_test": tree_score_test,
        "full_tree": full_tree,
        "cluster_scaler": cluster_scaler,
        "default_clusters": default_clusters,
    }


def get_risk_badge(probability: float) -> dict[str, str]:
    """Choose the label and colors shown for a predicted fraud probability."""
    if probability < 0.30:
        return {
            "label": "SAFE - APPROVE TRANSACTION",
            "background": "#E6F4EA",
            "border": "#1E8E3E",
            "text": "#137333",
            "message": "Low fraud probability (<30%). The transaction looks normal.",
        }
    if probability <= 0.65:
        return {
            "label": "FLAGGED - MANUAL REVIEW / 2FA",
            "background": "#FEF7E0",
            "border": "#F9AB00",
            "text": "#B06000",
            "message": "Moderate fraud probability (30%-65%). Additional verification is advised.",
        }
    return {
        "label": "HIGH-RISK - BLOCK TRANSACTION",
        "background": "#FCE8E6",
        "border": "#D93025",
        "text": "#A50E0E",
        "message": "High fraud probability (>65%). Block the transaction and notify the cardholder.",
    }


def show_risk_badge(probability: float) -> None:
    """Render the risk level as a colored callout in the app."""
    style = get_risk_badge(probability)
    st.markdown(
        f"""
        <div style="background-color: {style['background']};
                    border-left: 6px solid {style['border']};
                    border-radius: 8px;
                    padding: 16px 20px;
                    margin: 10px 0 18px 0;">
            <div style="font-size: 1.22rem; font-weight: 700;
                        color: {style['text']};">
                {style['label']}
            </div>
            <div style="font-size: 0.98rem; color: #202124; margin-top: 6px;">
                Predicted fraud probability:
                <strong>{probability * 100:.1f}%</strong> - {style['message']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_coefficients(coefficients: pd.DataFrame) -> plt.Figure:
    """Plot each standardized model coefficient and its direction."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bar_colors = [
        "#2E8B57" if value < 0 else "#D64545"
        for value in coefficients["Coefficient"]
    ]

    bars = ax.barh(
        coefficients["Feature"],
        coefficients["Coefficient"],
        color=bar_colors,
        edgecolor="#222222",
        linewidth=0.6,
    )
    ax.axvline(0, color="#333333", linestyle="--", linewidth=1.0)

    minimum = coefficients["Coefficient"].min()
    maximum = coefficients["Coefficient"].max()
    padding = max(abs(minimum), abs(maximum)) * 0.04

    for bar, value in zip(bars, coefficients["Coefficient"]):
        label_x = value + padding if value >= 0 else value - padding
        alignment = "left" if value >= 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}",
            va="center",
            ha=alignment,
            fontsize=9.5,
        )

    ax.set_xlim(
        min(minimum - padding * 6, -0.5),
        max(maximum + padding * 6, 0.5),
    )
    ax.set_xlabel("Standardized coefficient")
    ax.set_title("Fraud risk drivers")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    """Build the dashboard and handle its interactive sections."""
    models = train_models()
    data = models["data"]

    st.title("SentinelGuard: Banking Fraud Detection")
    st.markdown(
        """
        This dashboard uses synthetic card transaction data to compare a few
        different ways of estimating fraud risk. It includes live scoring,
        exploratory analysis, logistic regression, risk-score regression,
        and transaction clustering.
        """
    )

    st.sidebar.header("Test a transaction")

    tx_amount = st.sidebar.number_input(
        "Transaction amount ($)",
        min_value=1.0,
        max_value=10000.0,
        value=250.0,
        step=25.0,
    )
    hour = st.sidebar.slider("Hour of day", 0, 23, 14)
    account_age = st.sidebar.slider("Account age (days)", 1, 2000, 450)
    failed_pins = st.sidebar.slider("Failed PIN attempts", 0, 5, 0)
    international = st.sidebar.checkbox("International transaction")
    assess = st.sidebar.button(
        "Assess transaction risk",
        type="primary",
        use_container_width=True,
    )

    live_tab, eda_tab, class_tab, reg_tab, cluster_tab = st.tabs(
        [
            "Live risk scoring",
            "Data exploration",
            "Logistic regression",
            "Regression and decision tree",
            "Transaction clusters",
        ]
    )

    with live_tab:
        if assess:
            incoming = pd.DataFrame(
                [
                    {
                        "Transaction_Amount": tx_amount,
                        "Hour_Of_Day": hour,
                        "Account_Age_Days": account_age,
                        "Failed_Pin_Attempts": failed_pins,
                        "Is_International": int(international),
                    }
                ]
            )

            scaled_for_class = models["classifier_scaler"].transform(incoming)
            prediction = int(models["classifier"].predict(scaled_for_class)[0])
            probability = float(
                models["classifier"].predict_proba(scaled_for_class)[0][1]
            )

            risk_score = float(
                np.clip(models["full_tree"].predict(incoming)[0], 0.0, 100.0)
            )
            scaled_for_cluster = models["cluster_scaler"].transform(incoming)
            cluster_number = int(
                models["default_clusters"].predict(scaled_for_cluster)[0]
            )

            st.session_state["last_assessment"] = {
                "incoming": incoming,
                "prediction": prediction,
                "probability": probability,
                "risk_score": risk_score,
                "cluster": cluster_number,
            }

        if "last_assessment" in st.session_state:
            result = st.session_state["last_assessment"]
            show_risk_badge(result["probability"])

            first, second, third, fourth = st.columns(4)
            first.metric(
                "Predicted class",
                "Fraud" if result["prediction"] == 1 else "Legitimate",
            )
            second.metric(
                "Fraud probability",
                f"{result['probability'] * 100:.2f}%",
            )
            third.metric(
                "Estimated risk score",
                f"{result['risk_score']:.1f} / 100",
            )
            fourth.metric(
                "Behavior segment",
                f"Cluster {result['cluster']}",
            )

            st.subheader("Transaction details")
            st.dataframe(
                result["incoming"],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "Change the values in the sidebar, then choose "
                "**Assess transaction risk**."
            )

    with eda_tab:
        st.subheader("Data overview")
        one, two, three, four = st.columns(4)
        one.metric("Transactions", data.shape[0])
        two.metric("Columns", data.shape[1])
        three.metric("Fraud cases", int(data["Is_Fraud"].sum()))
        four.metric("Fraud rate", f"{data['Is_Fraud'].mean() * 100:.1f}%")

        left, right = st.columns(2)
        smallest = float(data["Transaction_Amount"].min())
        largest = float(data["Transaction_Amount"].max())
        amount_range = left.slider(
            "Transaction amount range",
            smallest,
            largest,
            (smallest, largest),
        )
        geography = right.selectbox(
            "Transaction type",
            [
                "All transactions",
                "Domestic only",
                "International only",
            ],
        )

        filtered = data[
            (data["Transaction_Amount"] >= amount_range[0])
            & (data["Transaction_Amount"] <= amount_range[1])
        ]
        if geography == "Domestic only":
            filtered = filtered[filtered["Is_International"] == 0]
        elif geography == "International only":
            filtered = filtered[filtered["Is_International"] == 1]

        st.caption(f"Showing {len(filtered)} of {len(data)} records")
        st.dataframe(
            filtered.sort_values(
                "Transaction_Amount",
                ascending=False,
            ).head(15),
            use_container_width=True,
            hide_index=True,
        )

        st.write("Average values by fraud label")
        st.dataframe(
            data.groupby("Is_Fraud")[FEATURES + ["Risk_Score"]]
            .mean()
            .round(2),
            use_container_width=True,
        )

        chart_left, chart_right = st.columns(2)
        with chart_left:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                data=filtered,
                x="Is_Fraud",
                y="Transaction_Amount",
                palette=["#2563EB", "#DC2626"],
                ax=ax,
            )
            ax.set_xticklabels(["Legitimate", "Fraud"])
            ax.set_title("Transaction amount by fraud label")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with chart_right:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                data[FEATURES + ["Is_Fraud"]].corr(),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                ax=ax,
            )
            ax.set_title("Feature correlations")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("Descriptive statistics"):
            st.dataframe(data.describe().round(2), use_container_width=True)

    with class_tab:
        st.subheader("Logistic regression results")
        fraud_metrics = models["class_report"]["Fraud"]

        metric_one, metric_two, metric_three, metric_four = st.columns(4)
        metric_one.metric("Accuracy", f"{models['accuracy'] * 100:.1f}%")
        metric_two.metric(
            "Fraud precision",
            f"{fraud_metrics['precision'] * 100:.1f}%",
        )
        metric_three.metric(
            "Fraud recall",
            f"{fraud_metrics['recall'] * 100:.1f}%",
        )
        metric_four.metric(
            "Fraud F1",
            f"{fraud_metrics['f1-score'] * 100:.1f}%",
        )

        confusion_column, coefficient_column = st.columns(2)
        with confusion_column:
            fig, ax = plt.subplots(figsize=(5.4, 4.2))
            display = ConfusionMatrixDisplay(
                confusion_matrix=models["confusion_matrix"],
                display_labels=["Legitimate", "Fraud"],
            )
            display.plot(cmap="Blues", ax=ax, colorbar=False)
            ax.set_title("Confusion matrix")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with coefficient_column:
            coefficient_fig = draw_coefficients(models["coefficients"])
            st.pyplot(coefficient_fig)
            plt.close(coefficient_fig)

        st.write("Classification report")
        st.dataframe(
            pd.DataFrame(models["class_report"]).transpose().round(3),
            use_container_width=True,
        )

        with st.expander("Test predictions"):
            st.dataframe(
                models["test_results"].head(20),
                use_container_width=True,
                hide_index=True,
            )

    with reg_tab:
        st.subheader("Linear regression")
        slope, intercept, mae, r_squared = st.columns(4)
        slope.metric(
            "Slope",
            f"{models['linear_model'].coef_[0]:.4f} points per dollar",
        )
        intercept.metric(
            "Intercept",
            f"{models['linear_model'].intercept_:0.2f}",
        )
        mae.metric("Test MAE", f"{models['linear_mae']:.2f}")
        r_squared.metric("Test R squared", f"{models['linear_r2']:.3f}")

        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.scatter(
            models["amount_test"]["Transaction_Amount"],
            models["score_test"],
            color="#2563EB",
            alpha=0.7,
            label="Actual",
        )
        line_x = np.linspace(
            data["Transaction_Amount"].min(),
            data["Transaction_Amount"].max(),
            100,
        ).reshape(-1, 1)
        ax.plot(
            line_x,
            models["linear_model"].predict(line_x),
            color="#DC2626",
            linewidth=2.2,
            label="Regression line",
        )
        ax.set_xlabel("Transaction amount ($)")
        ax.set_ylabel("Risk score")
        ax.set_title("Amount compared with estimated risk")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Decision tree")
        max_depth = st.slider(
            "Maximum tree depth",
            min_value=1,
            max_value=6,
            value=2,
        )
        limited_tree = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=DEFAULT_SEED,
        )
        limited_tree.fit(
            models["tree_train"],
            models["tree_score_train"],
        )

        full_tree = models["full_tree"]
        full_train_predictions = full_tree.predict(models["tree_train"])
        full_test_predictions = full_tree.predict(models["tree_test"])
        limited_train_predictions = limited_tree.predict(models["tree_train"])
        limited_test_predictions = limited_tree.predict(models["tree_test"])

        comparison = pd.DataFrame(
            [
                {
                    "Model": "Full tree",
                    "Train R squared": round(
                        r2_score(
                            models["tree_score_train"],
                            full_train_predictions,
                        ),
                        3,
                    ),
                    "Test R squared": round(
                        r2_score(
                            models["tree_score_test"],
                            full_test_predictions,
                        ),
                        3,
                    ),
                    "Test MAE": round(
                        mean_absolute_error(
                            models["tree_score_test"],
                            full_test_predictions,
                        ),
                        2,
                    ),
                    "Depth": full_tree.get_depth(),
                },
                {
                    "Model": f"Limited tree (depth={max_depth})",
                    "Train R squared": round(
                        r2_score(
                            models["tree_score_train"],
                            limited_train_predictions,
                        ),
                        3,
                    ),
                    "Test R squared": round(
                        r2_score(
                            models["tree_score_test"],
                            limited_test_predictions,
                        ),
                        3,
                    ),
                    "Test MAE": round(
                        mean_absolute_error(
                            models["tree_score_test"],
                            limited_test_predictions,
                        ),
                        2,
                    ),
                    "Depth": limited_tree.get_depth(),
                },
            ]
        )
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(11.5, 4.6))
        plot_tree(
            limited_tree,
            feature_names=FEATURES,
            filled=True,
            rounded=True,
            fontsize=9,
            ax=ax,
        )
        ax.set_title(f"Decision tree (maximum depth {max_depth})")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with cluster_tab:
        st.subheader("K-means transaction clusters")
        number_of_clusters = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=5,
            value=3,
        )

        scaled_data = models["cluster_scaler"].transform(data[FEATURES])
        kmeans = KMeans(
            n_clusters=number_of_clusters,
            random_state=8,
            n_init=10,
        )
        kmeans.fit(scaled_data)

        clustered = data.copy()
        clustered["Cluster"] = kmeans.labels_

        summary = (
            clustered.groupby("Cluster")[FEATURES + ["Is_Fraud"]]
            .mean()
            .round(2)
        )
        summary["Fraud_Rate_%"] = (
            summary.pop("Is_Fraud") * 100
        ).round(1)
        summary["Transaction_Count"] = (
            clustered["Cluster"].value_counts().sort_index()
        )

        plot_column, table_column = st.columns([1.2, 1.0])
        with plot_column:
            fig, ax = plt.subplots(figsize=(6.4, 4.3))
            colors = [
                "#2563EB",
                "#16A34A",
                "#EA580C",
                "#9333EA",
                "#DC2626",
            ]
            for cluster_id in sorted(clustered["Cluster"].unique()):
                points = clustered[clustered["Cluster"] == cluster_id]
                ax.scatter(
                    points["Account_Age_Days"],
                    points["Transaction_Amount"],
                    label=(
                        f"Cluster {cluster_id} "
                        f"(n={len(points)})"
                    ),
                    color=colors[cluster_id % len(colors)],
                    alpha=0.75,
                    s=45,
                )
            ax.set_xlabel("Account age (days)")
            ax.set_ylabel("Transaction amount ($)")
            ax.set_title("Amount compared with account age")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with table_column:
            st.write("Cluster averages")
            st.dataframe(summary, use_container_width=True)

            st.write("Cluster profiles")
            for cluster_id, row in summary.iterrows():
                st.markdown(
                    f"- **Cluster {cluster_id}** "
                    f"({int(row['Transaction_Count'])} transactions, "
                    f"{row['Fraud_Rate_%']}% fraud): average "
                    f"${row['Transaction_Amount']}, account age "
                    f"{row['Account_Age_Days']} days, "
                    f"{row['Failed_Pin_Attempts']} failed PIN attempts."
                )


if __name__ == "__main__":
    main()
