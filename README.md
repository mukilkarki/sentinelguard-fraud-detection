# SentinelGuard: Banking Fraud Detection & Risk Scoring System

**SentinelGuard** is an interactive Python and Streamlit FinTech web application that simulates real-time banking fraud detection and risk scoring. It is built strictly using foundational concepts from introductory **Python**, **Data Analysis (NumPy, Pandas, Matplotlib, Seaborn)**, and **Machine Learning (Scikit-Learn)**.

**Live Demo:** [https://sentinelguard-fraud-detection-by-mukilkarki.streamlit.app/](https://sentinelguard-fraud-detection-by-mukilkarki.streamlit.app/)

## Business Problem

Retail banks process thousands of card transactions every minute, yet fewer than **5%** are typically fraudulent. Missing a fraudulent charge causes direct financial loss, while wrongly blocking a legitimate customer transaction hurts trust and retention. SentinelGuard provides an interpretable risk-scoring workflow that classifies transactions into three actionable tiers:

| Risk Tier | Predicted Fraud Probability | Recommended Action |
| :--- | :--- | :--- |
| **Safe (Green)** | `< 30%` | Approve transaction automatically |
| **Flagged (Yellow)** | `30% – 65%` | Trigger step-up verification (2FA / manual review) |
| **High-Risk Block (Red)** | `> 65%` | Decline transaction and alert cardholder |

## Curriculum Alignment & Features

SentinelGuard brings together all three training modules in a single, runnable Streamlit application:

| Module | Concepts Applied in SentinelGuard | App Location |
| :--- | :--- | :--- |
| **Module 1: Python Fundamentals** | Functions, dictionaries, lists, loops, conditionals (`if`/`elif`/`else`), string formatting | Risk tier routing & modular helpers |
| **Module 2: Data Analysis & EDA** | `NumPy`, `Pandas` (`shape`, `describe`, filtering, sorting, `groupby().mean()`), `Matplotlib`, `Seaborn` boxplots & correlation heatmaps | **Data Analysis & EDA** tab |
| **Module 3.1: Linear Regression** | 80/20 `train_test_split`, `LinearRegression`, slope/intercept, **MAE**, **R² score**, regression line plot | **Risk Score Regression & Tree** tab |
| **Module 3.2: Decision Tree Regression** | `DecisionTreeRegressor`, comparing full vs. depth-limited (`max_depth`) trees for overfitting vs. generalization, `plot_tree` visualization | **Risk Score Regression & Tree** tab |
| **Module 3.3: Logistic Regression** | Stratified 80/20 split (`stratify=y`), `StandardScaler`, `LogisticRegression(max_iter=1000, class_weight="balanced")`, `predict_proba()`, `accuracy_score`, `ConfusionMatrixDisplay`, `classification_report`, feature coefficients bar chart | **Live Risk Scoring** & **Logistic Regression Analytics** tabs |
| **Module 3.4: K-Means Clustering** | `StandardScaler`, `KMeans(n_clusters=3, random_state=8, n_init=10)`, `value_counts()`, `groupby("Cluster").mean()`, cluster scatter plot & segment profiles | **K-Means Transaction Clusters** tab |

## Synthetic Dataset Schema

The application generates a reproducible 500-transaction dataset (~5% fraud rate, `random_state=42`) directly inside `app.py` with no external CSV required:

| Column | Type | Description |
| :--- | :--- | :--- |
| `Transaction_Amount` | Float | Transaction value in USD (`$`) |
| `Hour_Of_Day` | Integer | Hour of the transaction in 24-hour format (`0–23`) |
| `Account_Age_Days` | Integer | Age of the customer's bank account in days |
| `Failed_Pin_Attempts` | Integer | Failed PIN entries prior to the transaction (`0–5`) |
| `Is_International` | Binary | `1` for cross-border transaction, `0` for domestic |
| `Is_Fraud` | Binary | Target label (`1 = Fraudulent`, `0 = Legitimate`) |
| `Risk_Score` | Float | Continuous risk score (`0–100`) used for regression labs |

## Project Structure

```text
sentinelguard-fraud-detection/
├── app.py              # Complete single-file Streamlit application
├── requirements.txt    # Python dependencies for local & Streamlit Cloud runs
└── README.md           # Project documentation
```

## Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/sentinelguard-fraud-detection.git
   cd sentinelguard-fraud-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deploy Free on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to the root of a public GitHub repository.
2. Sign in with GitHub at [share.streamlit.io](https://share.streamlit.io/).
3. Click **Create app**, choose your repository (`sentinelguard-fraud-detection`), set branch to `main` and main file path to `app.py`, and click **Deploy**.

## Tech Stack

- **Language:** Python 3
- **Web Framework:** Streamlit
- **Data Analysis:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn
