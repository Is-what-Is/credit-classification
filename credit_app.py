from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model.pkl"

# Feature order must match the order the model was trained on (see logistic_prototype.ipynb).
# Each entry: column name -> label, description, min, max, default, step, format.
# Integer min/max/default/step give an integer input; floats give a decimal input.
# Defaults are close to the training data medians.
FEATURES = {
    "Age": (
        "Age", "your age in years.",
        16, 100, 33, 1, "%d",
    ),
    "Monthly_Inhand_Salary": (
        "Monthly in-hand salary", "take-home pay per month.",
        0.0, 25_000.0, 3_100.0, 50.0, "%.2f",
    ),
    "Num_Bank_Accounts": (
        "Number of bank accounts", "how many bank accounts you hold.",
        0, 20, 6, 1, "%d",
    ),
    "Num_Credit_Card": (
        "Number of credit cards", "how many credit cards you hold.",
        0, 20, 5, 1, "%d",
    ),
    "Interest_Rate": (
        "Interest rate (%)", "the interest rate on your credit card, as a percentage.",
        0.0, 100.0, 14.0, 0.5, "%.1f",
    ),
    "Num_of_Loan": (
        "Number of loans", "how many loans you currently have.",
        0, 20, 3, 1, "%d",
    ),
    "Outstanding_Debt": (
        "Outstanding debt", "the total debt you still have to repay.",
        0.0, 10_000_000.0, 1_350.0, 50.0, "%.2f",
    ),
    "Credit_History_Age": (
        "Credit history age (years)", "how many full years since you opened your first credit account.",
        0, 84, 16, 1, "%d",
    ),
    "Amount_invested_monthly": (
        "Amount invested monthly", "how much you invest each month.",
        0.0, 25_000.0, 125.0, 10.0, "%.2f",
    ),
    "Monthly_Balance": (
        "Monthly balance", "the money left in your account at the end of a typical month.",
        0.0, 1_000_000.0, 330.0, 10.0, "%.2f",
    ),
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def format_range(low, high):
    return f"{low:,.0f} to {high:,.0f}"


st.set_page_config(page_title="Credit Rating Predictor", page_icon="💳")

st.title("Credit Rating Predictor")
st.write(
    "This app uses a logistic regression model to predict whether a credit rating is "
    "**Good** or **Poor** from ten financial details. It was trained on the Kaggle "
    "[Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) dataset."
)
st.info("This is a portfolio demo, not financial advice.", icon="ℹ️")

model = load_model()

with st.form("credit_form"):
    inputs = {}
    columns = st.columns(2)
    for i, (name, (label, description, low, high, default, step, fmt)) in enumerate(FEATURES.items()):
        with columns[i % 2]:
            st.markdown(f"**{label}**: {description}  \nAccepted range: {format_range(low, high)}.")
            inputs[name] = st.number_input(
                label,
                min_value=low,
                max_value=high,
                value=default,
                step=step,
                format=fmt,
                key=name,
                label_visibility="collapsed",
            )
    submitted = st.form_submit_button("Predict credit rating", type="primary", use_container_width=True)

if submitted:
    X = pd.DataFrame([inputs], columns=list(FEATURES))
    proba_good = model.predict_proba(X)[0, list(model.classes_).index(1)]
    is_good = proba_good >= 0.5
    rating, confidence, colour = ("GOOD", proba_good, "#1a7f37") if is_good else ("POOR", 1 - proba_good, "#cf222e")

    st.markdown(
        f"""
        <div style="border: 2px solid {colour}; border-radius: 0.5rem; padding: 1rem 1.25rem; margin-top: 1rem;">
            <div style="font-size: 2rem; font-weight: 700; color: {colour};">
                Predicted credit rating: {rating}
            </div>
            <div style="font-size: 1.25rem; margin-top: 0.25rem;">
                Model confidence: {confidence:.1%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Confidence is the model's estimated probability for the predicted rating, not a guarantee. "
        "Inputs far outside the values seen in training (e.g. very large debts) make the estimate less reliable."
    )
