# Credit Rating Predictor

A Streamlit web app that predicts whether a person's credit rating is **Good** or **Poor** from ten financial details they would normally know. It's a portfolio project, not financial advice.

## Model

- **Algorithm:** logistic regression in a scikit-learn `Pipeline` with a `StandardScaler`, saved to `model.pkl`.
- **Data:** the Kaggle [Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) dataset. The notebook drops rows labelled `Standard`, so the model only learns Good vs Poor. The data isn't included in this repo; the notebook downloads it with `kagglehub`.
- **Cleaning:** outliers outside 3×IQR are replaced with the column median, and rows with impossible ages are dropped.
- **Tuning:** grid search over penalty, `C` and class weighting, using `StratifiedGroupKFold` grouped by customer so no customer appears in both train and test.
- **Test set results:** macro F1 0.84, ROC AUC 0.91.

Training code is in [`logistic_prototype.ipynb`](logistic_prototype.ipynb).

## Input features

| Feature | Description | Accepted range |
|---|---|---|
| `Age` | Age in years | 16 to 100 |
| `Monthly_Inhand_Salary` | Take-home pay per month | 0 to 25,000 |
| `Num_Bank_Accounts` | Number of bank accounts | 0 to 20 |
| `Num_Credit_Card` | Number of credit cards | 0 to 20 |
| `Interest_Rate` | Credit card interest rate (%) | 0 to 100 |
| `Num_of_Loan` | Number of loans | 0 to 20 |
| `Outstanding_Debt` | Total debt still owed | 0 to 10,000,000 |
| `Credit_History_Age` | Full years since the first credit account was opened | 0 to 84 |
| `Amount_invested_monthly` | Amount invested each month | 0 to 25,000 |
| `Monthly_Balance` | Money left at the end of a typical month | 0 to 1,000,000 |

## Run locally

Python 3.12 is recommended (the pinned versions were tested on 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run credit_app.py
```

The app then opens at http://localhost:8501.

## Retrain the model

Run every cell in `logistic_prototype.ipynb`. The last cell overwrites `model.pkl`. The notebook also needs `kagglehub`, `matplotlib` and `pandas-dq`, which aren't in `requirements.txt` because the app doesn't use them. Keep `scikit-learn` in `requirements.txt` at the version you trained with.
