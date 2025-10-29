# 🏦 Smart Loan Recommendation System

**AI-Powered Loan Default Prediction System using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1+-green.svg)](https://xgboost.readthedocs.io/)
[![AUC](https://img.shields.io/badge/AUC-98.7%25-brightgreen.svg)]()

---

## 🎯 Overview

A complete machine learning system that predicts loan defaults and provides instant approval decisions. Built with Logistic Regression and XGBoost models achieving **98.7% AUC**.

### Key Features
- ✅ Predictive system using structured financial data (7 features)
- ✅ ML Models: Logistic Regression (96.4% AUC) + XGBoost (98.7% AUC)
- ✅ Probability of Default (PD) estimation
- ✅ Interactive Streamlit dashboard
- ✅ Instant loan decisions (2 seconds)
- ✅ Risk assessment: Approve/Reject/Review

---

## 📊 Dataset

**Simple, clean dataset with 7 essential features:**

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `AGE` | Borrower age in years | Direct input from application |
| `ANNUAL_INCOME` | Yearly income (₹) | Total gross income before taxes |
| `CREDIT_SCORE` | CIBIL score (300-900) | Calculated by credit bureau based on payment history, amounts owed, credit length, etc. |
| `LOAN_AMOUNT` | Requested loan amount (₹) | Amount borrower wants to borrow |
| `YEARS_AT_JOB` | Employment stability | Years employed at current job |
| `EXISTING_LOANS` | Current loan count | Number of active loans (mortgage, car, personal, etc.) |
| `DEBT_TO_INCOME_RATIO` | Financial burden | (Total Monthly Debt / Monthly Income) × 100 |

### 📈 Key Calculated Features

#### **1. Debt-to-Income Ratio (DTI)**
**Formula:** `(Total Monthly Debt Payments / Gross Monthly Income) × 100%`

**Example:**
- Monthly debt: ₹15,000 (EMI + credit cards + other loans)
- Monthly income: ₹50,000
- DTI = (15,000 / 50,000) × 100 = **30%**

**Risk Interpretation:**
- DTI < 30%: Low risk ✅
- DTI 30-50%: Moderate risk ⚠️
- DTI > 50%: High risk ❌

#### **2. Probability of Default (PD)**
**What it is:** Likelihood (0-1 or 0-100%) that borrower will default on loan

**How it's calculated:** Using Machine Learning models

**Process:**
1. Train models on historical data (features → actual outcomes)
2. Models learn patterns (e.g., low income + high DTI = more defaults)
3. Apply to new applicants to predict PD score

**Example Predictions:**
```
Applicant A: PD = 0.15 (15%) → Low Risk → APPROVE
Applicant B: PD = 0.65 (65%) → High Risk → REJECT
```

**Stats:**
- Training samples: 5,000
- Test samples: 1,000
- Default rate: 9%

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python3 simple_main.py
```

**Output:**
```
Logistic Regression: AUC 0.9638 (96.4%)
XGBoost:           AUC 0.9868 (98.7%) ⭐
```

### 3. Run Dashboard
```bash
streamlit run simple_dashboard.py
```

Open http://localhost:8501

---

## 📁 Project Structure

```
Loan_System/
├── README.md                    # This file
├── simple_main.py               # Train models
├── simple_dashboard.py          # Interactive UI
├── requirements.txt             # Dependencies
├── src/
│   ├── simple_data_loader.py   # Load data
│   └── simple_model_trainer.py # Train models
├── data/
│   ├── application_train_simple.csv  # 5,000 samples
│   └── application_test_simple.csv   # 1,000 samples
└── models/
    └── simple_model.pkl         # Trained model
```

---

## 🎯 How It Works

```
1. User enters financial information
   ↓
2. ML models analyze risk profile
   ↓
3. Probability of Default (PD) calculated
   ↓
4. Decision made:
   - PD < 30% → APPROVED
   - PD 30-70% → REVIEW
   - PD > 70% → REJECTED
   ↓
5. Show result with loan terms
```

---

## 📊 Model Performance

| Model | AUC | Accuracy |
|-------|-----|----------|
| Logistic Regression | 96.4% | 94.4% |
| **XGBoost** | **98.7%** | **95.6%** |

**Feature Importance (What matters most):**
1. **Employment stability (YEARS_AT_JOB)** - 49%
   - Long-term employment = lower risk
   - Shows consistent income source
2. **Debt-to-income ratio** - 31%
   - Lower DTI = can afford more loans
   - Shows financial burden level
3. **Annual income** - 7%
   - Higher income = lower default risk
   - Shows repayment capacity
4. **Credit score** - 6%
   - Past payment behavior indicator
   - Higher score = better history
5. **Loan amount** - 2%
   - Larger loans = higher risk
   - Repayment burden consideration
6. **Age** - 2%
   - Age affects risk profile
   - Too young/old = slightly riskier
7. **Existing loans** - 2%
   - More loans = more financial pressure

---

## 💡 Real-World Example

**Scenario:** Rahul applies for ₹10 Lakh personal loan

**His Profile:**
- Age: 35 years
- Annual Income: ₹6,00,000 (₹50,000/month)
- Credit Score: 720 (Good)
- Loan Amount: ₹10,00,000
- Years at Job: 5 years (Stable)
- Existing Loans: 1 (Car loan)
- Monthly Debt Payments: ₹8,000
- **DTI Ratio:** (8,000 / 50,000) × 100 = **16%**

**ML Model Analysis:**
- YEARS_AT_JOB (49% importance) = 5 years ✅ Low risk
- DTI (31% importance) = 16% ✅ Low risk
- ANNUAL_INCOME (7%) = ₹6L ✅ Good
- CREDIT_SCORE (6%) = 720 ✅ Good

**Model Prediction:**
- PD Score: 0.12 (12% probability of default)
- **Decision: APPROVED** ✅
- Interest Rate: 11.5% p.a.
- EMI: ₹21,456/month for 5 years

---

## 💼 Business Impact

- **Speed:** Instant decisions (2 seconds vs. weeks)
- **Accuracy:** 98.7% AUC identifies risky borrowers
- **Efficiency:** Automated processing for 80% of applications
- **Cost Savings:** Reduces bad loans and operational costs

---

## 🛠️ Technologies Used

- **Python** - Core language
- **XGBoost** - Gradient boosting model
- **Scikit-learn** - Logistic Regression
- **Streamlit** - Web interface
- **Pandas** - Data processing
- **NumPy** - Numerical operations

---

## 📝 Resume Points Demonstrated

✅ **Predictive system** for personalized loan recommendations  
✅ **ML models** (Logistic Regression, XGBoost) estimating PD  
✅ **Interactive dashboard** (Streamlit) visualizing risk  
✅ **Fintech solution** improving lending efficiency  

---

## 📄 License

MIT License - Free to use and modify

---

## 👤 Author

Luqmaan  
GitHub: [@Luqmaan29](https://github.com/Luqmaan29)

---

<div align="center">
  
**Built with ❤️ for smarter lending decisions**
  
</div>

