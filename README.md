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
- `AGE` - Borrower age
- `ANNUAL_INCOME` - Yearly income (₹)
- `CREDIT_SCORE` - CIBIL score (300-900)
- `LOAN_AMOUNT` - Requested amount (₹)
- `YEARS_AT_JOB` - Employment stability
- `EXISTING_LOANS` - Current loan count
- `DEBT_TO_INCOME_RATIO` - Financial burden

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

**Feature Importance:**
1. Employment stability (YEARS_AT_JOB) - 49%
2. Debt-to-income ratio - 31%
3. Annual income - 7%
4. Credit score - 6%
5. Loan amount - 2%
6. Age - 2%
7. Existing loans - 2%

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

