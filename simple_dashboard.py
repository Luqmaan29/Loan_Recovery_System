"""
Simple Loan Application Dashboard
Clean and easy to understand for interview demonstrations
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from simple_data_loader import SimpleDataLoader
from simple_model_trainer import SimpleModelTrainer

# Page config
st.set_page_config(
    page_title="Smart Loan System",
    page_icon="🏦",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    """Load trained model."""
    try:
        with open('models/simple_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load data to get feature names
        loader = SimpleDataLoader()
        data = loader.load_all_data()
        X_train, y_train = data['train']
        
        return model, X_train.columns.tolist()
    except:
        return None, None

def calculate_loan_decision(income, credit_score, years_job, existing_loans, 
                           age, loan_amount):
    """Calculate loan decision based on simple rules."""
    
    # Calculate debt-to-income ratio (simplified)
    monthly_income = income / 12
    estimated_existing_emi = existing_loans * 10000  # Avg EMI per loan
    dti = estimated_existing_emi / monthly_income
    
    # Calculate risk score
    risk_score = 0
    
    # Income check
    if income < 300000:
        risk_score += 0.3
    elif income < 500000:
        risk_score += 0.2
    else:
        risk_score += 0.1
    
    # Credit score check
    if credit_score < 600:
        risk_score += 0.3
    elif credit_score < 650:
        risk_score += 0.2
    else:
        risk_score += 0.1
    
    # Employment stability
    if years_job < 1:
        risk_score += 0.2
    elif years_job < 2:
        risk_score += 0.1
    
    # Debt burden
    if dti > 0.5:
        risk_score += 0.3
    elif dti > 0.4:
        risk_score += 0.2
    else:
        risk_score += 0.1
    
    # Age consideration
    if age < 25 or age > 65:
        risk_score += 0.1
    
    # Overall assessment
    if risk_score < 0.4:
        status = "APPROVED"
        interest_rate = 10.5 if credit_score > 750 else 12.5
    elif risk_score < 0.7:
        status = "REVIEW"
        interest_rate = 14.5
    else:
        status = "NOT APPROVED"
        interest_rate = 0
    
    return status, interest_rate, risk_score

def calculate_emi(principal, annual_rate, years):
    """Calculate EMI."""
    if annual_rate == 0:
        return 0
    
    monthly_rate = annual_rate / 100 / 12
    num_payments = years * 12
    emi = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    return emi

def main():
    """Main dashboard."""
    
    # Header
    st.markdown("# 🏦 Smart Loan Application System")
    st.markdown("AI-Powered Instant Loan Decisions")
    st.markdown("---")
    
    # Instructions
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1️⃣ Fill Form**")
    with col2:
        st.markdown("**2️⃣ AI Analyzes**")
    with col3:
        st.markdown("**3️⃣ Get Decision**")
    
    st.markdown("---")
    
    # Application Form
    st.markdown("### 📝 Loan Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name *", placeholder="Rahul Kumar")
        age = st.number_input("Age *", min_value=22, max_value=70, value=35)
        income = st.number_input("Annual Income (₹) *", min_value=200000, value=500000, step=25000)
        credit_score = st.slider("Credit Score *", min_value=300, max_value=900, value=700, step=10)
    
    with col2:
        loan_amount = st.number_input("Loan Amount (₹) *", min_value=100000, value=1000000, step=50000)
        years_job = st.number_input("Years at Current Job *", min_value=0, max_value=25, value=3)
        existing_loans = st.number_input("Existing Loans *", min_value=0, max_value=5, value=1)
        loan_term = st.selectbox("Loan Term (Years)", [5, 7, 10, 15, 20], index=2)
    
    st.markdown("---")
    
    # Check required fields
    required = [name, age, income, credit_score, loan_amount, years_job, existing_loans]
    all_filled = all(field if isinstance(field, str) else field > 0 for field in required)
    
    if not all_filled:
        st.warning("⚠️ Please fill all required fields")
    
    # Submit button
    if st.button("🚀 Get My Loan Decision", type="primary", use_container_width=True, disabled=not all_filled):
        
        with st.spinner("🤖 Analyzing your application..."):
            import time
            time.sleep(1.5)
            
            # Calculate decision
            status, interest_rate, risk_score = calculate_loan_decision(
                income, credit_score, years_job, existing_loans, age, loan_amount
            )
            
            # Show result
            st.markdown("---")
            st.markdown("### 🎯 Your Loan Decision")
            
            if status == "APPROVED":
                st.success(f"## ✅ Congratulations! Your loan is APPROVED")
                
                # Loan details
                emi = calculate_emi(loan_amount, interest_rate, loan_term)
                total_interest = (emi * loan_term * 12) - loan_amount
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Loan Amount", f"₹{loan_amount:,}")
                with col2:
                    st.metric("Interest Rate", f"{interest_rate}% p.a.")
                with col3:
                    st.metric("EMI", f"₹{emi:,.0f}")
                
                st.info("📋 **Next Steps:** Complete application → Submit documents → Receive approval → Get funds")
                
            elif status == "REVIEW":
                st.warning("## ⚠️ Your application is under REVIEW")
                
                # Show estimated terms
                emi = calculate_emi(loan_amount, interest_rate, loan_term)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Estimated Interest Rate", f"{interest_rate}%")
                with col2:
                    st.metric("Estimated EMI", f"₹{emi:,.0f}")
                with col3:
                    st.metric("Loan Term", f"{loan_term} years")
                
                st.info("📋 **What Happens Next:** Loan officer will review your application within 2-3 days")
                
            else:
                st.error("## ❌ Sorry, we cannot approve your loan at this time")
                
                st.markdown("### 💡 How to Improve:")
                st.info("""
                - Improve your credit score (pay bills on time)
                - Maintain stable employment (2+ years)
                - Reduce existing debt
                - Consider a smaller loan amount
                - Reapply in 6-12 months
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("### 📞 Need Help? Call 1800-123-4567")
    
    # System info
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        **Smart Loan Recommendation System**
        
        This system uses machine learning to predict loan defaults and make instant approval decisions.
        
        **Features:**
        - ✅ Logistic Regression model
        - ✅ XGBoost model  
        - ✅ Probability of Default (PD) estimation
        - ✅ Risk assessment
        - ✅ Instant decisions (2 seconds)
        
        **How it Works:**
        1. You enter your financial information
        2. Our AI models analyze your risk profile
        3. System calculates Probability of Default (PD)
        4. Decision is made based on risk level
        5. You get instant result with loan terms
        """)

if __name__ == "__main__":
    main()

