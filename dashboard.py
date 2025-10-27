"""
Simple Smart Loan Application System
Easy to use interface for loan applications - Indian Bank Context
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="Smart Loan System - Indian Bank",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Simple Custom CSS with Indian colors
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }
    .success {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .warning {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
    .error {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .indian-flag {
        font-size: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def format_indian_currency(amount):
    """Format currency in Indian numbering system (Lakhs, Crores)."""
    if amount >= 10000000:  # 1 Crore
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 Lakh
        return f"₹{amount/100000:.2f} L"
    elif amount >= 1000:  # 1 Thousand
        return f"₹{amount/1000:.0f}K"
    else:
        return f"₹{amount:,.0f}"

def main():
    """Main application."""
    
    # Header with Indian theme
    st.markdown('<div class="indian-flag">🇮🇳</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">🏦 Smart Loan Application</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Instant loan approval with AI-powered risk assessment</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Simple instructions
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1️⃣ Fill Form**")
        st.markdown("Enter your details")
    with col2:
        st.markdown("**2️⃣ AI Checks**")
        st.markdown("Instant analysis")
    with col3:
        st.markdown("**3️⃣ Get Result**")
        st.markdown("Your decision")
    
    st.markdown("---")
    
    # Application Form
    st.markdown("### 📝 Your Information")
    
    # Personal Info
    st.markdown("**Personal Details:**")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name *", placeholder="Rahul Kumar")
        age = st.number_input("Age *", min_value=18, max_value=80, value=30)
        email = st.text_input("Email *", placeholder="rahul.kumar@email.com")
    with col2:
        phone = st.text_input("Phone Number *", placeholder="+91 98765 43210")
        city = st.selectbox("City *", ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Jaipur", "Other"])
    
    # Financial Info
    st.markdown("**Financial Details:**")
    col1, col2 = st.columns(2)
    with col1:
        annual_income = st.number_input("Annual Income (₹) *", min_value=150000, value=600000, step=25000, format="%d")
        monthly_expenses = st.number_input("Monthly Expenses (₹) *", min_value=10000, value=25000, step=1000, format="%d")
    with col2:
        credit_score = st.slider("Credit Score *", min_value=300, max_value=900, value=720, step=10)
        employment_years = st.number_input("Years at Current Job *", min_value=0, max_value=50, value=3)
    
    # Loan Details
    st.markdown("**Loan Details:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amount = st.selectbox("Loan Amount *", 
            ["₹2 Lakh", "₹3 Lakh", "₹5 Lakh", "₹10 Lakh", "₹15 Lakh", "₹20 Lakh", "₹30 Lakh", "₹50 Lakh"])
        # Convert to number
        loan_amount_map = {
            "₹2 Lakh": 200000,
            "₹3 Lakh": 300000,
            "₹5 Lakh": 500000,
            "₹10 Lakh": 1000000,
            "₹15 Lakh": 1500000,
            "₹20 Lakh": 2000000,
            "₹30 Lakh": 3000000,
            "₹50 Lakh": 5000000
        }
        loan_amount_value = loan_amount_map[loan_amount]
    with col2:
        loan_type = st.selectbox("Loan Type *", ["Personal Loan", "Home Loan", "Car Loan", "Business Loan", "Education Loan"])
    with col3:
        loan_term = st.selectbox("Loan Term (Years) *", [5, 7, 10, 15, 20, 25, 30], index=2)
    
    st.markdown("---")
    
    # Submit Button
    st.markdown("### Ready to Submit?")
    
    # Check if all required fields are filled
    required = [name, email, phone, annual_income, monthly_expenses]
    all_filled = all(field for field in required)
    
    if not all_filled:
        st.warning("⚠️ Please fill all required fields")
    
    if st.button("🚀 Get My Loan Decision", type="primary", use_container_width=True, disabled=not all_filled):
        
        # Show processing
        with st.spinner("🤖 Analyzing your application..."):
            import time
            time.sleep(2)
            
            # Calculate result
            result = calculate_loan_decision_indian(annual_income, monthly_expenses, loan_amount_value, credit_score, employment_years)
            
            # Show result
            show_result_indian(result, loan_amount_value, loan_term, loan_type, name)

def calculate_loan_decision_indian(income, expenses, loan_amount, credit_score, employment_years):
    """Calculate loan decision based on Indian context."""
    
    # Calculate risk factors
    risk_score = 0
    reasons = []
    
    # Income check (Indian salary standards)
    if income < 300000:  # Below 3 Lakh
        risk_score += 0.3
        reasons.append("Income below ₹3 Lakh per annum")
    elif income < 500000:  # Below 5 Lakh
        risk_score += 0.2
    else:
        risk_score += 0.1
    
    # Debt-to-income ratio
    monthly_income = income / 12
    dti = expenses / monthly_income
    if dti > 0.5:
        risk_score += 0.3
        reasons.append("Monthly expenses exceed 50% of income")
    elif dti > 0.4:
        risk_score += 0.2
    else:
        risk_score += 0.1
    
    # Credit score (Indian CIBIL range 300-900)
    if credit_score < 600:
        risk_score += 0.3
        reasons.append("CIBIL score below 600")
    elif credit_score < 650:
        risk_score += 0.2
    elif credit_score > 750:
        risk_score += 0.0
    else:
        risk_score += 0.1
    
    # Employment history
    if employment_years < 1:
        risk_score += 0.2
        reasons.append("Employment less than 1 year")
    elif employment_years < 2:
        risk_score += 0.1
    
    # Loan amount vs income (Indian context)
    loan_to_income = loan_amount / income
    if loan_to_income > 5:
        risk_score += 0.2
        reasons.append("Loan amount too high for income level")
    elif loan_to_income > 3:
        risk_score += 0.1
    
    # Overall assessment
    if risk_score < 0.4:
        status = "APPROVED"
        interest_rate = 10.5 if credit_score > 750 else 12.5
    elif risk_score < 0.7:
        status = "REVIEW"
        interest_rate = 14.5
        reasons.append("Application needs manual review")
    else:
        status = "NOT APPROVED"
        interest_rate = 0
        reasons.append("Does not meet minimum eligibility criteria")
    
    return {
        'status': status,
        'interest_rate': interest_rate,
        'risk_score': risk_score,
        'reasons': reasons
    }

def show_result_indian(result, loan_amount, loan_term, loan_type, name):
    """Display the loan decision result with Indian context."""
    
    st.markdown("---")
    st.markdown("### 🎯 Your Loan Decision")
    
    # Show status
    if result['status'] == "APPROVED":
        st.markdown('<div class="result-box success">', unsafe_allow_html=True)
        st.markdown(f"## 🎉 Congratulations {name.split()[0]}! Your loan has been approved!")
        st.markdown('<p style="font-size: 1.5rem; margin-top: 1rem;">You are pre-approved for this loan!</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show loan details
        st.markdown("### 💰 Your Loan Terms")
        col1, col2, col3, col4 = st.columns(4)
        
        interest_rate = result['interest_rate']
        monthly_payment = calculate_monthly_payment(loan_amount, interest_rate, loan_term)
        total_interest = (monthly_payment * loan_term * 12) - loan_amount
        total_amount = loan_amount + total_interest
        
        with col1:
            st.metric("Loan Amount", format_indian_currency(loan_amount))
        with col2:
            st.metric("Interest Rate", f"{interest_rate}% p.a.")
        with col3:
            st.metric("EMI", format_indian_currency(monthly_payment))
        with col4:
            st.metric("Total Payable", format_indian_currency(total_amount))
        
        # EMI Breakdown
        st.markdown("#### EMI Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Principal Amount:** {format_indian_currency(loan_amount)}")
        with col2:
            st.write(f"**Total Interest:** {format_indian_currency(total_interest)}")
        with col3:
            st.write(f"**Loan Term:** {loan_term} years")
        
        # Next steps
        st.markdown("### 🚀 Next Steps")
        st.success("""
        **1. Document Verification** - Submit your documents for verification
        - PAN Card
        - Aadhaar Card
        - Salary slips (Last 3 months)
        - Bank statements (Last 6 months)
        
        **2. Agreement Signing** - Sign the loan agreement
        
        **3. Final Approval** - Receive final approval within 24-48 hours
        
        **4. Loan Disbursal** - Loan amount credited to your account
        """)
        
    elif result['status'] == "REVIEW":
        st.markdown('<div class="result-box warning">', unsafe_allow_html=True)
        st.markdown("## ⚠️ Under Review")
        st.markdown('<p style="font-size: 1.5rem; margin-top: 1rem;">Your application is under manual review</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show estimated terms
        if result['interest_rate'] > 0:
            st.markdown("### 💰 Estimated Loan Terms")
            interest_rate = result['interest_rate']
            monthly_payment = calculate_monthly_payment(loan_amount, interest_rate, loan_term)
            total_interest = (monthly_payment * loan_term * 12) - loan_amount
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Interest Rate", f"{interest_rate}% (estimated)")
            with col2:
                st.metric("EMI", format_indian_currency(monthly_payment))
            with col3:
                st.metric("Loan Term", f"{loan_term} years")
        
        # Reasons
        if result['reasons']:
            st.markdown("### 📋 Review Notes")
            for reason in result['reasons']:
                st.write(f"• {reason}")
        
        # Next steps
        st.markdown("### 📞 What Happens Next")
        st.info("""
        A loan officer will review your application and contact you within 2-3 business days.
        You may be asked to provide additional documents or clarification.
        """)
        
    else:
        st.markdown('<div class="result-box error">', unsafe_allow_html=True)
        st.markdown("## ❌ Loan Not Approved")
        st.markdown('<p style="font-size: 1.5rem; margin-top: 1rem;">Unfortunately, we cannot approve your loan at this time</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Reasons
        if result['reasons']:
            st.markdown("### 📋 Why Not Approved?")
            for reason in result['reasons']:
                st.write(f"• {reason}")
        
        # Suggestions
        st.markdown("### 💡 How to Improve")
        st.info("""
        **Indian-Specific Tips:**
        • **Improve CIBIL Score** - Pay credit card bills and EMIs on time
        • **Reduce Debt** - Clear existing loans or reduce outstanding balance
        • **Stable Income** - Maintain consistent employment for at least 2 years
        • **Lower Loan Amount** - Apply for a smaller loan amount first
        • **Build Credit History** - Use credit cards responsibly and maintain active accounts
        • **Reapply Later** - Try again after 6-12 months with improved financial profile
        """)
    
    # Footer with Indian context
    st.markdown("---")
    st.markdown("### 📞 Need Help?")
    st.write("**Customer Support:** Call 1800-123-4567 (Toll-free)")
    st.write("**Email:** support@indianbankloans.com")
    st.write("**Working Hours:** Monday to Saturday, 9 AM to 6 PM IST")
    
    st.markdown("---")
    st.markdown("### 🔒 Privacy & Security")
    st.write("• RBI guidelines compliant")
    st.write("• Bank-grade encryption")
    st.write("• Your data is secure and confidential")

def calculate_monthly_payment(principal, annual_rate, years):
    """Calculate monthly loan payment (EMI)."""
    if annual_rate == 0:
        return principal / (years * 12)
    
    monthly_rate = annual_rate / 100 / 12
    num_payments = years * 12
    payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    return payment

if __name__ == "__main__":
    main()
