"""
Generate Simple Loan Dataset with Essential Features Only
Perfect for demonstrating the core concept without complexity
"""

import pandas as pd
import numpy as np

def generate_simple_loan_data(n_samples=5000):
    """
    Generate simple loan application data with only essential features.
    
    Args:
        n_samples: Number of loan applications to generate
        
    Returns:
        DataFrame with loan application data
    """
    np.random.seed(42)
    
    # Generate customer IDs
    ids = range(1001, 1001 + n_samples)
    
    # AGE - Important risk factor
    age = np.random.normal(42, 12, n_samples).astype(int)
    age = np.clip(age, 22, 70)
    
    # ANNUAL INCOME - Critical for loan decisions
    income = np.random.lognormal(mean=12.5, sigma=0.9, size=n_samples)
    income = (income / 1000).astype(int) * 1000  # Round to nearest 1000
    income = np.clip(income, 200000, 10000000)  # ₹2 Lakh to ₹1 Crore
    
    # CREDIT SCORE - Another critical factor (300-900 CIBIL range)
    credit_score = np.random.normal(650, 100, n_samples).astype(int)
    credit_score = np.clip(credit_score, 300, 900)
    
    # LOAN AMOUNT REQUESTED
    loan_amount = (income * np.random.uniform(1.5, 4, n_samples)).astype(int)
    loan_amount = np.clip(loan_amount, 100000, 5000000)  # ₹1 Lakh to ₹50 Lakh
    
    # YEARS AT CURRENT JOB - Employment stability
    years_at_job = np.random.exponential(3, size=n_samples).astype(int)
    years_at_job = np.clip(years_at_job, 0, 25)
    
    # EXISTING LOANS - Current debt burden
    existing_loans = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.5, 0.3, 0.15, 0.05])
    
    # DEBT-TO-INCOME RATIO (calculated)
    monthly_income = income / 12
    estimated_existing_emi = existing_loans * np.random.uniform(5000, 15000, n_samples)
    dti_ratio = estimated_existing_emi / monthly_income
    
    # Calculate realistic default probability based on features
    default_probability = (
        0.30 * (1 - np.minimum(credit_score / 900, 1)) +  # Lower credit = higher risk
        0.25 * (1 - np.minimum(income / 1000000, 1)) +    # Lower income = higher risk
        0.20 * np.minimum(dti_ratio / 0.5, 1) +           # High debt = higher risk
        0.15 * (years_at_job < 2) +                       # Unstable job = higher risk
        0.10 * np.random.random(size=n_samples)           # Random component
    )
    
    # Apply threshold (should be ~8-10% defaults)
    default_probability = np.clip(default_probability, 0, 1)
    target = (default_probability > np.percentile(default_probability, 91)).astype(int)
    
    # Create simple, clean DataFrame
    data = pd.DataFrame({
        'ID': ids,
        'AGE': age,
        'ANNUAL_INCOME': income,
        'CREDIT_SCORE': credit_score,
        'LOAN_AMOUNT': loan_amount,
        'YEARS_AT_JOB': years_at_job,
        'EXISTING_LOANS': existing_loans,
        'DEBT_TO_INCOME_RATIO': dti_ratio,
        'PROBABILITY_OF_DEFAULT': default_probability,
        'TARGET': target  # 1 = Default, 0 = No Default
    })
    
    return data

def main():
    """Generate and save simple loan data."""
    print("🚀 Generating SIMPLE loan dataset...")
    
    # Generate data
    train_data = generate_simple_loan_data(n_samples=5000)
    test_data = generate_simple_loan_data(n_samples=1000)
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    train_file = 'data/application_train_simple.csv'
    test_file = 'data/application_test_simple.csv'
    
    print(f"📝 Saving training data: {train_file}")
    train_data.to_csv(train_file, index=False)
    
    print(f"📝 Saving test data: {test_file}")
    test_data.to_csv(test_file, index=False)
    
    print("\n✅ Simple dataset generation complete!")
    print(f"\n📊 Training data: {len(train_data)} rows, {len(train_data.columns)} features")
    print(f"📊 Test data: {len(test_data)} rows, {len(test_data.columns)} features")
    print(f"\n📋 Features included:")
    for col in train_data.columns:
        print(f"   - {col}")
    
    print(f"\n🎯 Default rate: {train_data['TARGET'].mean():.2%}")
    
    # Show sample statistics
    print(f"\n📈 Key Statistics:")
    print(f"   - Average Income: ₹{train_data['ANNUAL_INCOME'].mean():,.0f}")
    print(f"   - Average Credit Score: {train_data['CREDIT_SCORE'].mean():.0f}")
    print(f"   - Average Loan Amount: ₹{train_data['LOAN_AMOUNT'].mean():,.0f}")
    print(f"   - Average Age: {train_data['AGE'].mean():.0f} years")
    
    print("\n💡 This simple dataset is perfect for:")
    print("   - Quick demos")
    print("   - Interview explanations")
    print("   - Understanding core concepts")
    print("   - Fast model training")

if __name__ == "__main__":
    main()

