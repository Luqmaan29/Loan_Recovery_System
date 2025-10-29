"""
Generate Synthetic Loan Dataset for Demo
This creates realistic loan application data with all required features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_loan_data(n_samples=10000):
    """
    Generate synthetic loan application data with realistic patterns.
    
    Args:
        n_samples: Number of loan applications to generate
        
    Returns:
        DataFrame with loan application data
    """
    np.random.seed(42)
    
    # Generate realistic IDs
    ids = range(100001, 100001 + n_samples)
    
    # Demographics
    gender = np.random.choice(['M', 'F'], size=n_samples)
    age = np.random.normal(40, 12, n_samples).astype(int)
    age = np.clip(age, 20, 75)
    
    # Income (realistic distribution for Indian context)
    base_income = np.random.lognormal(mean=12.5, sigma=0.8, size=n_samples)
    income = (base_income / 1000).astype(int) * 1000  # Round to nearest 1000
    income = np.clip(income, 150000, 5000000)
    
    # Loan amounts (correlated with income)
    loan_amount = (income * np.random.uniform(1.5, 5, n_samples)).astype(int)
    loan_amount = np.clip(loan_amount, 100000, 10000000)
    
    # Annuity (monthly payment if taking loan)
    annuity = (loan_amount * np.random.uniform(0.01, 0.02, n_samples)).astype(int)
    
    # Credit bureau data
    ext_source_1 = np.random.beta(2, 5, size=n_samples)
    ext_source_2 = np.random.beta(3, 4, size=n_samples)
    ext_source_3 = np.random.beta(2.5, 4.5, size=n_samples)
    
    # Calculate averages
    ext_source_mean = (ext_source_1 + ext_source_2 + ext_source_3) / 3
    
    # Credit amount from credit bureau
    credit_amount = (income * np.random.uniform(0.3, 1.5, n_samples)).astype(int)
    
    # Days employed (some unemployed = 0)
    employed_days = np.random.exponential(1000, size=n_samples).astype(int)
    employed_days = np.where(np.random.random(size=n_samples) < 0.15, 0, employed_days)  # 15% unemployed
    employed_days = np.clip(employed_days, 0, 18000)  # Max ~50 years
    
    # Days registration (age when they registered)
    days_registered = (age * 365) - np.random.exponential(2000, size=n_samples).astype(int)
    days_registered = np.clip(days_registered, -365, age * 365)
    
    # Own car flag
    own_car = np.random.choice(['Y', 'N'], size=n_samples, p=[0.35, 0.65])
    
    # Own realty flag
    own_realty = np.random.choice(['Y', 'N'], size=n_samples, p=[0.45, 0.55])
    
    # Family members
    family_size = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.1, 0.3, 0.25, 0.2, 0.1, 0.05])
    children_count = np.maximum(0, family_size - 2)  # At least 2 adults
    
    # Income type
    income_type = np.random.choice(['Working', 'State servant', 'Commercial associate', 'Pensioner', 'Unemployed'],
                                   size=n_samples, p=[0.5, 0.15, 0.15, 0.1, 0.1])
    
    # Education
    education = np.random.choice(['Higher education', 'Secondary / secondary special', 'Incomplete higher', 
                                  'Lower secondary', 'Academic degree'],
                                 size=n_samples, p=[0.3, 0.3, 0.15, 0.15, 0.1])
    
    # Housing type
    housing = np.random.choice(['Rented apartment', 'House / apartment', 'Municipal apartment', 
                                'With parents', 'Co-op apartment', 'Office apartment'],
                               size=n_samples, p=[0.25, 0.35, 0.15, 0.1, 0.1, 0.05])
    
    # Occupation
    occupation = np.random.choice(['Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 
                                   'High skill tech staff', 'Accountants', 'Medicine staff', 'Cleaning staff',
                                   'Cooking staff', 'Private service staff', 'Security staff', 'Low-skill Laborers',
                                   'Waiters/barmen staff', 'Secretaries', 'HR staff', 'IT staff', 'Realty agents'],
                                  size=n_samples)
    
    # Organization type
    organization_type = np.random.choice(['Business Entity Type 3', 'School', 'Government', 'Religion', 
                                          'Other', 'Electricity', 'Medicine', 'Business Entity Type 2', 
                                          'Self-employed', 'Trade', 'Construction', 'Housing', 
                                          'Kindergarten', 'Business Entity Type 1', 'Military',
                                          'Police', 'Emergency', 'Security Ministries', 'Services', 
                                          'Transport type 3', 'Industry', 'Agriculture', 'Restaurant', 
                                          'Transport type 2', 'University', 'Transport type 1', 'Insurance'],
                                         size=n_samples)
    
    # Flag document submissions
    has_mobile = np.random.choice([0, 1], size=n_samples, p=[0.05, 0.95])
    has_email = np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8])
    has_work_phone = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    has_phone = np.random.choice([0, 1], size=n_samples, p=[0.05, 0.95])
    
    # Contact flags
    flag_mobile = np.where(has_mobile == 1, 1, 0)
    flag_email = np.where(has_email == 1, 1, 0)
    flag_work_phone = np.where(has_work_phone == 1, 1, 0)
    flag_phone = np.where(has_phone == 1, 1, 0)
    
    # Goods price (for goods credit)
    goods_price = np.random.lognormal(mean=12, sigma=0.7, size=n_samples)
    goods_price = (goods_price / 1000).astype(int) * 1000
    
    # Region rating
    region_rating = np.random.choice([1, 2, 3], size=n_samples, p=[0.3, 0.5, 0.2])
    
    # Days birth and employed in absolute terms
    days_birth = -(age * 365 + np.random.randint(0, 365, size=n_samples))
    
    # Calculate target (default flag) with realistic correlations
    # Higher default probability for:
    # - Lower income
    # - Lower external source scores
    # - Higher debt-to-income
    # - Unemployed
    debt_ratio = loan_amount / (income + 1)
    
    default_probability = (
        0.15 * (1 - ext_source_mean) +  # Lower external score = higher risk
        0.25 * (1 - np.minimum(income / 1000000, 1)) +  # Lower income = higher risk
        0.20 * (employed_days < 365) +  # Short employment = higher risk
        0.15 * np.minimum(debt_ratio / 3, 1) +  # High debt = higher risk
        0.05 * np.random.random(size=n_samples)  # Random component
    )
    
    # Apply threshold (target should be ~8% defaults)
    default_probability = np.clip(default_probability, 0, 1)
    target = (default_probability > np.percentile(default_probability, 92)).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'SK_ID_CURR': ids,
        'TARGET': target,
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], size=n_samples),
        'CODE_GENDER': gender,
        'FLAG_OWN_CAR': own_car,
        'FLAG_OWN_REALTY': own_realty,
        'CNT_CHILDREN': children_count,
        'AMT_INCOME_TOTAL': income,
        'AMT_CREDIT': credit_amount,
        'AMT_ANNUITY': annuity,
        'AMT_GOODS_PRICE': goods_price,
        'NAME_TYPE_SUITE': np.random.choice(['Unaccompanied', 'Family', 'Spouse, partner', 
                                             'Children', 'Other_A', 'Other_B', 'Group of people'],
                                            size=n_samples),
        'NAME_INCOME_TYPE': income_type,
        'NAME_EDUCATION_TYPE': education,
        'NAME_FAMILY_STATUS': np.random.choice(['Single / not married', 'Married', 'Civil marriage', 
                                                'Separated', 'Widow'],
                                               size=n_samples),
        'NAME_HOUSING_TYPE': housing,
        'REGION_POPULATION_RELATIVE': np.random.uniform(0.001, 0.05, size=n_samples),
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': employed_days,
        'DAYS_REGISTRATION': days_registered,
        'DAYS_ID_PUBLISH': np.random.randint(-4000, 0, size=n_samples),
        'OWN_CAR_AGE': np.where(own_car == 'Y', 
                                np.random.randint(0, 20, size=n_samples), 0),
        'FLAG_MOBIL': flag_mobile,
        'FLAG_EMP_PHONE': flag_work_phone,
        'FLAG_WORK_PHONE': flag_work_phone,
        'FLAG_CONT_MOBILE': flag_mobile,
        'FLAG_PHONE': flag_phone,
        'FLAG_EMAIL': flag_email,
        'OCCUPATION_TYPE': occupation,
        'CNT_FAM_MEMBERS': family_size,
        'REGION_RATING_CLIENT': region_rating,
        'REGION_RATING_CLIENT_W_CITY': region_rating + np.random.randint(-1, 1, size=n_samples),
        'WEEKDAY_APPR_PROCESS_START': np.random.choice(['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 
                                                        'FRIDAY', 'SATURDAY', 'SUNDAY'],
                                                       size=n_samples),
        'HOUR_APPR_PROCESS_START': np.random.randint(8, 20, size=n_samples),
        'REG_REGION_NOT_LIVE_REGION': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        'REG_REGION_NOT_WORK_REGION': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
        'LIVE_REGION_NOT_WORK_REGION': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        'REG_CITY_NOT_LIVE_CITY': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
        'REG_CITY_NOT_WORK_CITY': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        'LIVE_CITY_NOT_WORK_CITY': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        'ORGANIZATION_TYPE': organization_type,
        'EXT_SOURCE_1': ext_source_1,
        'EXT_SOURCE_2': ext_source_2,
        'EXT_SOURCE_3': ext_source_3,
        'APARTMENTS_AVG': np.random.uniform(0, 1, size=n_samples),
        'BASEMENTAREA_AVG': np.random.uniform(0, 100, size=n_samples),
        'YEARS_BEGINEXPLUATATION_AVG': np.random.uniform(1970, 2020, size=n_samples),
        'YEARS_BUILD_AVG': np.random.uniform(1950, 2020, size=n_samples),
        'COMMONAREA_AVG': np.random.uniform(0, 50, size=n_samples),
        'ELEVATORS_AVG': np.random.uniform(0, 1, size=n_samples),
        'ENTRANCES_AVG': np.random.uniform(0, 2, size=n_samples),
        'FLOORSMAX_AVG': np.random.uniform(1, 30, size=n_samples),
        'FLOORSMIN_AVG': np.random.uniform(0.5, 10, size=n_samples),
        'LANDAREA_AVG': np.random.uniform(0, 1000, size=n_samples),
        'LIVINGAPARTMENTS_AVG': np.random.uniform(0, 50, size=n_samples),
        'LIVINGAREA_AVG': np.random.uniform(0, 200, size=n_samples),
        'NONLIVINGAPARTMENTS_AVG': np.random.uniform(0, 10, size=n_samples),
        'NONLIVINGAREA_AVG': np.random.uniform(0, 100, size=n_samples),
        'APARTMENTS_MODE': np.random.uniform(0, 1, size=n_samples),
        'BASEMENTAREA_MODE': np.random.uniform(0, 100, size=n_samples),
        'YEARS_BEGINEXPLUATATION_MODE': np.random.uniform(1970, 2020, size=n_samples),
        'YEARS_BUILD_MODE': np.random.uniform(1950, 2020, size=n_samples),
        'COMMONAREA_MODE': np.random.uniform(0, 50, size=n_samples),
        'ELEVATORS_MODE': np.random.uniform(0, 1, size=n_samples),
        'ENTRANCES_MODE': np.random.uniform(0, 2, size=n_samples),
        'FLOORSMAX_MODE': np.random.uniform(1, 30, size=n_samples),
        'FLOORSMIN_MODE': np.random.uniform(0.5, 10, size=n_samples),
        'LANDAREA_MODE': np.random.uniform(0, 1000, size=n_samples),
        'LIVINGAPARTMENTS_MODE': np.random.uniform(0, 50, size=n_samples),
        'LIVINGAREA_MODE': np.random.uniform(0, 200, size=n_samples),
        'NONLIVINGAPARTMENTS_MODE': np.random.uniform(0, 10, size=n_samples),
        'NONLIVINGAREA_MODE': np.random.uniform(0, 100, size=n_samples),
        'APARTMENTS_MEDI': np.random.uniform(0, 1, size=n_samples),
        'BASEMENTAREA_MEDI': np.random.uniform(0, 100, size=n_samples),
        'YEARS_BEGINEXPLUATATION_MEDI': np.random.uniform(1970, 2020, size=n_samples),
        'YEARS_BUILD_MEDI': np.random.uniform(1950, 2020, size=n_samples),
        'COMMONAREA_MEDI': np.random.uniform(0, 50, size=n_samples),
        'ELEVATORS_MEDI': np.random.uniform(0, 1, size=n_samples),
        'ENTRANCES_MEDI': np.random.uniform(0, 2, size=n_samples),
        'FLOORSMAX_MEDI': np.random.uniform(1, 30, size=n_samples),
        'FLOORSMIN_MEDI': np.random.uniform(0.5, 10, size=n_samples),
        'LANDAREA_MEDI': np.random.uniform(0, 1000, size=n_samples),
        'LIVINGAPARTMENTS_MEDI': np.random.uniform(0, 50, size=n_samples),
        'LIVINGAREA_MEDI': np.random.uniform(0, 200, size=n_samples),
        'NONLIVINGAPARTMENTS_MEDI': np.random.uniform(0, 10, size=n_samples),
        'NONLIVINGAREA_MEDI': np.random.uniform(0, 100, size=n_samples),
    })
    
    return data

def main():
    """Generate and save synthetic loan data."""
    print("🚀 Generating synthetic loan dataset...")
    
    # Generate data
    train_data = generate_synthetic_loan_data(n_samples=20000)
    test_data = generate_synthetic_loan_data(n_samples=5000)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    train_file = 'data/application_train_synthetic.csv'
    test_file = 'data/application_test_synthetic.csv'
    
    print(f"📝 Saving training data: {train_file}")
    train_data.to_csv(train_file, index=False)
    
    print(f"📝 Saving test data: {test_file}")
    test_data.to_csv(test_file, index=False)
    
    print("\n✅ Dataset generation complete!")
    print(f"\n📊 Training data: {len(train_data)} rows, {len(train_data.columns)} features")
    print(f"📊 Test data: {len(test_data)} rows, {len(test_data.columns)} features")
    print(f"\n🎯 Default rate: {train_data['TARGET'].mean():.2%}")
    print(f"🎯 Test default rate: {test_data['TARGET'].mean():.2%}")
    
    print("\n📁 Files created:")
    print(f"   - data/application_train_synthetic.csv")
    print(f"   - data/application_test_synthetic.csv")
    
    print("\n💡 Note: Update your code to use these synthetic files instead of original data")

if __name__ == "__main__":
    main()

