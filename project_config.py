"""
Project Configuration for Smart Digital Lending Recommendation System
"""

# System Configuration
SYSTEM_NAME = "Smart Digital Lending Recommendation System"
VERSION = "1.0.0"
DESCRIPTION = "AI-powered loan recommendation system with user-friendly interface and risk assessment"

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "title": "🏦 Smart Digital Lending System",
    "icon": "🏦",
    "layout": "wide",
    "port": 8501,
    "host": "localhost"
}

# Data Configuration
DATA_CONFIG = {
    "data_path": "data/",
    "models_path": "models/",
    "reports_path": "reports/",
    "cache_enabled": True
}

# Model Configuration
MODEL_CONFIG = {
    "default_models": ["XGBoost", "LightGBM", "Random Forest"],
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5
}

# Risk Assessment Configuration
RISK_CONFIG = {
    "low_risk_threshold": 0.3,
    "medium_risk_threshold": 0.7,
    "high_risk_threshold": 1.0
}

# Loan Configuration
LOAN_CONFIG = {
    "min_loan_amount": 1000,
    "max_loan_amount": 1000000,
    "min_interest_rate": 3.5,
    "max_interest_rate": 15.0,
    "default_term_years": 30,
    "currency": "$",
    "currency_name": "US Dollar"
}

# UI Configuration
UI_CONFIG = {
    "theme": "light",
    "primary_color": "#667eea",
    "secondary_color": "#764ba2",
    "success_color": "#28a745",
    "warning_color": "#ffc107",
    "danger_color": "#dc3545"
}
