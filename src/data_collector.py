#!/usr/bin/env python3
"""
Data Collector for New User Applications
Handles adding new user data to the database for continuous learning
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import sqlite3
from typing import Dict, Any, Optional

class DataCollector:
    """Collects and stores new user application data for continuous learning."""
    
    def __init__(self, db_path: str = "data/user_applications.db"):
        """Initialize the data collector with database path."""
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Create database and tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create applications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                name TEXT,
                age INTEGER,
                email TEXT,
                phone TEXT,
                annual_income REAL,
                monthly_expenses REAL,
                credit_score INTEGER,
                employment_years REAL,
                loan_amount REAL,
                loan_purpose TEXT,
                preferred_term INTEGER,
                home_ownership TEXT,
                dependents INTEGER,
                existing_loans REAL,
                ai_decision TEXT,
                risk_level TEXT,
                interest_rate REAL,
                confidence_score REAL,
                raw_data TEXT
            )
        ''')
        
        # Create outcomes table (for tracking actual results)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                final_decision TEXT,
                actual_default BOOLEAN,
                payment_status TEXT,
                notes TEXT,
                FOREIGN KEY (application_id) REFERENCES applications (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_new_application(self, user_data: Dict[str, Any], ai_result: Dict[str, Any]) -> int:
        """Add new user application to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Prepare data for insertion
        application_data = {
            'name': user_data.get('name', ''),
            'age': user_data.get('age', 0),
            'email': user_data.get('email', ''),
            'phone': user_data.get('phone', ''),
            'annual_income': user_data.get('annual_income', 0),
            'monthly_expenses': user_data.get('monthly_expenses', 0),
            'credit_score': user_data.get('credit_score', 0),
            'employment_years': user_data.get('employment_years', 0),
            'loan_amount': user_data.get('loan_amount', 0),
            'loan_purpose': user_data.get('loan_purpose', ''),
            'preferred_term': user_data.get('preferred_term', 0),
            'home_ownership': user_data.get('home_ownership', ''),
            'dependents': user_data.get('dependents', 0),
            'existing_loans': user_data.get('existing_loans', 0),
            'ai_decision': ai_result.get('decision', ''),
            'risk_level': ai_result.get('risk_level', ''),
            'interest_rate': ai_result.get('interest_rate', 0),
            'confidence_score': ai_result.get('confidence', 0),
            'raw_data': json.dumps(user_data)
        }
        
        # Insert application
        cursor.execute('''
            INSERT INTO applications (
                name, age, email, phone, annual_income, monthly_expenses,
                credit_score, employment_years, loan_amount, loan_purpose,
                preferred_term, home_ownership, dependents, existing_loans,
                ai_decision, risk_level, interest_rate, confidence_score, raw_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', tuple(application_data.values()))
        
        application_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return application_id
    
    def add_outcome(self, application_id: int, outcome_data: Dict[str, Any]) -> None:
        """Add outcome data for an application (for model improvement)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO outcomes (
                application_id, final_decision, actual_default,
                payment_status, notes
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            application_id,
            outcome_data.get('final_decision', ''),
            outcome_data.get('actual_default', False),
            outcome_data.get('payment_status', ''),
            outcome_data.get('notes', '')
        ))
        
        conn.commit()
        conn.close()
    
    def get_all_applications(self) -> pd.DataFrame:
        """Get all applications as a DataFrame."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM applications", conn)
        conn.close()
        return df
    
    def get_applications_with_outcomes(self) -> pd.DataFrame:
        """Get applications with their outcomes for model training."""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT a.*, o.final_decision, o.actual_default, o.payment_status
            FROM applications a
            LEFT JOIN outcomes o ON a.id = o.application_id
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total applications
        cursor.execute("SELECT COUNT(*) FROM applications")
        total_applications = cursor.fetchone()[0]
        
        # Applications by decision
        cursor.execute("SELECT ai_decision, COUNT(*) FROM applications GROUP BY ai_decision")
        decisions = dict(cursor.fetchall())
        
        # Applications by risk level
        cursor.execute("SELECT risk_level, COUNT(*) FROM applications GROUP BY risk_level")
        risk_levels = dict(cursor.fetchall())
        
        # Average metrics
        cursor.execute("""
            SELECT 
                AVG(annual_income) as avg_income,
                AVG(credit_score) as avg_credit_score,
                AVG(loan_amount) as avg_loan_amount,
                AVG(interest_rate) as avg_interest_rate
            FROM applications
        """)
        averages = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_applications': total_applications,
            'decisions': decisions,
            'risk_levels': risk_levels,
            'averages': {
                'income': averages[0] or 0,
                'credit_score': averages[1] or 0,
                'loan_amount': averages[2] or 0,
                'interest_rate': averages[3] or 0
            }
        }
    
    def export_for_training(self, output_path: str = "data/new_training_data.csv") -> str:
        """Export new data for model retraining."""
        df = self.get_applications_with_outcomes()
        
        # Create features for training
        df['debt_to_income_ratio'] = df['monthly_expenses'] / (df['annual_income'] / 12 + 1)
        df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
        df['age_risk'] = np.where(df['age'] < 25, 1, np.where(df['age'] > 65, 1, 0))
        df['employment_risk'] = np.where(df['employment_years'] < 1, 1, 0)
        
        # Create target variable (if we have outcomes)
        if 'actual_default' in df.columns:
            df['TARGET'] = df['actual_default'].astype(int)
        else:
            # Use AI decision as proxy (for demonstration)
            df['TARGET'] = np.where(df['ai_decision'] == 'REJECT', 1, 0)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        return output_path

class PrivacyManager:
    """Manages data privacy and compliance."""
    
    def __init__(self):
        self.sensitive_fields = ['name', 'email', 'phone']
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize sensitive data for analysis."""
        anonymized = data.copy()
        
        # Hash sensitive fields
        for field in self.sensitive_fields:
            if field in anonymized:
                anonymized[field] = f"ANON_{hash(anonymized[field]) % 10000}"
        
        return anonymized
    
    def check_consent(self, user_data: Dict[str, Any]) -> bool:
        """Check if user has consented to data collection."""
        # In a real system, this would check a consent database
        return user_data.get('data_consent', False)
    
    def get_retention_policy(self) -> Dict[str, Any]:
        """Get data retention policy."""
        return {
            'retention_period_days': 2555,  # 7 years
            'anonymize_after_days': 365,    # 1 year
            'delete_after_days': 2555       # 7 years
        }






