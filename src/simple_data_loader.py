"""
Simple Data Loader for Clean Loan Dataset
"""

import pandas as pd
import numpy as np
import os

class SimpleDataLoader:
    """Load simple loan dataset."""
    
    def __init__(self, data_path="data/"):
        self.data_path = data_path
    
    def load_data(self):
        """Load train and test data."""
        train_path = os.path.join(self.data_path, "application_train_simple.csv")
        test_path = os.path.join(self.data_path, "application_test_simple.csv")
        
        print("📊 Loading loan data...")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"✅ Training: {train_df.shape[0]} rows, {train_df.shape[1]} features")
        print(f"✅ Test: {test_df.shape[0]} rows, {test_df.shape[1]} features")
        print(f"✅ Default rate: {train_df['TARGET'].mean():.2%}")
        
        return train_df, test_df
    
    def prepare_features(self, df):
        """Prepare features for training."""
        # Select features (exclude ID, TARGET, and PROBABILITY_OF_DEFAULT)
        feature_cols = [
            'AGE', 'ANNUAL_INCOME', 'CREDIT_SCORE', 
            'LOAN_AMOUNT', 'YEARS_AT_JOB', 'EXISTING_LOANS', 
            'DEBT_TO_INCOME_RATIO'
        ]
        
        X = df[feature_cols].copy()
        y = df['TARGET'].copy()
        
        return X, y
    
    def load_all_data(self):
        """Load and prepare all data."""
        train_df, test_df = self.load_data()
        
        X_train, y_train = self.prepare_features(train_df)
        X_test, y_test = self.prepare_features(test_df)
        
        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
            'raw_train': train_df,
            'raw_test': test_df
        }

