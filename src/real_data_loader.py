"""
Real Data Loader for Home Credit Default Risk Dataset
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RealDataLoader:
    """
    Data loader specifically for the Home Credit Default Risk dataset.
    """
    
    def __init__(self, data_path: str = "data/"):
        """Initialize with data path."""
        self.data_path = data_path
        self.data = {}
        
    def load_application_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load application train and test data."""
        train_path = os.path.join(self.data_path, "application_train.csv")
        test_path = os.path.join(self.data_path, "application_test.csv")
        
        print("📊 Loading application data...")
        
        # Load training data
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            print(f"  ✅ Training data: {train_df.shape}")
        else:
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        # Load test data
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            print(f"  ✅ Test data: {test_df.shape}")
        else:
            raise FileNotFoundError(f"Test data not found at {test_path}")
        
        return train_df, test_df
    
    def load_bureau_data(self) -> pd.DataFrame:
        """Load bureau data."""
        bureau_path = os.path.join(self.data_path, "bureau.csv")
        
        if os.path.exists(bureau_path):
            print("📊 Loading bureau data...")
            bureau_df = pd.read_csv(bureau_path)
            print(f"  ✅ Bureau data: {bureau_df.shape}")
            return bureau_df
        else:
            print("⚠️ Bureau data not found")
            return pd.DataFrame()
    
    def load_previous_applications(self) -> pd.DataFrame:
        """Load previous applications data."""
        prev_path = os.path.join(self.data_path, "previous_application.csv")
        
        if os.path.exists(prev_path):
            print("📊 Loading previous applications...")
            prev_df = pd.read_csv(prev_path)
            print(f"  ✅ Previous applications: {prev_df.shape}")
            return prev_df
        else:
            print("⚠️ Previous applications data not found")
            return pd.DataFrame()
    
    def load_installments_data(self) -> pd.DataFrame:
        """Load installments payments data."""
        install_path = os.path.join(self.data_path, "installments_payments.csv")
        
        if os.path.exists(install_path):
            print("📊 Loading installments data...")
            install_df = pd.read_csv(install_path)
            print(f"  ✅ Installments data: {install_df.shape}")
            return install_df
        else:
            print("⚠️ Installments data not found")
            return pd.DataFrame()
    
    def load_credit_card_data(self) -> pd.DataFrame:
        """Load credit card balance data."""
        cc_path = os.path.join(self.data_path, "credit_card_balance.csv")
        
        if os.path.exists(cc_path):
            print("📊 Loading credit card data...")
            cc_df = pd.read_csv(cc_path)
            print(f"  ✅ Credit card data: {cc_df.shape}")
            return cc_df
        else:
            print("⚠️ Credit card data not found")
            return pd.DataFrame()
    
    def load_pos_cash_data(self) -> pd.DataFrame:
        """Load POS cash balance data."""
        pos_path = os.path.join(self.data_path, "POS_CASH_balance.csv")
        
        if os.path.exists(pos_path):
            print("📊 Loading POS cash data...")
            pos_df = pd.read_csv(pos_path)
            print(f"  ✅ POS cash data: {pos_df.shape}")
            return pos_df
        else:
            print("⚠️ POS cash data not found")
            return pd.DataFrame()
    
    def load_bureau_balance_data(self) -> pd.DataFrame:
        """Load bureau balance data."""
        bureau_balance_path = os.path.join(self.data_path, "bureau_balance.csv")
        
        if os.path.exists(bureau_balance_path):
            print("📊 Loading bureau balance data...")
            bureau_balance_df = pd.read_csv(bureau_balance_path)
            print(f"  ✅ Bureau balance data: {bureau_balance_df.shape}")
            return bureau_balance_df
        else:
            print("⚠️ Bureau balance data not found")
            return pd.DataFrame()
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available datasets."""
        print("🏦 Loading Home Credit Default Risk Dataset")
        print("=" * 50)
        
        # Load all datasets
        self.data['application_train'], self.data['application_test'] = self.load_application_data()
        self.data['bureau'] = self.load_bureau_data()
        self.data['previous_application'] = self.load_previous_applications()
        self.data['installments_payments'] = self.load_installments_data()
        self.data['credit_card_balance'] = self.load_credit_card_data()
        self.data['POS_CASH_balance'] = self.load_pos_cash_data()
        self.data['bureau_balance'] = self.load_bureau_balance_data()
        
        # Remove empty dataframes
        self.data = {k: v for k, v in self.data.items() if not v.empty}
        
        print(f"\n✅ Loaded {len(self.data)} datasets successfully!")
        return self.data
    
    def get_data_summary(self) -> Dict:
        """Get summary of loaded data."""
        summary = {}
        
        for name, df in self.data.items():
            summary[name] = {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                'dtypes': df.dtypes.value_counts().to_dict()
            }
        
        return summary
    
    def create_sample_for_demo(self, sample_size: int = 10000) -> Dict[str, pd.DataFrame]:
        """Create a sample dataset for demonstration purposes."""
        print(f"🎯 Creating sample dataset ({sample_size:,} records) for demo...")
        
        sample_data = {}
        
        # Sample application data
        if 'application_train' in self.data:
            app_sample = self.data['application_train'].sample(n=min(sample_size, len(self.data['application_train'])), random_state=42)
            sample_data['application'] = app_sample
            print(f"  ✅ Application sample: {app_sample.shape}")
        
        # Sample other datasets proportionally
        for name, df in self.data.items():
            if name != 'application_train' and not df.empty:
                # Sample proportionally to maintain relationships
                sample_ratio = min(1.0, sample_size / len(self.data['application_train']))
                sample_n = max(1000, int(len(df) * sample_ratio))
                sample_n = min(sample_n, len(df))
                
                sample_df = df.sample(n=sample_n, random_state=42)
                sample_data[name] = sample_df
                print(f"  ✅ {name} sample: {sample_df.shape}")
        
        return sample_data


if __name__ == "__main__":
    # Example usage
    loader = RealDataLoader()
    
    # Load all data
    data = loader.load_all_data()
    
    # Get summary
    summary = loader.get_data_summary()
    print("\n📊 Data Summary:")
    for name, info in summary.items():
        print(f"  {name}: {info['shape']} - Memory: {info['memory_usage_mb']:.1f}MB - Missing: {info['missing_percentage']:.1f}%")
    
    # Create sample for demo
    sample_data = loader.create_sample_for_demo(5000)
    print(f"\n🎯 Sample data created: {len(sample_data)} datasets")


