"""
Data Preprocessing Module for Smart Digital Lending Recommendation System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for the lending recommendation system.
    Handles missing values, encoding, scaling, and feature engineering.
    """
    
    def __init__(self):
        """Initialize the preprocessor with default settings."""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        
    def identify_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify different types of columns in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, List[str]]: Dictionary with column types
        """
        column_types = {
            'categorical': [],
            'numerical': [],
            'binary': [],
            'datetime': [],
            'id_columns': []
        }
        
        for col in df.columns:
            if col in ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU']:
                column_types['id_columns'].append(col)
            elif df[col].dtype == 'object':
                if df[col].nunique() <= 2:
                    column_types['binary'].append(col)
                else:
                    column_types['categorical'].append(col)
            elif df[col].dtype in ['int64', 'float64']:
                if df[col].nunique() <= 2:
                    column_types['binary'].append(col)
                else:
                    column_types['numerical'].append(col)
            elif 'DAYS' in col.upper():
                column_types['datetime'].append(col)
                
        return column_types
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Strategy for handling missing values
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        df_processed = df.copy()
        
        # Identify column types
        column_types = self.identify_column_types(df_processed)
        
        # Handle different column types
        for col_type, columns in column_types.items():
            if col_type == 'id_columns':
                continue
                
            for col in columns:
                if df_processed[col].isnull().sum() > 0:
                    if col_type == 'categorical':
                        # For categorical, use mode or 'Unknown'
                        mode_value = df_processed[col].mode()
                        if len(mode_value) > 0:
                            df_processed[col].fillna(mode_value[0], inplace=True)
                        else:
                            df_processed[col].fillna('Unknown', inplace=True)
                            
                    elif col_type == 'binary':
                        # For binary, use mode
                        mode_value = df_processed[col].mode()
                        if len(mode_value) > 0:
                            df_processed[col].fillna(mode_value[0], inplace=True)
                        else:
                            df_processed[col].fillna(0, inplace=True)
                            
                    elif col_type == 'numerical':
                        if strategy == 'auto':
                            # Use median for numerical columns
                            df_processed[col].fillna(df_processed[col].median(), inplace=True)
                        elif strategy == 'mean':
                            df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                        elif strategy == 'median':
                            df_processed[col].fillna(df_processed[col].median(), inplace=True)
                        elif strategy == 'knn':
                            # Use KNN imputation for numerical columns
                            if col not in self.imputers:
                                self.imputers[col] = KNNImputer(n_neighbors=5)
                            df_processed[col] = self.imputers[col].fit_transform(df_processed[[col]])
                            
                    elif col_type == 'datetime':
                        # For days columns, use median
                        df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        return df_processed
    
    def encode_categorical_variables(self, df: pd.DataFrame, method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Encoding method ('label', 'onehot', 'target')
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        df_encoded = df.copy()
        column_types = self.identify_column_types(df_encoded)
        
        # Also check for any remaining object columns
        object_columns = df_encoded.select_dtypes(include=['object']).columns
        categorical_columns = list(set(column_types['categorical'] + list(object_columns)))
        
        for col in categorical_columns:
            if col in df_encoded.columns and col not in ['TARGET', 'SK_ID_CURR']:
                if method == 'label':
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                        df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                    else:
                        # Handle unseen categories
                        unique_values = df_encoded[col].unique()
                        known_values = self.encoders[col].classes_
                        for val in unique_values:
                            if val not in known_values:
                                df_encoded.loc[df_encoded[col] == val, col] = 'Unknown'
                        df_encoded[col] = self.encoders[col].transform(df_encoded[col].astype(str))
                        
                elif method == 'onehot':
                    # One-hot encoding
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded.drop(col, axis=1, inplace=True)
                
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Scaling method ('standard', 'robust', 'minmax')
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        df_scaled = df.copy()
        column_types = self.identify_column_types(df_scaled)
        
        for col in column_types['numerical']:
            if col not in ['TARGET']:  # Don't scale target variable
                if method == 'standard':
                    if col not in self.scalers:
                        self.scalers[col] = StandardScaler()
                    df_scaled[col] = self.scalers[col].fit_transform(df_scaled[[col]])
                elif method == 'robust':
                    if col not in self.scalers:
                        self.scalers[col] = RobustScaler()
                    df_scaled[col] = self.scalers[col].fit_transform(df_scaled[[col]])
                    
        return df_scaled
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for better model performance.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with derived features
        """
        df_features = df.copy()
        
        # Income to credit ratio
        if 'AMT_INCOME_TOTAL' in df_features.columns and 'AMT_CREDIT' in df_features.columns:
            df_features['INCOME_CREDIT_RATIO'] = df_features['AMT_INCOME_TOTAL'] / (df_features['AMT_CREDIT'] + 1)
        
        # Annuity to income ratio
        if 'AMT_ANNUITY' in df_features.columns and 'AMT_INCOME_TOTAL' in df_features.columns:
            df_features['ANNUITY_INCOME_RATIO'] = df_features['AMT_ANNUITY'] / (df_features['AMT_INCOME_TOTAL'] + 1)
        
        # Credit to goods price ratio
        if 'AMT_CREDIT' in df_features.columns and 'AMT_GOODS_PRICE' in df_features.columns:
            df_features['CREDIT_GOODS_RATIO'] = df_features['AMT_CREDIT'] / (df_features['AMT_GOODS_PRICE'] + 1)
        
        # Age in years
        if 'DAYS_BIRTH' in df_features.columns:
            df_features['AGE_YEARS'] = -df_features['DAYS_BIRTH'] / 365.25
        
        # Employment years
        if 'DAYS_EMPLOYED' in df_features.columns:
            df_features['EMPLOYMENT_YEARS'] = -df_features['DAYS_EMPLOYED'] / 365.25
            # Handle unemployed (365243 days)
            df_features['EMPLOYMENT_YEARS'] = df_features['EMPLOYMENT_YEARS'].replace(365243/365.25, 0)
        
        # External source average
        ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        available_sources = [col for col in ext_sources if col in df_features.columns]
        if available_sources:
            df_features['EXT_SOURCE_AVG'] = df_features[available_sources].mean(axis=1)
            df_features['EXT_SOURCE_MAX'] = df_features[available_sources].max(axis=1)
            df_features['EXT_SOURCE_MIN'] = df_features[available_sources].min(axis=1)
            df_features['EXT_SOURCE_STD'] = df_features[available_sources].std(axis=1)
        
        # Document flags count
        doc_flags = [col for col in df_features.columns if col.startswith('FLAG_DOCUMENT_')]
        if doc_flags:
            df_features['DOCUMENT_COUNT'] = df_features[doc_flags].sum(axis=1)
        
        # Credit bureau requests count
        bureau_cols = [col for col in df_features.columns if 'AMT_REQ_CREDIT_BUREAU' in col]
        if bureau_cols:
            df_features['BUREAU_REQUESTS_TOTAL'] = df_features[bureau_cols].sum(axis=1)
        
        # Social circle default rates
        if 'OBS_30_CNT_SOCIAL_CIRCLE' in df_features.columns and 'DEF_30_CNT_SOCIAL_CIRCLE' in df_features.columns:
            df_features['SOCIAL_CIRCLE_DEFAULT_RATE_30'] = (
                df_features['DEF_30_CNT_SOCIAL_CIRCLE'] / (df_features['OBS_30_CNT_SOCIAL_CIRCLE'] + 1)
            )
        
        if 'OBS_60_CNT_SOCIAL_CIRCLE' in df_features.columns and 'DEF_60_CNT_SOCIAL_CIRCLE' in df_features.columns:
            df_features['SOCIAL_CIRCLE_DEFAULT_RATE_60'] = (
                df_features['DEF_60_CNT_SOCIAL_CIRCLE'] / (df_features['OBS_60_CNT_SOCIAL_CIRCLE'] + 1)
            )
        
        return df_features
    
    def aggregate_external_data(self, main_df: pd.DataFrame, external_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate external datasets to create additional features.
        
        Args:
            main_df (pd.DataFrame): Main application dataframe
            external_dfs (Dict[str, pd.DataFrame]): Dictionary of external dataframes
            
        Returns:
            pd.DataFrame: Main dataframe with aggregated features
        """
        df_aggregated = main_df.copy()
        
        # Aggregate bureau data
        if 'bureau' in external_dfs:
            bureau_df = external_dfs['bureau']
            
            # Bureau aggregations
            bureau_agg = bureau_df.groupby('SK_ID_CURR').agg({
                'DAYS_CREDIT': ['min', 'max', 'mean'],
                'CREDIT_DAY_OVERDUE': ['max', 'mean'],
                'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
                'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max'],
                'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean', 'max'],
                'CNT_CREDIT_PROLONG': ['sum', 'mean'],
                'AMT_ANNUITY': ['sum', 'mean', 'max']
            }).reset_index()
            
            # Flatten column names
            bureau_agg.columns = ['SK_ID_CURR'] + [f'BUREAU_{col[0]}_{col[1]}' for col in bureau_agg.columns[1:]]
            
            # Merge with main dataframe
            df_aggregated = df_aggregated.merge(bureau_agg, on='SK_ID_CURR', how='left')
        
        # Aggregate previous applications
        if 'previous_application' in external_dfs:
            prev_df = external_dfs['previous_application']
            
            # Previous application aggregations
            prev_agg = prev_df.groupby('SK_ID_CURR').agg({
                'AMT_ANNUITY': ['sum', 'mean', 'max'],
                'AMT_APPLICATION': ['sum', 'mean', 'max'],
                'AMT_CREDIT': ['sum', 'mean', 'max'],
                'DAYS_DECISION': ['min', 'max', 'mean'],
                'CNT_PAYMENT': ['sum', 'mean', 'max']
            }).reset_index()
            
            # Flatten column names
            prev_agg.columns = ['SK_ID_CURR'] + [f'PREV_{col[0]}_{col[1]}' for col in prev_agg.columns[1:]]
            
            # Merge with main dataframe
            df_aggregated = df_aggregated.merge(prev_agg, on='SK_ID_CURR', how='left')
        
        # Aggregate installments data
        if 'installments_payments' in external_dfs:
            install_df = external_dfs['installments_payments']
            
            # Installment aggregations
            install_agg = install_df.groupby('SK_ID_CURR').agg({
                'AMT_INSTALMENT': ['sum', 'mean', 'max'],
                'AMT_PAYMENT': ['sum', 'mean', 'max'],
                'DAYS_INSTALMENT': ['min', 'max', 'mean'],
                'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean']
            }).reset_index()
            
            # Flatten column names
            install_agg.columns = ['SK_ID_CURR'] + [f'INSTALL_{col[0]}_{col[1]}' for col in install_agg.columns[1:]]
            
            # Calculate payment behavior
            install_df['PAYMENT_DIFF'] = install_df['AMT_PAYMENT'] - install_df['AMT_INSTALMENT']
            install_df['PAYMENT_RATIO'] = install_df['AMT_PAYMENT'] / (install_df['AMT_INSTALMENT'] + 1)
            
            payment_agg = install_df.groupby('SK_ID_CURR').agg({
                'PAYMENT_DIFF': ['sum', 'mean', 'min', 'max'],
                'PAYMENT_RATIO': ['mean', 'min', 'max']
            }).reset_index()
            
            payment_agg.columns = ['SK_ID_CURR'] + [f'PAYMENT_{col[0]}_{col[1]}' for col in payment_agg.columns[1:]]
            
            # Merge with main dataframe
            df_aggregated = df_aggregated.merge(install_agg, on='SK_ID_CURR', how='left')
            df_aggregated = df_aggregated.merge(payment_agg, on='SK_ID_CURR', how='left')
        
        return df_aggregated
    
    def preprocess_pipeline(self, df: pd.DataFrame, external_dfs: Optional[Dict] = None, 
                          target_col: str = 'TARGET') -> Tuple[pd.DataFrame, List[str]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            external_dfs (Optional[Dict]): External datasets for aggregation
            target_col (str): Target column name
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Processed dataframe and feature columns
        """
        print("Starting preprocessing pipeline...")
        
        # Step 1: Handle missing values
        print("Handling missing values...")
        df_processed = self.handle_missing_values(df)
        
        # Step 2: Create derived features
        print("Creating derived features...")
        df_processed = self.create_derived_features(df_processed)
        
        # Step 3: Aggregate external data
        if external_dfs:
            print("Aggregating external data...")
            df_processed = self.aggregate_external_data(df_processed, external_dfs)
        
        # Step 4: Encode categorical variables
        print("Encoding categorical variables...")
        df_processed = self.encode_categorical_variables(df_processed)
        
        # Step 5: Scale features
        print("Scaling features...")
        df_processed = self.scale_features(df_processed)
        
        # Step 6: Identify feature columns
        feature_columns = [col for col in df_processed.columns 
                          if col not in ['SK_ID_CURR', 'TARGET', 'SK_ID_PREV', 'SK_ID_BUREAU']]
        
        self.feature_columns = feature_columns
        self.categorical_columns = [col for col in feature_columns 
                                  if df_processed[col].dtype == 'object']
        self.numerical_columns = [col for col in feature_columns 
                                if df_processed[col].dtype in ['int64', 'float64']]
        
        print(f"Preprocessing complete. Features: {len(feature_columns)}")
        print(f"Categorical features: {len(self.categorical_columns)}")
        print(f"Numerical features: {len(self.numerical_columns)}")
        
        return df_processed, feature_columns
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get summary of preprocessing steps.
        
        Returns:
            Dict: Summary of preprocessing
        """
        return {
            'feature_columns': len(self.feature_columns),
            'categorical_columns': len(self.categorical_columns),
            'numerical_columns': len(self.numerical_columns),
            'scalers_fitted': len(self.scalers),
            'encoders_fitted': len(self.encoders),
            'imputers_fitted': len(self.imputers)
        }


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    data = loader.load_all_data()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess main application data
    app_df = data['application']
    processed_df, features = preprocessor.preprocess_pipeline(app_df, data)
    
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Number of features: {len(features)}")
    print(f"Missing values: {processed_df.isnull().sum().sum()}")


