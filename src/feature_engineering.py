"""
Advanced Feature Engineering Module for Smart Digital Lending Recommendation System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Advanced feature engineering for the lending recommendation system.
    Handles feature selection, creation, and transformation.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_selector = None
        self.pca = None
        self.feature_importance = {}
        self.selected_features = []
        
    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk assessment features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with risk features
        """
        df_risk = df.copy()
        
        # Debt-to-income ratio
        if 'AMT_CREDIT' in df_risk.columns and 'AMT_INCOME_TOTAL' in df_risk.columns:
            df_risk['DEBT_TO_INCOME_RATIO'] = df_risk['AMT_CREDIT'] / (df_risk['AMT_INCOME_TOTAL'] + 1)
        
        # Payment burden (annuity to income)
        if 'AMT_ANNUITY' in df_risk.columns and 'AMT_INCOME_TOTAL' in df_risk.columns:
            df_risk['PAYMENT_BURDEN'] = df_risk['AMT_ANNUITY'] / (df_risk['AMT_INCOME_TOTAL'] + 1)
        
        # Credit utilization
        if 'AMT_CREDIT' in df_risk.columns and 'AMT_GOODS_PRICE' in df_risk.columns:
            df_risk['CREDIT_UTILIZATION'] = df_risk['AMT_CREDIT'] / (df_risk['AMT_GOODS_PRICE'] + 1)
        
        # Age risk (very young or very old)
        if 'AGE_YEARS' in df_risk.columns:
            df_risk['AGE_RISK'] = np.where(
                (df_risk['AGE_YEARS'] < 25) | (df_risk['AGE_YEARS'] > 65), 1, 0
            )
        
        # Employment stability
        if 'EMPLOYMENT_YEARS' in df_risk.columns:
            df_risk['EMPLOYMENT_STABILITY'] = np.where(df_risk['EMPLOYMENT_YEARS'] > 5, 1, 0)
            df_risk['UNEMPLOYED'] = np.where(df_risk['EMPLOYMENT_YEARS'] == 0, 1, 0)
        
        # Income stability indicators
        if 'AMT_INCOME_TOTAL' in df_risk.columns:
            income_median = df_risk['AMT_INCOME_TOTAL'].median()
            df_risk['LOW_INCOME'] = np.where(df_risk['AMT_INCOME_TOTAL'] < income_median * 0.5, 1, 0)
            df_risk['HIGH_INCOME'] = np.where(df_risk['AMT_INCOME_TOTAL'] > income_median * 2, 1, 0)
        
        # Credit amount risk
        if 'AMT_CREDIT' in df_risk.columns:
            credit_median = df_risk['AMT_CREDIT'].median()
            df_risk['HIGH_CREDIT_AMOUNT'] = np.where(df_risk['AMT_CREDIT'] > credit_median * 2, 1, 0)
        
        # Family risk indicators
        if 'CNT_CHILDREN' in df_risk.columns:
            df_risk['MANY_CHILDREN'] = np.where(df_risk['CNT_CHILDREN'] > 3, 1, 0)
        
        if 'CNT_FAM_MEMBERS' in df_risk.columns:
            df_risk['LARGE_FAMILY'] = np.where(df_risk['CNT_FAM_MEMBERS'] > 5, 1, 0)
        
        return df_risk
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral pattern features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with behavioral features
        """
        df_behavior = df.copy()
        
        # Application timing features
        if 'HOUR_APPR_PROCESS_START' in df_behavior.columns:
            df_behavior['EARLY_APPLICATION'] = np.where(df_behavior['HOUR_APPR_PROCESS_START'] < 9, 1, 0)
            df_behavior['LATE_APPLICATION'] = np.where(df_behavior['HOUR_APPR_PROCESS_START'] > 17, 1, 0)
        
        # Document submission behavior
        doc_flags = [col for col in df_behavior.columns if col.startswith('FLAG_DOCUMENT_')]
        if doc_flags:
            df_behavior['DOCUMENT_COMPLETENESS'] = df_behavior[doc_flags].sum(axis=1)
            df_behavior['MINIMAL_DOCUMENTATION'] = np.where(df_behavior['DOCUMENT_COMPLETENESS'] < 3, 1, 0)
        
        # Contact information completeness
        contact_flags = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']
        available_contact = [col for col in contact_flags if col in df_behavior.columns]
        if available_contact:
            df_behavior['CONTACT_COMPLETENESS'] = df_behavior[available_contact].sum(axis=1)
            df_behavior['POOR_CONTACT_INFO'] = np.where(df_behavior['CONTACT_COMPLETENESS'] < 2, 1, 0)
        
        # Credit bureau activity
        bureau_cols = [col for col in df_behavior.columns if 'AMT_REQ_CREDIT_BUREAU' in col]
        if bureau_cols:
            df_behavior['BUREAU_ACTIVITY_TOTAL'] = df_behavior[bureau_cols].sum(axis=1)
            df_behavior['HIGH_BUREAU_ACTIVITY'] = np.where(df_behavior['BUREAU_ACTIVITY_TOTAL'] > 10, 1, 0)
        
        return df_behavior
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with interaction features
        """
        df_interaction = df.copy()
        
        # Income and credit interactions
        if 'AMT_INCOME_TOTAL' in df_interaction.columns and 'AMT_CREDIT' in df_interaction.columns:
            df_interaction['INCOME_CREDIT_INTERACTION'] = (
                df_interaction['AMT_INCOME_TOTAL'] * df_interaction['AMT_CREDIT']
            )
        
        # Age and employment interactions
        if 'AGE_YEARS' in df_interaction.columns and 'EMPLOYMENT_YEARS' in df_interaction.columns:
            df_interaction['AGE_EMPLOYMENT_RATIO'] = (
                df_interaction['EMPLOYMENT_YEARS'] / (df_interaction['AGE_YEARS'] + 1)
            )
        
        # External sources interactions
        ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        available_sources = [col for col in ext_sources if col in df_interaction.columns]
        if len(available_sources) >= 2:
            for i, source1 in enumerate(available_sources):
                for source2 in available_sources[i+1:]:
                    df_interaction[f'{source1}_{source2}_INTERACTION'] = (
                        df_interaction[source1] * df_interaction[source2]
                    )
        
        # Income and family size interaction
        if 'AMT_INCOME_TOTAL' in df_interaction.columns and 'CNT_FAM_MEMBERS' in df_interaction.columns:
            df_interaction['INCOME_PER_FAMILY_MEMBER'] = (
                df_interaction['AMT_INCOME_TOTAL'] / (df_interaction['CNT_FAM_MEMBERS'] + 1)
            )
        
        return df_interaction
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal and time-based features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with temporal features
        """
        df_temporal = df.copy()
        
        # Convert days to more interpretable features
        days_columns = [col for col in df_temporal.columns if 'DAYS' in col]
        
        for col in days_columns:
            if col in df_temporal.columns:
                # Convert to years
                years_col = col.replace('DAYS_', 'YEARS_')
                df_temporal[years_col] = -df_temporal[col] / 365.25
                
                # Create recency indicators
                recent_col = col.replace('DAYS_', 'RECENT_')
                df_temporal[recent_col] = np.where(df_temporal[col] > -365, 1, 0)
        
        # Application seasonality
        if 'WEEKDAY_APPR_PROCESS_START' in df_temporal.columns:
            weekday_mapping = {
                'MONDAY': 1, 'TUESDAY': 2, 'WEDNESDAY': 3, 'THURSDAY': 4, 'FRIDAY': 5,
                'SATURDAY': 6, 'SUNDAY': 7
            }
            df_temporal['WEEKDAY_NUMERIC'] = df_temporal['WEEKDAY_APPR_PROCESS_START'].map(weekday_mapping)
            df_temporal['WEEKEND_APPLICATION'] = np.where(
                df_temporal['WEEKDAY_NUMERIC'].isin([6, 7]), 1, 0
            )
        
        return df_temporal
    
    def create_aggregated_features(self, df: pd.DataFrame, group_col: str = 'SK_ID_CURR') -> pd.DataFrame:
        """
        Create aggregated features from external data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            group_col (str): Column to group by
            
        Returns:
            pd.DataFrame: Dataframe with aggregated features
        """
        df_agg = df.copy()
        
        # Bureau-related aggregations
        bureau_cols = [col for col in df_agg.columns if 'BUREAU_' in col]
        if bureau_cols:
            # Count of bureau records
            df_agg['BUREAU_RECORDS_COUNT'] = df_agg[bureau_cols].notna().sum(axis=1)
            
            # Average bureau scores
            numeric_bureau = df_agg[bureau_cols].select_dtypes(include=[np.number])
            if not numeric_bureau.empty:
                df_agg['BUREAU_AVERAGE_SCORE'] = numeric_bureau.mean(axis=1)
                df_agg['BUREAU_MAX_SCORE'] = numeric_bureau.max(axis=1)
                df_agg['BUREAU_MIN_SCORE'] = numeric_bureau.min(axis=1)
        
        # Previous application aggregations
        prev_cols = [col for col in df_agg.columns if 'PREV_' in col]
        if prev_cols:
            df_agg['PREV_APPLICATIONS_COUNT'] = df_agg[prev_cols].notna().sum(axis=1)
            
            numeric_prev = df_agg[prev_cols].select_dtypes(include=[np.number])
            if not numeric_prev.empty:
                df_agg['PREV_AVERAGE_AMOUNT'] = numeric_prev.mean(axis=1)
                df_agg['PREV_MAX_AMOUNT'] = numeric_prev.max(axis=1)
        
        # Installment aggregations
        install_cols = [col for col in df_agg.columns if 'INSTALL_' in col or 'PAYMENT_' in col]
        if install_cols:
            df_agg['INSTALLMENT_RECORDS_COUNT'] = df_agg[install_cols].notna().sum(axis=1)
            
            numeric_install = df_agg[install_cols].select_dtypes(include=[np.number])
            if not numeric_install.empty:
                df_agg['INSTALLMENT_AVERAGE_AMOUNT'] = numeric_install.mean(axis=1)
        
        return df_agg
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', 
                       k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the most important features.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Selection method ('mutual_info', 'f_score', 'random_forest')
            k (int): Number of features to select
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Selected features and feature names
        """
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_score':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'random_forest':
            # Use Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            feature_importance = rf.feature_importances_
            feature_names = X.columns
            top_k_indices = np.argsort(feature_importance)[-k:]
            selected_features = feature_names[top_k_indices]
            return X[selected_features], selected_features.tolist()
        
        # Fit selector
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.feature_selector = selector
        self.selected_features = selected_features
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def apply_pca(self, X: pd.DataFrame, n_components: float = 0.95) -> pd.DataFrame:
        """
        Apply Principal Component Analysis for dimensionality reduction.
        
        Args:
            X (pd.DataFrame): Feature matrix
            n_components (float): Number of components or explained variance ratio
            
        Returns:
            pd.DataFrame: Transformed features
        """
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        
        # Create column names for PCA components
        n_components_actual = X_pca.shape[1]
        pca_columns = [f'PC_{i+1}' for i in range(n_components_actual)]
        
        return pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calculate feature importance using multiple methods.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        importance_scores = {}
        
        # Mutual information
        mi_scores = mutual_info_classif(X, y)
        for i, feature in enumerate(X.columns):
            importance_scores[f'{feature}_mutual_info'] = mi_scores[i]
        
        # F-score
        f_scores, _ = f_classif(X, y)
        for i, feature in enumerate(X.columns):
            importance_scores[f'{feature}_f_score'] = f_scores[i]
        
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        for i, feature in enumerate(X.columns):
            importance_scores[f'{feature}_rf_importance'] = rf.feature_importances_[i]
        
        self.feature_importance = importance_scores
        return importance_scores
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for key ratios to capture non-linear relationships.
        
        Args:
            df (pd.DataFrame): Input dataframe
            degree (int): Degree of polynomial features
            
        Returns:
            pd.DataFrame: Dataframe with polynomial features
        """
        df_poly = df.copy()
        
        # Key ratios for polynomial features
        key_ratios = [
            'DEBT_TO_INCOME_RATIO',
            'PAYMENT_BURDEN', 
            'CREDIT_UTILIZATION',
            'INCOME_CREDIT_RATIO',
            'ANNUITY_INCOME_RATIO'
        ]
        
        available_ratios = [col for col in key_ratios if col in df_poly.columns]
        
        for ratio in available_ratios:
            # Create polynomial features
            for deg in range(2, degree + 1):
                poly_col = f'{ratio}_POLY_{deg}'
                df_poly[poly_col] = df_poly[ratio] ** deg
            
            # Create log transformation (handle zeros)
            log_col = f'{ratio}_LOG'
            df_poly[log_col] = np.log1p(np.abs(df_poly[ratio]))
            
            # Create square root transformation
            sqrt_col = f'{ratio}_SQRT'
            df_poly[sqrt_col] = np.sqrt(np.abs(df_poly[ratio]))
        
        return df_poly
    
    def create_target_encoding(self, df: pd.DataFrame, target_col: str = 'TARGET') -> pd.DataFrame:
        """
        Create target encoding for categorical variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            
        Returns:
            pd.DataFrame: Dataframe with target encoded features
        """
        df_encoded = df.copy()
        
        # Skip target encoding if target column is not present or has no variance
        if target_col not in df_encoded.columns or df_encoded[target_col].nunique() <= 1:
            return df_encoded
        
        # Categorical columns for target encoding (both object and low-cardinality numeric)
        categorical_cols = []
        for col in df_encoded.columns:
            if col != target_col and col != 'SK_ID_CURR':
                if df_encoded[col].dtype == 'object' or df_encoded[col].nunique() < 20:
                    categorical_cols.append(col)
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                try:
                    # Calculate target mean for each category
                    target_mean = df_encoded.groupby(col)[target_col].mean()
                    
                    # Create target encoded feature
                    encoded_col = f'{col}_TARGET_ENC'
                    df_encoded[encoded_col] = df_encoded[col].map(target_mean)
                    
                    # Fill missing values with overall mean
                    df_encoded[encoded_col].fillna(df_encoded[target_col].mean(), inplace=True)
                except Exception as e:
                    print(f"Warning: Could not create target encoding for {col}: {str(e)}")
                    continue
        
        return df_encoded
    
    def create_clustering_features(self, df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """
        Create clustering-based features using K-means.
        
        Args:
            df (pd.DataFrame): Input dataframe
            n_clusters (int): Number of clusters
            
        Returns:
            pd.DataFrame: Dataframe with clustering features
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        df_cluster = df.copy()
        
        # Select numerical features for clustering
        numerical_cols = df_cluster.select_dtypes(include=[np.number]).columns
        clustering_features = [col for col in numerical_cols 
                             if col not in ['TARGET', 'SK_ID_CURR'] and not col.startswith('CLUSTER_')]
        
        if len(clustering_features) > 0:
            # Prepare data for clustering
            X_cluster = df_cluster[clustering_features].fillna(0)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster features
            df_cluster['CUSTOMER_CLUSTER'] = cluster_labels
            
            # Distance to cluster centers
            distances = kmeans.transform(X_scaled)
            for i in range(n_clusters):
                df_cluster[f'DISTANCE_TO_CLUSTER_{i}'] = distances[:, i]
            
            # Cluster center features
            for i, center in enumerate(kmeans.cluster_centers_):
                for j, feature in enumerate(clustering_features[:len(center)]):
                    df_cluster[f'CLUSTER_{i}_{feature}_CENTER'] = center[j]
        
        return df_cluster
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features for better model performance.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with statistical features
        """
        df_stats = df.copy()
        
        # External source statistics
        ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        available_sources = [col for col in ext_sources if col in df_stats.columns]
        
        if len(available_sources) >= 2:
            ext_data = df_stats[available_sources]
            
            # Statistical measures
            df_stats['EXT_SOURCES_MEAN'] = ext_data.mean(axis=1)
            df_stats['EXT_SOURCES_STD'] = ext_data.std(axis=1)
            df_stats['EXT_SOURCES_MIN'] = ext_data.min(axis=1)
            df_stats['EXT_SOURCES_MAX'] = ext_data.max(axis=1)
            df_stats['EXT_SOURCES_RANGE'] = ext_data.max(axis=1) - ext_data.min(axis=1)
            df_stats['EXT_SOURCES_MEDIAN'] = ext_data.median(axis=1)
            df_stats['EXT_SOURCES_SKEW'] = ext_data.skew(axis=1)
            df_stats['EXT_SOURCES_KURTOSIS'] = ext_data.kurtosis(axis=1)
        
        # Income-based statistical features
        income_cols = [col for col in df_stats.columns if 'INCOME' in col or 'AMT_INCOME' in col]
        if len(income_cols) > 1:
            income_data = df_stats[income_cols]
            df_stats['INCOME_STABILITY'] = income_data.std(axis=1)
            df_stats['INCOME_CONSISTENCY'] = 1 / (income_data.std(axis=1) + 1)
        
        # Credit amount statistics
        credit_cols = [col for col in df_stats.columns if 'AMT_CREDIT' in col or 'CREDIT' in col]
        if len(credit_cols) > 1:
            credit_data = df_stats[credit_cols]
            df_stats['CREDIT_VARIABILITY'] = credit_data.std(axis=1)
            df_stats['CREDIT_CONSISTENCY'] = 1 / (credit_data.std(axis=1) + 1)
        
        return df_stats
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps including advanced techniques.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with all engineered features
        """
        print("Creating risk features...")
        df_features = self.create_risk_features(df)
        
        print("Creating behavioral features...")
        df_features = self.create_behavioral_features(df_features)
        
        print("Creating interaction features...")
        df_features = self.create_interaction_features(df_features)
        
        print("Creating temporal features...")
        df_features = self.create_temporal_features(df_features)
        
        print("Creating aggregated features...")
        df_features = self.create_aggregated_features(df_features)
        
        print("Creating polynomial features...")
        df_features = self.create_polynomial_features(df_features)
        
        print("Creating target encoding...")
        df_features = self.create_target_encoding(df_features)
        
        print("Creating clustering features...")
        df_features = self.create_clustering_features(df_features)
        
        print("Creating statistical features...")
        df_features = self.create_statistical_features(df_features)
        
        print(f"Advanced feature engineering complete. Shape: {df_features.shape}")
        return df_features
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary of feature engineering.
        
        Returns:
            Dict: Feature engineering summary
        """
        return {
            'selected_features': len(self.selected_features),
            'feature_importance_methods': len(self.feature_importance) if self.feature_importance else 0,
            'pca_components': self.pca.n_components_ if self.pca else None,
            'explained_variance_ratio': self.pca.explained_variance_ratio_.sum() if self.pca else None
        }


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    
    # Load and preprocess data
    loader = DataLoader()
    data = loader.load_all_data()
    
    preprocessor = DataPreprocessor()
    processed_df, features = preprocessor.preprocess_pipeline(data['application'], data)
    
    # Apply feature engineering
    engineer = FeatureEngineer()
    engineered_df = engineer.create_all_features(processed_df)
    
    print(f"Engineered features shape: {engineered_df.shape}")
    print(f"New features created: {engineered_df.shape[1] - processed_df.shape[1]}")


