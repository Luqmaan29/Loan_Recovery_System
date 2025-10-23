"""
Quick Performance Test for Smart Digital Lending Recommendation System
Tests the improved model performance with proper data cleaning
"""

import pandas as pd
import numpy as np
import sys
import os
import time
from typing import Dict, List, Tuple

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from real_data_loader import RealDataLoader
from data_preprocessor import DataPreprocessor
from feature_engineering import FeatureEngineer

def test_performance_improvements():
    """Test the performance improvements with advanced techniques."""
    
    print("🚀 Testing Performance Improvements")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    loader = RealDataLoader()
    data = loader.load_all_data()
    
    # Create sample for testing
    sample_data = loader.create_sample_for_demo(3000)  # Smaller sample for faster testing
    app_df = sample_data['application']
    print(f"Sample data shape: {app_df.shape}")
    
    # Preprocessing
    print("\n🔧 Preprocessing data...")
    preprocessor = DataPreprocessor()
    processed_df, features = preprocessor.preprocess_pipeline(app_df)
    print(f"Processed data shape: {processed_df.shape}")
    
    # Feature Engineering
    print("\n⚙️ Advanced feature engineering...")
    engineer = FeatureEngineer()
    engineered_df = engineer.create_all_features(processed_df)
    print(f"Engineered data shape: {engineered_df.shape}")
    
    # Clean data - ensure all columns are numeric and no NaN values
    print("\n🧹 Cleaning data for ML...")
    # Remove any remaining non-numeric columns except target and ID
    numeric_cols = engineered_df.select_dtypes(include=[np.number]).columns
    keep_cols = ['TARGET', 'SK_ID_CURR'] + [col for col in numeric_cols if col not in ['TARGET', 'SK_ID_CURR']]
    engineered_df = engineered_df[keep_cols]
    
    # Fill any remaining NaN values
    engineered_df = engineered_df.fillna(0)
    
    # Remove infinite values
    engineered_df = engineered_df.replace([np.inf, -np.inf], 0)
    
    # Prepare data for training
    X = engineered_df.drop(['TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')
    y = engineered_df['TARGET'].dropna()
    X = X.loc[y.index]
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts()}")
    print(f"Data types: {X.dtypes.value_counts()}")
    print(f"NaN values: {X.isnull().sum().sum()}")
    print(f"Infinite values: {np.isinf(X).sum().sum()}")
    
    # Import required modules
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Test 1: Original Model Performance
    print("\n" + "="*50)
    print("📈 TEST 1: ORIGINAL MODEL PERFORMANCE")
    print("="*50)
    
    # Train original models
    original_models = {
        'XGBoost Original': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100),
        'LightGBM Original': lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=100),
        'Random Forest Original': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    original_results = {}
    
    for name, model in original_models.items():
        print(f"Training {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            original_results[name] = {
                'model': model,
                'auc': auc_score,
                'f1': f1,
                'accuracy': accuracy
            }
            
            print(f"{name} - AUC: {auc_score:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    # Get best original model
    if original_results:
        best_original_name = max(original_results.keys(), key=lambda k: original_results[k]['auc'])
        best_original_auc = original_results[best_original_name]['auc']
        print(f"\nOriginal Best Model: {best_original_name}")
        print(f"Original Best AUC: {best_original_auc:.4f}")
    else:
        print("No original models trained successfully")
        return
    
    # Test 2: Advanced Model Performance with Better Parameters
    print("\n" + "="*50)
    print("🚀 TEST 2: ADVANCED MODEL PERFORMANCE")
    print("="*50)
    
    # Train models with better parameters
    advanced_models = {
        'XGBoost Advanced': xgb.XGBClassifier(
            random_state=42, eval_metric='logloss', 
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=1
        ),
        'LightGBM Advanced': lgb.LGBMClassifier(
            random_state=42, verbose=-1, 
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=1
        ),
        'Random Forest Advanced': RandomForestClassifier(
            random_state=42, n_estimators=300, max_depth=15, 
            min_samples_split=5, min_samples_leaf=2
        )
    }
    
    advanced_results = {}
    
    for name, model in advanced_models.items():
        print(f"Training {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            advanced_results[name] = {
                'model': model,
                'auc': auc_score,
                'f1': f1,
                'accuracy': accuracy
            }
            
            print(f"{name} - AUC: {auc_score:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    # Test 3: Ensemble Methods
    print("\n" + "="*50)
    print("🎯 TEST 3: ENSEMBLE METHODS")
    print("="*50)
    
    # Voting Ensemble
    base_models = [
        ('xgb', advanced_models['XGBoost Advanced']),
        ('lgb', advanced_models['LightGBM Advanced']),
        ('rf', advanced_models['Random Forest Advanced'])
    ]
    
    voting_clf = VotingClassifier(estimators=base_models, voting='soft')
    
    print("Training Voting Ensemble...")
    voting_clf.fit(X_train, y_train)
    
    y_pred_voting = voting_clf.predict(X_test)
    y_pred_proba_voting = voting_clf.predict_proba(X_test)[:, 1]
    
    auc_voting = roc_auc_score(y_test, y_pred_proba_voting)
    f1_voting = f1_score(y_test, y_pred_voting)
    accuracy_voting = accuracy_score(y_test, y_pred_voting)
    
    print(f"Voting Ensemble - AUC: {auc_voting:.4f}, F1: {f1_voting:.4f}, Accuracy: {accuracy_voting:.4f}")
    
    # Performance Comparison
    print("\n" + "="*50)
    print("📊 PERFORMANCE COMPARISON")
    print("="*50)
    
    # Get best advanced AUC
    best_advanced_auc = max([result['auc'] for result in advanced_results.values()] + [auc_voting])
    
    improvement = best_advanced_auc - best_original_auc
    improvement_pct = (improvement / best_original_auc) * 100
    
    print(f"Original Best AUC: {best_original_auc:.4f}")
    print(f"Advanced Best AUC: {best_advanced_auc:.4f}")
    print(f"Improvement: +{improvement:.4f} ({improvement_pct:.1f}%)")
    
    # Model Performance Summary
    print("\n" + "="*50)
    print("📈 MODEL PERFORMANCE SUMMARY")
    print("="*50)
    
    print("Original Models:")
    for name, result in original_results.items():
        print(f"  {name}: {result['auc']:.4f}")
    
    print("\nAdvanced Models:")
    for name, result in advanced_results.items():
        print(f"  {name}: {result['auc']:.4f}")
    
    print(f"\nEnsemble Models:")
    print(f"  Voting Ensemble: {auc_voting:.4f}")
    
    # Feature Importance Analysis
    print("\n" + "="*50)
    print("🔍 TOP FEATURE IMPORTANCE")
    print("="*50)
    
    # Get feature importance from best advanced model
    best_advanced_name = max(advanced_results.keys(), key=lambda k: advanced_results[k]['auc'])
    best_advanced_model = advanced_results[best_advanced_name]['model']
    
    if hasattr(best_advanced_model, 'feature_importances_'):
        importance = best_advanced_model.feature_importances_
        feature_names = X.columns
        top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"Top 10 features from {best_advanced_name}:")
        for i, (feature, score) in enumerate(top_features):
            print(f"  {i+1}. {feature}: {score:.4f}")
    
    # Performance Recommendations
    print("\n" + "="*50)
    print("💡 PERFORMANCE RECOMMENDATIONS")
    print("="*50)
    
    if best_advanced_auc >= 0.80:
        print("🎉 EXCELLENT! Your model achieves industry-leading performance!")
        print("✅ Ready for production deployment")
        print("✅ Suitable for fintech interviews")
    elif best_advanced_auc >= 0.75:
        print("👍 GOOD! Your model shows strong performance")
        print("✅ Suitable for fintech interviews")
        print("💡 Consider additional feature engineering for even better results")
    else:
        print("⚠️ Model needs improvement")
        print("💡 Try more advanced feature engineering")
        print("💡 Consider ensemble methods")
    
    return {
        'original_auc': best_original_auc,
        'advanced_auc': best_advanced_auc,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'best_advanced_name': best_advanced_name,
        'voting_auc': auc_voting
    }

if __name__ == "__main__":
    results = test_performance_improvements()
    print(f"\n🎯 Performance test completed!")
    print(f"Final AUC: {results['advanced_auc']:.4f}")
    print(f"Improvement: +{results['improvement']:.4f} ({results['improvement_pct']:.1f}%)")
    print(f"Best Model: {results['best_advanced_name']}")
    print(f"Voting Ensemble AUC: {results['voting_auc']:.4f}")

