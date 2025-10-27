"""
Simplified Performance Test for Smart Digital Lending Recommendation System
Tests the improved model performance with proper data handling
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
from model_trainer import ModelTrainer

def test_performance_improvements():
    """Test the performance improvements with advanced techniques."""
    
    print("🚀 Testing Performance Improvements")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    loader = RealDataLoader()
    data = loader.load_all_data()
    
    # Create sample for testing
    sample_data = loader.create_sample_for_demo(5000)  # Smaller sample for faster testing
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
    
    # Clean data - ensure all columns are numeric
    print("\n🧹 Cleaning data for ML...")
    # Remove any remaining non-numeric columns except target and ID
    numeric_cols = engineered_df.select_dtypes(include=[np.number]).columns
    keep_cols = ['TARGET', 'SK_ID_CURR'] + [col for col in numeric_cols if col not in ['TARGET', 'SK_ID_CURR']]
    engineered_df = engineered_df[keep_cols]
    
    # Prepare data for training
    X = engineered_df.drop(['TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')
    y = engineered_df['TARGET'].dropna()
    X = X.loc[y.index]
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts()}")
    print(f"Data types: {X.dtypes.value_counts()}")
    
    # Test 1: Original Model Performance
    print("\n" + "="*50)
    print("📈 TEST 1: ORIGINAL MODEL PERFORMANCE")
    print("="*50)
    
    trainer_original = ModelTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer_original.prepare_data(X, y)
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced = trainer_original.handle_class_imbalance(X_train, y_train, method='smote')
    
    # Train baseline models
    print("Training original models...")
    original_results = trainer_original.train_baseline_models(X_train_balanced, y_train_balanced, X_val, y_val)
    
    # Get best original model
    original_summary = trainer_original.get_model_summary()
    print(f"\nOriginal Best Model: {original_summary['best_model']}")
    print(f"Original Best AUC: {original_summary['best_score']:.4f}")
    
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
            model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_val, y_pred_proba)
            f1 = f1_score(y_val, y_pred)
            accuracy = accuracy_score(y_val, y_pred)
            
            advanced_results[name] = {
                'model': model,
                'auc': auc_score,
                'f1': f1,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
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
    from sklearn.ensemble import VotingClassifier
    
    base_models = [
        ('xgb', advanced_models['XGBoost Advanced']),
        ('lgb', advanced_models['LightGBM Advanced']),
        ('rf', advanced_models['Random Forest Advanced'])
    ]
    
    voting_clf = VotingClassifier(estimators=base_models, voting='soft')
    
    print("Training Voting Ensemble...")
    voting_clf.fit(X_train_balanced, y_train_balanced)
    
    y_pred_voting = voting_clf.predict(X_val)
    y_pred_proba_voting = voting_clf.predict_proba(X_val)[:, 1]
    
    auc_voting = roc_auc_score(y_val, y_pred_proba_voting)
    f1_voting = f1_score(y_val, y_pred_voting)
    accuracy_voting = accuracy_score(y_val, y_pred_voting)
    
    print(f"Voting Ensemble - AUC: {auc_voting:.4f}, F1: {f1_voting:.4f}, Accuracy: {accuracy_voting:.4f}")
    
    # Test 4: Final Evaluation
    print("\n" + "="*50)
    print("🏆 TEST 4: FINAL EVALUATION")
    print("="*50)
    
    # Use the best model for final evaluation
    best_auc = max([result['auc'] for result in advanced_results.values()] + [auc_voting])
    
    if auc_voting == best_auc:
        best_model = voting_clf
        best_name = "Voting Ensemble"
        print("Using Voting Ensemble for final evaluation")
    else:
        best_name = max(advanced_results.keys(), key=lambda k: advanced_results[k]['auc'])
        best_model = advanced_results[best_name]['model']
        print(f"Using {best_name} for final evaluation")
    
    # Evaluate on test set
    y_pred_final = best_model.predict(X_test)
    y_pred_proba_final = best_model.predict_proba(X_test)[:, 1]
    
    final_auc = roc_auc_score(y_test, y_pred_proba_final)
    final_f1 = f1_score(y_test, y_pred_final)
    final_accuracy = accuracy_score(y_test, y_pred_final)
    
    print(f"Final Test AUC: {final_auc:.4f}")
    print(f"Final Test F1: {final_f1:.4f}")
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    
    # Performance Comparison
    print("\n" + "="*50)
    print("📊 PERFORMANCE COMPARISON")
    print("="*50)
    
    improvement = final_auc - original_summary['best_score']
    improvement_pct = (improvement / original_summary['best_score']) * 100
    
    print(f"Original Best AUC: {original_summary['best_score']:.4f}")
    print(f"Advanced Best AUC: {final_auc:.4f}")
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
    
    # Performance Recommendations
    print("\n" + "="*50)
    print("💡 PERFORMANCE RECOMMENDATIONS")
    print("="*50)
    
    if final_auc >= 0.80:
        print("🎉 EXCELLENT! Your model achieves industry-leading performance!")
        print("✅ Ready for production deployment")
        print("✅ Suitable for fintech interviews")
    elif final_auc >= 0.75:
        print("👍 GOOD! Your model shows strong performance")
        print("✅ Suitable for fintech interviews")
        print("💡 Consider additional feature engineering for even better results")
    else:
        print("⚠️ Model needs improvement")
        print("💡 Try more advanced feature engineering")
        print("💡 Consider ensemble methods")
    
    return {
        'original_auc': original_summary['best_score'],
        'advanced_auc': final_auc,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'final_f1': final_f1,
        'final_accuracy': final_accuracy,
        'best_model_name': best_name
    }

if __name__ == "__main__":
    # Import required modules
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    
    results = test_performance_improvements()
    print(f"\n🎯 Performance test completed!")
    print(f"Final AUC: {results['advanced_auc']:.4f}")
    print(f"Improvement: +{results['improvement']:.4f} ({results['improvement_pct']:.1f}%)")
    print(f"Best Model: {results['best_model_name']}")



