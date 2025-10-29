"""
Performance Testing Script for Smart Digital Lending Recommendation System
Tests the improved model performance with advanced techniques
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
from advanced_model_trainer import AdvancedModelTrainer

def test_performance_improvements():
    """Test the performance improvements with advanced techniques."""
    
    print("🚀 Testing Performance Improvements")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    loader = RealDataLoader()
    data = loader.load_all_data()
    
    # Create sample for testing
    sample_data = loader.create_sample_for_demo(10000)  # Larger sample for better results
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
    
    # Prepare data for training
    X = engineered_df.drop(['TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')
    y = engineered_df['TARGET'].dropna()
    X = X.loc[y.index]
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts()}")
    
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
    
    # Test 2: Advanced Model Performance
    print("\n" + "="*50)
    print("🚀 TEST 2: ADVANCED MODEL PERFORMANCE")
    print("="*50)
    
    trainer_advanced = AdvancedModelTrainer()
    X_train_adv, X_val_adv, X_test_adv, y_train_adv, y_val_adv, y_test_adv = trainer_advanced.prepare_data_advanced(X, y)
    
    # Handle class imbalance with advanced method
    X_train_balanced_adv, y_train_balanced_adv = trainer_advanced.handle_class_imbalance_advanced(
        X_train_adv, y_train_adv, method='adasyn'
    )
    
    # Train advanced models
    print("Training advanced models...")
    advanced_results = trainer_advanced.train_advanced_models(
        X_train_balanced_adv, y_train_balanced_adv, X_val_adv, y_val_adv
    )
    
    # Create ensembles
    print("\nCreating advanced ensembles...")
    ensemble_results = trainer_advanced.create_advanced_ensembles(
        X_train_balanced_adv, y_train_balanced_adv, X_val_adv, y_val_adv
    )
    
    # Get best advanced model
    best_model, best_name, best_score = trainer_advanced.get_best_model()
    print(f"\nAdvanced Best Model: {best_name}")
    print(f"Advanced Best AUC: {best_score:.4f}")
    
    # Test 3: Hyperparameter Optimization
    print("\n" + "="*50)
    print("🎯 TEST 3: HYPERPARAMETER OPTIMIZATION")
    print("="*50)
    
    print("Optimizing XGBoost hyperparameters...")
    opt_results = trainer_advanced.optimize_hyperparameters_advanced(
        X_train_balanced_adv, y_train_balanced_adv, X_val_adv, y_val_adv, 
        model_name='XGBoost', n_trials=100  # Reduced for testing
    )
    
    print(f"Optimized XGBoost AUC: {opt_results['best_score']:.4f}")
    
    # Test 4: Final Evaluation
    print("\n" + "="*50)
    print("🏆 TEST 4: FINAL EVALUATION")
    print("="*50)
    
    # Evaluate best model on test set
    final_evaluation = trainer_advanced.evaluate_final_model(X_test_adv, y_test_adv)
    
    print(f"Final Test AUC: {final_evaluation['auc']:.4f}")
    print(f"Final Test F1: {final_evaluation['f1']:.4f}")
    print(f"Final Test Accuracy: {final_evaluation['accuracy']:.4f}")
    
    # Performance Comparison
    print("\n" + "="*50)
    print("📊 PERFORMANCE COMPARISON")
    print("="*50)
    
    improvement = final_evaluation['auc'] - original_summary['best_score']
    improvement_pct = (improvement / original_summary['best_score']) * 100
    
    print(f"Original Best AUC: {original_summary['best_score']:.4f}")
    print(f"Advanced Best AUC: {final_evaluation['auc']:.4f}")
    print(f"Improvement: +{improvement:.4f} ({improvement_pct:.1f}%)")
    
    # Model Performance Summary
    print("\n" + "="*50)
    print("📈 MODEL PERFORMANCE SUMMARY")
    print("="*50)
    
    print("Individual Models:")
    for name, result in advanced_results.items():
        print(f"  {name}: {result['auc']:.4f}")
    
    print("\nEnsemble Models:")
    for name, result in ensemble_results.items():
        print(f"  {name}: {result['auc']:.4f}")
    
    # Feature Importance Analysis
    if hasattr(trainer_advanced, 'feature_importance') and trainer_advanced.feature_importance:
        print("\n" + "="*50)
        print("🔍 TOP FEATURE IMPORTANCE")
        print("="*50)
        
        # Get feature importance from best model
        if best_name in trainer_advanced.feature_importance:
            importance = trainer_advanced.feature_importance[best_name]
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            print(f"Top 10 features from {best_name}:")
            for i, (feature, score) in enumerate(sorted_features[:10]):
                print(f"  {i+1}. {feature}: {score:.4f}")
    
    # Performance Recommendations
    print("\n" + "="*50)
    print("💡 PERFORMANCE RECOMMENDATIONS")
    print("="*50)
    
    if final_evaluation['auc'] >= 0.80:
        print("🎉 EXCELLENT! Your model achieves industry-leading performance!")
        print("✅ Ready for production deployment")
        print("✅ Suitable for fintech interviews")
    elif final_evaluation['auc'] >= 0.75:
        print("👍 GOOD! Your model shows strong performance")
        print("✅ Suitable for fintech interviews")
        print("💡 Consider additional feature engineering for even better results")
    else:
        print("⚠️ Model needs improvement")
        print("💡 Try more advanced feature engineering")
        print("💡 Consider ensemble methods")
    
    return {
        'original_auc': original_summary['best_score'],
        'advanced_auc': final_evaluation['auc'],
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'final_f1': final_evaluation['f1'],
        'final_accuracy': final_evaluation['accuracy']
    }

if __name__ == "__main__":
    results = test_performance_improvements()
    print(f"\n🎯 Performance test completed!")
    print(f"Final AUC: {results['advanced_auc']:.4f}")
    print(f"Improvement: +{results['improvement']:.4f} ({results['improvement_pct']:.1f}%)")




