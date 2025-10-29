"""
Simple Model Trainer - Logistic Regression + XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb

class SimpleModelTrainer:
    """Train and evaluate models on simple loan dataset."""
    
    def __init__(self):
        self.models = {}
        self.feature_names = []
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train Logistic Regression and XGBoost."""
        
        print("\n🤖 Training Machine Learning Models...")
        print("=" * 60)
        
        self.feature_names = X_train.columns.tolist()
        results = {}
        
        # 1. Logistic Regression
        print("\n📊 Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        
        y_pred_proba_lr = lr_model.predict_proba(X_val)[:, 1]
        y_pred_lr = (y_pred_proba_lr > 0.5).astype(int)
        
        auc_lr = roc_auc_score(y_val, y_pred_proba_lr)
        acc_lr = accuracy_score(y_val, y_pred_lr)
        
        results['Logistic Regression'] = {
            'model': lr_model,
            'auc': auc_lr,
            'accuracy': acc_lr,
            'predictions': y_pred_lr,
            'probabilities': y_pred_proba_lr
        }
        
        print(f"   AUC: {auc_lr:.4f}")
        print(f"   Accuracy: {acc_lr:.4f}")
        
        # 2. XGBoost
        print("\n📊 Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        
        y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
        y_pred_xgb = (y_pred_proba_xgb > 0.5).astype(int)
        
        auc_xgb = roc_auc_score(y_val, y_pred_proba_xgb)
        acc_xgb = accuracy_score(y_val, y_pred_xgb)
        
        results['XGBoost'] = {
            'model': xgb_model,
            'auc': auc_xgb,
            'accuracy': acc_xgb,
            'predictions': y_pred_xgb,
            'probabilities': y_pred_proba_xgb
        }
        
        print(f"   AUC: {auc_xgb:.4f}")
        print(f"   Accuracy: {acc_xgb:.4f}")
        
        self.models = results
        
        # Print comparison
        print("\n" + "=" * 60)
        print("📊 MODEL COMPARISON")
        print("=" * 60)
        for name, result in results.items():
            print(f"{name:20s} | AUC: {result['auc']:.4f} | Accuracy: {result['accuracy']:.4f}")
        
        # Determine best model
        best_model_name = max(results.items(), key=lambda x: x[1]['auc'])[0]
        print(f"\n🏆 Best Model: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")
        
        return results
    
    def predict(self, model_name, X):
        """Make predictions with a specific model."""
        if model_name in self.models:
            model = self.models[model_name]['model']
            probabilities = model.predict_proba(X)[:, 1]
            predictions = (probabilities > 0.5).astype(int)
            return predictions, probabilities
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def get_feature_importance(self):
        """Get feature importance from XGBoost."""
        if 'XGBoost' in self.models:
            model = self.models['XGBoost']['model']
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            return feature_importance
        return None

