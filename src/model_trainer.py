"""
Model Training Module for Smart Digital Lending Recommendation System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Comprehensive model training and evaluation for the lending recommendation system.
    Supports multiple algorithms and hyperparameter optimization.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.feature_importance = {}
        self.cv_scores = {}
        self.test_scores = {}
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
        """
        Prepare data for training with train/validation/test splits.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Test set size
            val_size (float): Validation set size
            
        Returns:
            Tuple: Train, validation, and test sets
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, 
                              method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance in the dataset.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Method to handle imbalance ('smote', 'undersample', 'oversample')
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Balanced dataset
        """
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=self.random_state)
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
        elif method == 'oversample':
            from imblearn.over_sampling import RandomOverSampler
            oversampler = RandomOverSampler(random_state=self.random_state)
            X_balanced, y_balanced = oversampler.fit_resample(X, y)
        else:
            return X, y
            
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
    
    def train_baseline_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Train baseline models for comparison.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            
        Returns:
            Dict[str, Any]: Trained models and their scores
        """
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
            'SVM': SVC(random_state=self.random_state, probability=True),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_val, y_pred_proba)
            f1 = f1_score(y_val, y_pred)
            accuracy = accuracy_score(y_val, y_pred)
            
            results[name] = {
                'model': model,
                'auc': auc_score,
                'f1': f1,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X_train.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                self.feature_importance[name] = dict(zip(X_train.columns, model.coef_[0]))
            
            print(f"{name} - AUC: {auc_score:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        self.models = results
        return results
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series,
                                model_name: str = 'XGBoost', n_trials: int = 100) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            model_name (str): Model to optimize
            n_trials (int): Number of optimization trials
            
        Returns:
            Dict: Best parameters and model
        """
        def objective(trial):
            if model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': self.random_state
                }
                model = xgb.XGBClassifier(**params)
                
            elif model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': self.random_state,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params)
                
            elif model_name == 'Random Forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': self.random_state
                }
                model = RandomForestClassifier(**params)
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            return auc_score
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Train best model
        best_params = study.best_params
        if model_name == 'XGBoost':
            best_model = xgb.XGBClassifier(**best_params)
        elif model_name == 'LightGBM':
            best_model = lgb.LGBMClassifier(**best_params)
        elif model_name == 'Random Forest':
            best_model = RandomForestClassifier(**best_params)
        
        best_model.fit(X_train, y_train)
        
        return {
            'best_params': best_params,
            'best_model': best_model,
            'best_score': study.best_value
        }
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, 
                           model: Any, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation on a model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            model: Model to validate
            cv_folds (int): Number of CV folds
            
        Returns:
            Dict: Cross-validation results
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Calculate multiple metrics
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'auc_mean': auc_scores.mean(),
            'auc_std': auc_scores.std(),
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'accuracy_mean': accuracy_scores.mean(),
            'accuracy_std': accuracy_scores.std(),
            'scores': {
                'auc': auc_scores,
                'f1': f1_scores,
                'accuracy': accuracy_scores
            }
        }
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        
        # Precision-Recall curve data
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        return {
            'auc': auc_score,
            'f1': f1,
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'roc_curve': (fpr, tpr, roc_thresholds),
            'pr_curve': (precision, recall, pr_thresholds),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def plot_model_performance(self, evaluation_results: Dict, save_path: str = None):
        """
        Plot model performance metrics.
        
        Args:
            evaluation_results (Dict): Evaluation results
            save_path (str): Path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        fpr, tpr, _ = evaluation_results['roc_curve']
        axes[0, 0].plot(fpr, tpr, label=f"ROC Curve (AUC = {evaluation_results['auc']:.3f})")
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Precision-Recall Curve
        precision, recall, _ = evaluation_results['pr_curve']
        axes[0, 1].plot(recall, precision, label=f"PR Curve (F1 = {evaluation_results['f1']:.3f})")
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Confusion Matrix
        cm = evaluation_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Feature Importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
            top_features = np.argsort(importance)[-10:]
            
            axes[1, 1].barh(range(len(top_features)), importance[top_features])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels([feature_names[i] for i in top_features])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model: Any, filepath: str):
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            filepath (str): Path to save model
        """
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from disk.
        
        Args:
            filepath (str): Path to model file
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of all trained models.
        
        Returns:
            Dict: Model summary
        """
        summary = {
            'total_models': len(self.models),
            'best_model': None,
            'best_score': 0,
            'model_scores': {}
        }
        
        for name, results in self.models.items():
            summary['model_scores'][name] = {
                'auc': results['auc'],
                'f1': results['f1'],
                'accuracy': results['accuracy']
            }
            
            if results['auc'] > summary['best_score']:
                summary['best_score'] = results['auc']
                summary['best_model'] = name
        
        return summary


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Load and preprocess data
    loader = DataLoader()
    data = loader.load_all_data()
    
    preprocessor = DataPreprocessor()
    processed_df, features = preprocessor.preprocess_pipeline(data['application'], data)
    
    engineer = FeatureEngineer()
    engineered_df = engineer.create_all_features(processed_df)
    
    # Prepare data for training
    X = engineered_df.drop(['TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')
    y = engineered_df['TARGET'].dropna()
    X = X.loc[y.index]
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced = trainer.handle_class_imbalance(X_train, y_train)
    
    # Train baseline models
    results = trainer.train_baseline_models(X_train_balanced, y_train_balanced, X_val, y_val)
    
    # Get model summary
    summary = trainer.get_model_summary()
    print(f"Best model: {summary['best_model']} with AUC: {summary['best_score']:.4f}")


