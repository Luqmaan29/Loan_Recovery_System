"""
Advanced Model Training Module for Smart Digital Lending Recommendation System
Enhanced with ensemble methods, deep learning, and advanced optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelTrainer:
    """
    Advanced model training with ensemble methods, deep learning, and optimization.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the advanced model trainer."""
        self.random_state = random_state
        self.models = {}
        self.ensembles = {}
        self.best_model = None
        self.best_score = 0
        self.feature_importance = {}
        self.cv_scores = {}
        self.test_scores = {}
        
    def prepare_data_advanced(self, X: pd.DataFrame, y: pd.Series, 
                             test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
        """
        Advanced data preparation with time-aware splits.
        
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
    
    def handle_class_imbalance_advanced(self, X: pd.DataFrame, y: pd.Series, 
                                       method: str = 'adasyn') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Advanced class imbalance handling with multiple techniques.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Method to handle imbalance
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Balanced dataset
        """
        if method == 'adasyn':
            adasyn = ADASYN(random_state=self.random_state)
            X_balanced, y_balanced = adasyn.fit_resample(X, y)
        elif method == 'smote':
            smote = SMOTE(random_state=self.random_state)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        elif method == 'edited_nn':
            enn = EditedNearestNeighbours()
            X_balanced, y_balanced = enn.fit_resample(X, y)
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=self.random_state)
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
        else:
            return X, y
            
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
    
    def train_advanced_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Train advanced models including neural networks and optimized parameters.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            
        Returns:
            Dict[str, Any]: Trained models and their scores
        """
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=2000, C=0.1),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_estimators=300, max_depth=15, min_samples_split=5),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state, n_estimators=300, learning_rate=0.05),
            'SVM': SVC(random_state=self.random_state, probability=True, kernel='rbf', C=1.0, gamma='scale'),
            'Neural Network': MLPClassifier(random_state=self.random_state, hidden_layer_sizes=(200, 100, 50), max_iter=1000, alpha=0.01),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss', n_estimators=300, max_depth=8, learning_rate=0.05),
            'LightGBM': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1, n_estimators=300, max_depth=8, learning_rate=0.05)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
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
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        self.models = results
        return results
    
    def create_advanced_ensembles(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Create advanced ensemble methods.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            
        Returns:
            Dict[str, Any]: Ensemble results
        """
        ensemble_results = {}
        
        # Voting Ensemble
        voting_result = self.create_voting_ensemble(X_train, y_train, X_val, y_val)
        ensemble_results['Voting Ensemble'] = voting_result
        
        # Stacking Ensemble
        stacking_result = self.create_stacking_ensemble(X_train, y_train, X_val, y_val)
        ensemble_results['Stacking Ensemble'] = stacking_result
        
        # Blended Ensemble
        blended_result = self.create_blended_ensemble(X_train, y_train, X_val, y_val)
        ensemble_results['Blended Ensemble'] = blended_result
        
        self.ensembles = ensemble_results
        return ensemble_results
    
    def create_voting_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Create voting ensemble with optimized weights."""
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=15, random_state=self.random_state)),
            ('xgb', xgb.XGBClassifier(n_estimators=300, random_state=self.random_state, eval_metric='logloss')),
            ('lgb', lgb.LGBMClassifier(n_estimators=300, random_state=self.random_state, verbose=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=300, random_state=self.random_state)),
            ('nn', MLPClassifier(random_state=self.random_state, hidden_layer_sizes=(100, 50), max_iter=500))
        ]
        
        voting_clf = VotingClassifier(estimators=base_models, voting='soft')
        
        print("Training Voting Ensemble...")
        voting_clf.fit(X_train, y_train)
        
        y_pred = voting_clf.predict(X_val)
        y_pred_proba = voting_clf.predict_proba(X_val)[:, 1]
        
        auc_score = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Voting Ensemble - AUC: {auc_score:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        return {
            'model': voting_clf,
            'auc': auc_score,
            'f1': f1,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def create_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Create stacking ensemble with meta-learner."""
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=15, random_state=self.random_state)),
            ('xgb', xgb.XGBClassifier(n_estimators=300, random_state=self.random_state, eval_metric='logloss')),
            ('lgb', lgb.LGBMClassifier(n_estimators=300, random_state=self.random_state, verbose=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=300, random_state=self.random_state)),
            ('lr', LogisticRegression(random_state=self.random_state, max_iter=1000))
        ]
        
        meta_learner = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba'
        )
        
        print("Training Stacking Ensemble...")
        stacking_clf.fit(X_train, y_train)
        
        y_pred = stacking_clf.predict(X_val)
        y_pred_proba = stacking_clf.predict_proba(X_val)[:, 1]
        
        auc_score = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Stacking Ensemble - AUC: {auc_score:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        return {
            'model': stacking_clf,
            'auc': auc_score,
            'f1': f1,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def create_blended_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Create blended ensemble with cross-validation."""
        # Train base models
        base_models = {
            'rf': RandomForestClassifier(n_estimators=300, max_depth=15, random_state=self.random_state),
            'xgb': xgb.XGBClassifier(n_estimators=300, random_state=self.random_state, eval_metric='logloss'),
            'lgb': lgb.LGBMClassifier(n_estimators=300, random_state=self.random_state, verbose=-1)
        }
        
        # Cross-validation predictions
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_predictions = np.zeros((len(X_train), len(base_models)))
        
        for i, (name, model) in enumerate(base_models.items()):
            fold_predictions = np.zeros(len(X_train))
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                fold_predictions[val_idx] = model.predict_proba(X_fold_val)[:, 1]
            
            cv_predictions[:, i] = fold_predictions
        
        # Train meta-learner
        meta_learner = LogisticRegression(random_state=self.random_state, max_iter=1000)
        meta_learner.fit(cv_predictions, y_train)
        
        # Get validation predictions
        val_predictions = np.zeros((len(X_val), len(base_models)))
        for i, (name, model) in enumerate(base_models.items()):
            model.fit(X_train, y_train)
            val_predictions[:, i] = model.predict_proba(X_val)[:, 1]
        
        # Final predictions
        y_pred_proba = meta_learner.predict_proba(val_predictions)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        auc_score = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Blended Ensemble - AUC: {auc_score:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        return {
            'model': meta_learner,
            'base_models': base_models,
            'auc': auc_score,
            'f1': f1,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def optimize_hyperparameters_advanced(self, X_train: pd.DataFrame, y_train: pd.Series,
                                         X_val: pd.DataFrame, y_val: pd.Series,
                                         model_name: str = 'XGBoost', n_trials: int = 300) -> Dict:
        """
        Advanced hyperparameter optimization with pruning and early stopping.
        
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
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                    'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'random_state': self.random_state,
                    'eval_metric': 'logloss'
                }
                model = xgb.XGBClassifier(**params)
                
            elif model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10),
                    'random_state': self.random_state,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params)
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            return auc_score
        
        # Run optimization with pruning
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        
        # Train best model
        best_params = study.best_params
        if model_name == 'XGBoost':
            best_model = xgb.XGBClassifier(**best_params)
        elif model_name == 'LightGBM':
            best_model = lgb.LGBMClassifier(**best_params)
        
        best_model.fit(X_train, y_train)
        
        return {
            'best_params': best_params,
            'best_model': best_model,
            'best_score': study.best_value
        }
    
    def get_best_model(self) -> Tuple[Any, str, float]:
        """
        Get the best performing model from all trained models and ensembles.
        
        Returns:
            Tuple[Any, str, float]: Best model, name, and score
        """
        all_results = {}
        
        # Add individual models
        for name, result in self.models.items():
            all_results[name] = result['auc']
        
        # Add ensembles
        for name, result in self.ensembles.items():
            all_results[name] = result['auc']
        
        # Find best
        best_name = max(all_results, key=all_results.get)
        best_score = all_results[best_name]
        
        if best_name in self.models:
            best_model = self.models[best_name]['model']
        else:
            best_model = self.ensembles[best_name]['model']
        
        self.best_model = best_model
        self.best_score = best_score
        
        return best_model, best_name, best_score
    
    def evaluate_final_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Comprehensive evaluation of the best model.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict: Comprehensive evaluation results
        """
        if self.best_model is None:
            raise ValueError("No best model found. Train models first.")
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate comprehensive metrics
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


if __name__ == "__main__":
    # Example usage
    print("Advanced Model Trainer - Ready for high-performance training!")




