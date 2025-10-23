"""
Recommendation Engine for Smart Digital Lending Recommendation System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import warnings
warnings.filterwarnings('ignore')

class RiskCategory:
    """Risk category definitions for loan recommendations."""
    
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"
    
    # Risk thresholds
    LOW_RISK_THRESHOLD = 0.3
    HIGH_RISK_THRESHOLD = 0.7

class LoanRecommendation:
    """Loan recommendation result."""
    
    def __init__(self, client_id: str, risk_category: str, probability_of_default: float,
                 recommendation: str, confidence_score: float, 
                 suggested_interest_rate: float = None, 
                 suggested_loan_amount: float = None,
                 reasoning: List[str] = None):
        """
        Initialize loan recommendation.
        
        Args:
            client_id (str): Client identifier
            risk_category (str): Risk category (LOW_RISK, MEDIUM_RISK, HIGH_RISK)
            probability_of_default (float): Probability of default (0-1)
            recommendation (str): Recommendation (APPROVE, REVIEW, REJECT)
            confidence_score (float): Confidence in the recommendation (0-1)
            suggested_interest_rate (float): Suggested interest rate
            suggested_loan_amount (float): Suggested loan amount
            reasoning (List[str]): Reasoning for the recommendation
        """
        self.client_id = client_id
        self.risk_category = risk_category
        self.probability_of_default = probability_of_default
        self.recommendation = recommendation
        self.confidence_score = confidence_score
        self.suggested_interest_rate = suggested_interest_rate
        self.suggested_loan_amount = suggested_loan_amount
        self.reasoning = reasoning or []

class RecommendationEngine:
    """
    Advanced recommendation engine for loan decisions.
    Combines ML predictions with business rules and risk assessment.
    """
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize the recommendation engine.
        
        Args:
            model_path (str): Path to trained model
            scaler_path (str): Path to fitted scaler
        """
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.business_rules = {}
        
        if model_path:
            self.load_model(model_path)
        if scaler_path:
            self.load_scaler(scaler_path)
    
    def load_model(self, model_path: str):
        """Load trained model from file."""
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def load_scaler(self, scaler_path: str):
        """Load fitted scaler from file."""
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
    
    def set_business_rules(self, rules: Dict[str, Any]):
        """
        Set business rules for recommendations.
        
        Args:
            rules (Dict[str, Any]): Business rules dictionary
        """
        self.business_rules = rules
    
    def calculate_risk_category(self, probability_of_default: float) -> str:
        """
        Calculate risk category based on probability of default.
        
        Args:
            probability_of_default (float): Probability of default (0-1)
            
        Returns:
            str: Risk category
        """
        if probability_of_default <= RiskCategory.LOW_RISK_THRESHOLD:
            return RiskCategory.LOW_RISK
        elif probability_of_default <= RiskCategory.HIGH_RISK_THRESHOLD:
            return RiskCategory.MEDIUM_RISK
        else:
            return RiskCategory.HIGH_RISK
    
    def generate_recommendation(self, risk_category: str, probability_of_default: float,
                              client_features: Dict[str, Any]) -> str:
        """
        Generate loan recommendation based on risk category and business rules.
        
        Args:
            risk_category (str): Risk category
            probability_of_default (float): Probability of default
            client_features (Dict[str, Any]): Client features
            
        Returns:
            str: Recommendation (APPROVE, REVIEW, REJECT)
        """
        # Base recommendation from risk category
        if risk_category == RiskCategory.LOW_RISK:
            base_recommendation = "APPROVE"
        elif risk_category == RiskCategory.MEDIUM_RISK:
            base_recommendation = "REVIEW"
        else:
            base_recommendation = "REJECT"
        
        # Apply business rules
        recommendation = self._apply_business_rules(
            base_recommendation, risk_category, probability_of_default, client_features
        )
        
        return recommendation
    
    def _apply_business_rules(self, base_recommendation: str, risk_category: str,
                            probability_of_default: float, 
                            client_features: Dict[str, Any]) -> str:
        """
        Apply business rules to modify base recommendation.
        
        Args:
            base_recommendation (str): Base recommendation
            risk_category (str): Risk category
            probability_of_default (float): Probability of default
            client_features (Dict[str, Any]): Client features
            
        Returns:
            str: Modified recommendation
        """
        recommendation = base_recommendation
        
        # Rule 1: High income clients get more lenient treatment
        if 'AMT_INCOME_TOTAL' in client_features:
            income_threshold = self.business_rules.get('high_income_threshold', 100000)
            if client_features['AMT_INCOME_TOTAL'] > income_threshold:
                if recommendation == "REJECT" and probability_of_default < 0.8:
                    recommendation = "REVIEW"
                elif recommendation == "REVIEW" and probability_of_default < 0.5:
                    recommendation = "APPROVE"
        
        # Rule 2: Long-term employment gets preference
        if 'EMPLOYMENT_YEARS' in client_features:
            employment_threshold = self.business_rules.get('stable_employment_years', 5)
            if client_features['EMPLOYMENT_YEARS'] > employment_threshold:
                if recommendation == "REJECT" and probability_of_default < 0.75:
                    recommendation = "REVIEW"
        
        # Rule 3: Previous good payment history
        if 'PREV_AVERAGE_AMOUNT' in client_features and 'PAYMENT_AVERAGE_MEAN' in client_features:
            if (client_features['PREV_AVERAGE_AMOUNT'] > 0 and 
                client_features['PAYMENT_AVERAGE_MEAN'] > 0.95):  # 95% payment rate
                if recommendation == "REJECT" and probability_of_default < 0.7:
                    recommendation = "REVIEW"
        
        # Rule 4: Very high risk always rejected
        if probability_of_default > 0.9:
            recommendation = "REJECT"
        
        # Rule 5: Very low risk always approved
        if probability_of_default < 0.1:
            recommendation = "APPROVE"
        
        return recommendation
    
    def calculate_confidence_score(self, probability_of_default: float, 
                                   client_features: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the recommendation.
        
        Args:
            probability_of_default (float): Probability of default
            client_features (Dict[str, Any]): Client features
            
        Returns:
            float: Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence
        
        # Higher confidence for extreme probabilities
        if probability_of_default < 0.1 or probability_of_default > 0.9:
            confidence += 0.3
        
        # Higher confidence for complete information
        missing_info_penalty = 0
        important_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AGE_YEARS', 'EMPLOYMENT_YEARS']
        for feature in important_features:
            if feature not in client_features or pd.isna(client_features[feature]):
                missing_info_penalty += 0.1
        
        confidence -= missing_info_penalty
        
        # Higher confidence for consistent indicators
        if 'EXT_SOURCE_AVG' in client_features:
            if client_features['EXT_SOURCE_AVG'] > 0.7 and probability_of_default < 0.3:
                confidence += 0.1
            elif client_features['EXT_SOURCE_AVG'] < 0.3 and probability_of_default > 0.7:
                confidence += 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def suggest_interest_rate(self, risk_category: str, probability_of_default: float,
                             base_rate: float = 0.05) -> float:
        """
        Suggest interest rate based on risk assessment.
        
        Args:
            risk_category (str): Risk category
            probability_of_default (float): Probability of default
            base_rate (float): Base interest rate
            
        Returns:
            float: Suggested interest rate
        """
        if risk_category == RiskCategory.LOW_RISK:
            return base_rate + 0.01  # 1% above base rate
        elif risk_category == RiskCategory.MEDIUM_RISK:
            return base_rate + 0.03  # 3% above base rate
        else:
            return base_rate + 0.05  # 5% above base rate
    
    def suggest_loan_amount(self, requested_amount: float, risk_category: str,
                          probability_of_default: float, income: float = None) -> float:
        """
        Suggest loan amount based on risk assessment.
        
        Args:
            requested_amount (float): Requested loan amount
            risk_category (str): Risk category
            probability_of_default (float): Probability of default
            income (float): Client income
            
        Returns:
            float: Suggested loan amount
        """
        if risk_category == RiskCategory.HIGH_RISK:
            return 0  # No loan for high risk
        
        # Calculate maximum loan based on income (if available)
        max_loan_by_income = None
        if income:
            max_loan_by_income = income * 0.3  # 30% of annual income
        
        # Risk-based adjustments
        if risk_category == RiskCategory.LOW_RISK:
            risk_multiplier = 1.0
        elif risk_category == RiskCategory.MEDIUM_RISK:
            risk_multiplier = 0.8
        else:
            risk_multiplier = 0.0
        
        suggested_amount = requested_amount * risk_multiplier
        
        # Apply income constraint if available
        if max_loan_by_income:
            suggested_amount = min(suggested_amount, max_loan_by_income)
        
        return max(0, suggested_amount)
    
    def generate_reasoning(self, risk_category: str, probability_of_default: float,
                          client_features: Dict[str, Any]) -> List[str]:
        """
        Generate reasoning for the recommendation.
        
        Args:
            risk_category (str): Risk category
            probability_of_default (float): Probability of default
            client_features (Dict[str, Any]): Client features
            
        Returns:
            List[str]: List of reasoning statements
        """
        reasoning = []
        
        # Risk level reasoning
        if probability_of_default < 0.2:
            reasoning.append("Low probability of default based on historical data")
        elif probability_of_default < 0.5:
            reasoning.append("Moderate risk profile requiring careful review")
        else:
            reasoning.append("High probability of default based on risk factors")
        
        # Income reasoning
        if 'AMT_INCOME_TOTAL' in client_features:
            income = client_features['AMT_INCOME_TOTAL']
            if income > 100000:
                reasoning.append("High income provides strong repayment capacity")
            elif income < 30000:
                reasoning.append("Low income may limit repayment capacity")
        
        # Employment reasoning
        if 'EMPLOYMENT_YEARS' in client_features:
            employment_years = client_features['EMPLOYMENT_YEARS']
            if employment_years > 5:
                reasoning.append("Stable employment history indicates reliability")
            elif employment_years < 1:
                reasoning.append("Limited employment history increases risk")
        
        # Credit history reasoning
        if 'BUREAU_AVERAGE_SCORE' in client_features:
            bureau_score = client_features['BUREAU_AVERAGE_SCORE']
            if bureau_score > 0.7:
                reasoning.append("Strong credit history from external sources")
            elif bureau_score < 0.3:
                reasoning.append("Poor credit history from external sources")
        
        # Payment behavior reasoning
        if 'PAYMENT_AVERAGE_MEAN' in client_features:
            payment_rate = client_features['PAYMENT_AVERAGE_MEAN']
            if payment_rate > 0.95:
                reasoning.append("Excellent payment history on previous loans")
            elif payment_rate < 0.8:
                reasoning.append("Poor payment history on previous loans")
        
        return reasoning
    
    def predict_single_client(self, client_features: Dict[str, Any]) -> LoanRecommendation:
        """
        Predict and recommend for a single client.
        
        Args:
            client_features (Dict[str, Any]): Client features
            
        Returns:
            LoanRecommendation: Loan recommendation object
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a trained model first.")
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([client_features])
        
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(feature_df.columns)
        if missing_features:
            for feature in missing_features:
                feature_df[feature] = 0  # Fill missing with 0
        
        # Reorder columns to match training data
        feature_df = feature_df[self.feature_columns]
        
        # Scale features if scaler is available
        if self.scaler:
            feature_df = pd.DataFrame(
                self.scaler.transform(feature_df),
                columns=feature_df.columns
            )
        
        # Get probability of default
        probability_of_default = self.model.predict_proba(feature_df)[0, 1]
        
        # Calculate risk category
        risk_category = self.calculate_risk_category(probability_of_default)
        
        # Generate recommendation
        recommendation = self.generate_recommendation(
            risk_category, probability_of_default, client_features
        )
        
        # Calculate confidence
        confidence_score = self.calculate_confidence_score(
            probability_of_default, client_features
        )
        
        # Suggest interest rate and loan amount
        suggested_interest_rate = self.suggest_interest_rate(
            risk_category, probability_of_default
        )
        
        requested_amount = client_features.get('AMT_CREDIT', 0)
        suggested_loan_amount = self.suggest_loan_amount(
            requested_amount, risk_category, probability_of_default,
            client_features.get('AMT_INCOME_TOTAL')
        )
        
        # Generate reasoning
        reasoning = self.generate_reasoning(
            risk_category, probability_of_default, client_features
        )
        
        return LoanRecommendation(
            client_id=str(client_features.get('SK_ID_CURR', 'Unknown')),
            risk_category=risk_category,
            probability_of_default=probability_of_default,
            recommendation=recommendation,
            confidence_score=confidence_score,
            suggested_interest_rate=suggested_interest_rate,
            suggested_loan_amount=suggested_loan_amount,
            reasoning=reasoning
        )
    
    def predict_batch(self, clients_df: pd.DataFrame) -> List[LoanRecommendation]:
        """
        Predict and recommend for multiple clients.
        
        Args:
            clients_df (pd.DataFrame): DataFrame with client features
            
        Returns:
            List[LoanRecommendation]: List of loan recommendations
        """
        recommendations = []
        
        for idx, row in clients_df.iterrows():
            client_features = row.to_dict()
            recommendation = self.predict_single_client(client_features)
            recommendations.append(recommendation)
        
        return recommendations
    
    def get_recommendation_summary(self, recommendations: List[LoanRecommendation]) -> Dict:
        """
        Get summary statistics for a batch of recommendations.
        
        Args:
            recommendations (List[LoanRecommendation]): List of recommendations
            
        Returns:
            Dict: Summary statistics
        """
        total_clients = len(recommendations)
        
        # Count recommendations
        approve_count = sum(1 for r in recommendations if r.recommendation == "APPROVE")
        review_count = sum(1 for r in recommendations if r.recommendation == "REVIEW")
        reject_count = sum(1 for r in recommendations if r.recommendation == "REJECT")
        
        # Count risk categories
        low_risk_count = sum(1 for r in recommendations if r.risk_category == RiskCategory.LOW_RISK)
        medium_risk_count = sum(1 for r in recommendations if r.risk_category == RiskCategory.MEDIUM_RISK)
        high_risk_count = sum(1 for r in recommendations if r.risk_category == RiskCategory.HIGH_RISK)
        
        # Calculate averages
        avg_probability = np.mean([r.probability_of_default for r in recommendations])
        avg_confidence = np.mean([r.confidence_score for r in recommendations])
        avg_interest_rate = np.mean([r.suggested_interest_rate for r in recommendations if r.suggested_interest_rate])
        avg_loan_amount = np.mean([r.suggested_loan_amount for r in recommendations if r.suggested_loan_amount])
        
        return {
            'total_clients': total_clients,
            'recommendations': {
                'approve': approve_count,
                'review': review_count,
                'reject': reject_count
            },
            'risk_categories': {
                'low_risk': low_risk_count,
                'medium_risk': medium_risk_count,
                'high_risk': high_risk_count
            },
            'averages': {
                'probability_of_default': avg_probability,
                'confidence_score': avg_confidence,
                'interest_rate': avg_interest_rate,
                'loan_amount': avg_loan_amount
            }
        }


if __name__ == "__main__":
    # Example usage
    engine = RecommendationEngine()
    
    # Set business rules
    business_rules = {
        'high_income_threshold': 100000,
        'stable_employment_years': 5
    }
    engine.set_business_rules(business_rules)
    
    # Example client features
    client_features = {
        'SK_ID_CURR': 12345,
        'AMT_INCOME_TOTAL': 80000,
        'AMT_CREDIT': 50000,
        'AGE_YEARS': 35,
        'EMPLOYMENT_YEARS': 8,
        'EXT_SOURCE_AVG': 0.6,
        'PAYMENT_AVERAGE_MEAN': 0.95
    }
    
    # Note: This would require a trained model to work properly
    print("Recommendation engine initialized. Load a trained model to make predictions.")


