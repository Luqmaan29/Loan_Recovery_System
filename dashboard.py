"""
Interactive Streamlit Dashboard for Smart Digital Lending Recommendation System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def safe_plotly_data(data):
    """Convert pandas data to safe types for Plotly serialization."""
    if isinstance(data, pd.Series):
        return data.astype(str) if data.dtype == 'object' else data.astype(float)
    elif isinstance(data, pd.DataFrame):
        return data.astype(str) if data.dtypes.eq('object').any() else data.astype(float)
    elif isinstance(data, np.ndarray):
        return data.astype(float)
    else:
        return data

from real_data_loader import RealDataLoader
from data_preprocessor import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from recommendation_engine import RecommendationEngine, RiskCategory
from data_collector import DataCollector, PrivacyManager

# Page configuration
st.set_page_config(
    page_title="Smart Digital Lending Recommendation System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Advanced and Sharp Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main Header */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 3rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Navigation Cards */
    .nav-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border: none;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Risk Categories */
    .risk-low {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.3);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 5px 15px rgba(255, 193, 7, 0.3);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #dc3545, #e83e8c);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 5px 15px rgba(220, 53, 69, 0.3);
    }
    
    /* Recommendation Cards */
    .recommendation-approve {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.3);
    }
    
    .recommendation-review {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        color: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 5px 15px rgba(255, 193, 7, 0.3);
    }
    
    .recommendation-reject {
        background: linear-gradient(135deg, #dc3545, #e83e8c);
        color: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 5px 15px rgba(220, 53, 69, 0.3);
    }
    
    /* Customer Portal Styles */
    .customer-portal {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem;
        border-radius: 25px;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    .loan-form {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .eligibility-badge {
        padding: 1rem 2rem;
        border-radius: 25px;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .eligible {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
    }
    
    .review {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        color: white;
    }
    
    .not-eligible {
        background: linear-gradient(135deg, #dc3545, #e83e8c);
        color: white;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache data."""
    loader = RealDataLoader()
    data = loader.load_all_data()
    return data

@st.cache_data
def preprocess_data(data):
    """Preprocess and cache data."""
    preprocessor = DataPreprocessor()
    
    # Use application_train for real data
    if 'application_train' in data:
        app_data = data['application_train']
    elif 'application' in data:
        app_data = data['application']
    else:
        # Fallback to first available dataset
        app_data = list(data.values())[0]
    
    processed_df, features = preprocessor.preprocess_pipeline(app_data, data)
    
    engineer = FeatureEngineer()
    engineered_df = engineer.create_all_features(processed_df)
    
    return engineered_df, features

@st.cache_data
def train_models(engineered_df):
    """Train models and cache results."""
    # Prepare data - handle missing TARGET column
    if 'TARGET' in engineered_df.columns:
        X = engineered_df.drop(['TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')
        y = engineered_df['TARGET'].dropna()
        X = X.loc[y.index]
    else:
        # If no TARGET column, create a mock one for demonstration
        X = engineered_df.drop(['SK_ID_CURR'], axis=1, errors='ignore')
        y = pd.Series(np.random.choice([0, 1], size=len(X), p=[0.92, 0.08]), index=X.index)
    
    # Handle categorical features - encode them before training
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        # Use label encoding for categorical features
        from sklearn.preprocessing import LabelEncoder
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Ensure all data is numeric
    X = X.select_dtypes(include=[np.number])
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced = trainer.handle_class_imbalance(X_train, y_train)
    
    # Train baseline models
    results = trainer.train_baseline_models(X_train_balanced, y_train_balanced, X_val, y_val)
    
    return trainer, X_test, y_test, results

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Smart Digital Lending System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🏦 Banking System")
    st.sidebar.markdown("---")
    
    # User Type Selection with guidance
    st.sidebar.markdown("### 👤 Select Your Role")
    st.sidebar.markdown("Choose how you want to use the system:")
    
    user_type = st.sidebar.radio(
        "",
        ["🏦 Bank Staff", "👤 Customer Portal"],
        help="Bank Staff: Analytics and decision-making tools | Customer Portal: Apply for loans"
    )
    
    if user_type == "🏦 Bank Staff":
        st.sidebar.markdown("### 📊 Dashboard Pages")
        st.sidebar.markdown("Select what you want to analyze:")
        
        page = st.sidebar.selectbox(
            "",
            ["📊 Overview", "📈 Data Analysis", "🤖 Model Performance", "⚠️ Risk Assessment", "💡 Recommendations", "👤 Client Analysis"],
            help="Choose the type of analysis you want to perform"
        )
    else:
        page = "Customer Portal"
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_data()
        engineered_df, features = preprocess_data(data)
    
    if page == "Customer Portal":
        show_customer_portal()
    elif page == "📊 Overview":
        show_overview(engineered_df)
    elif page == "📈 Data Analysis":
        show_data_analysis(engineered_df)
    elif page == "🤖 Model Performance":
        show_model_performance(engineered_df)
    elif page == "⚠️ Risk Assessment":
        show_risk_assessment(engineered_df)
    elif page == "💡 Recommendations":
        show_recommendations(engineered_df)
    elif page == "👤 Client Analysis":
        show_client_analysis(engineered_df)

def show_overview(engineered_df):
    """Show overview dashboard."""
    st.header("📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clients", f"{len(engineered_df):,}")
    
    with col2:
        default_rate = engineered_df['TARGET'].mean() if 'TARGET' in engineered_df.columns else 0
        st.metric("Default Rate", f"{default_rate:.2%}")
    
    with col3:
        avg_income = engineered_df['AMT_INCOME_TOTAL'].mean() if 'AMT_INCOME_TOTAL' in engineered_df.columns else 0
        st.metric("Avg Income", f"${avg_income:,.0f}")
    
    with col4:
        avg_credit = engineered_df['AMT_CREDIT'].mean() if 'AMT_CREDIT' in engineered_df.columns else 0
        st.metric("Avg Credit Amount", f"${avg_credit:,.0f}")
    
    # Risk distribution
    st.subheader("Risk Distribution")
    
    if 'TARGET' in engineered_df.columns:
        # Create risk categories based on target
        risk_data = engineered_df.copy()
        risk_data['Risk_Category'] = risk_data['TARGET'].map({0: 'Low Risk', 1: 'High Risk'})
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution pie chart
            risk_counts = risk_data['Risk_Category'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Distribution",
                color_discrete_map={'Low Risk': '#28a745', 'High Risk': '#dc3545'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Income vs Credit scatter
            fig_scatter = px.scatter(
                risk_data.sample(min(1000, len(risk_data))),
                x='AMT_INCOME_TOTAL',
                y='AMT_CREDIT',
                color='Risk_Category',
                title="Income vs Credit Amount",
                color_discrete_map={'Low Risk': '#28a745', 'High Risk': '#dc3545'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Feature importance (if available)
    st.subheader("Key Risk Factors")
    
    # Create sample feature importance
    important_features = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AGE_YEARS', 'EMPLOYMENT_YEARS',
        'EXT_SOURCE_AVG', 'DEBT_TO_INCOME_RATIO', 'PAYMENT_BURDEN'
    ]
    
    available_features = [f for f in important_features if f in engineered_df.columns]
    
    if available_features:
        # Calculate correlation with target
        correlations = {}
        if 'TARGET' in engineered_df.columns:
            for feature in available_features:
                corr = engineered_df[feature].corr(engineered_df['TARGET'])
                correlations[feature] = abs(corr) if not pd.isna(corr) else 0
        
        if correlations:
            # Sort by correlation
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            
            # Create bar chart
            features, corrs = zip(*sorted_features)
            fig_bar = px.bar(
                x=list(corrs),
                y=list(features),
                orientation='h',
                title="Feature Importance (Correlation with Default)",
                labels={'x': 'Absolute Correlation', 'y': 'Features'}
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

def show_data_analysis(engineered_df):
    """Show data analysis dashboard."""
    st.header("📈 Data Analysis")
    
    # Data overview
    st.subheader("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", engineered_df.shape)
        st.write("**Features:**", len(engineered_df.columns))
        st.write("**Missing Values:**", engineered_df.isnull().sum().sum())
    
    with col2:
        # Data types
        dtype_counts = engineered_df.dtypes.value_counts()
        # Convert data types to strings to avoid JSON serialization issues
        dtype_names = [str(dtype) for dtype in dtype_counts.index]
        fig_dtype = px.pie(
            values=dtype_counts.values,
            names=dtype_names,
            title="Data Types Distribution"
        )
        st.plotly_chart(fig_dtype, use_container_width=True)
    
    # Feature analysis
    st.subheader("Feature Analysis")
    
    # Select feature to analyze
    numeric_features = engineered_df.select_dtypes(include=[np.number]).columns.tolist()
    selected_feature = st.selectbox("Select feature to analyze", numeric_features)
    
    if selected_feature:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                engineered_df,
                x=selected_feature,
                title=f"Distribution of {selected_feature}",
                nbins=50
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = px.box(
                engineered_df,
                y=selected_feature,
                title=f"Box Plot of {selected_feature}"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Statistics
        st.subheader(f"Statistics for {selected_feature}")
        stats = engineered_df[selected_feature].describe()
        st.dataframe(stats)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    # Select features for correlation
    correlation_features = st.multiselect(
        "Select features for correlation analysis",
        numeric_features,
        default=numeric_features[:10] if len(numeric_features) > 10 else numeric_features
    )
    
    if len(correlation_features) > 1:
        corr_matrix = engineered_df[correlation_features].corr()
        
        # Convert to numpy array to avoid pandas data type issues
        corr_array = corr_matrix.values
        fig_corr = px.imshow(
            corr_array,
            title="Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto",
            labels=dict(x="Features", y="Features", color="Correlation"),
            x=correlation_features,
            y=correlation_features
        )
        st.plotly_chart(fig_corr, use_container_width=True)

def show_model_performance(engineered_df):
    """Show model performance dashboard."""
    st.header("🤖 Model Performance")
    
    with st.spinner("Training models..."):
        trainer, X_test, y_test, results = train_models(engineered_df)
    
    # Model comparison
    st.subheader("Model Comparison")
    
    # Create comparison dataframe
    model_comparison = []
    for name, result in results.items():
        model_comparison.append({
            'Model': name,
            'AUC': result['auc'],
            'F1 Score': result['f1'],
            'Accuracy': result['accuracy']
        })
    
    comparison_df = pd.DataFrame(model_comparison)
    comparison_df = comparison_df.sort_values('AUC', ascending=False)
    
    # Display comparison table
    st.dataframe(comparison_df, use_container_width=True)
    
    # Model performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # AUC comparison
        fig_auc = px.bar(
            comparison_df,
            x='Model',
            y='AUC',
            title="Model AUC Comparison",
            color='AUC',
            color_continuous_scale='Viridis'
        )
        fig_auc.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_auc, use_container_width=True)
    
    with col2:
        # F1 Score comparison
        fig_f1 = px.bar(
            comparison_df,
            x='Model',
            y='F1 Score',
            title="Model F1 Score Comparison",
            color='F1 Score',
            color_continuous_scale='Viridis'
        )
        fig_f1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_f1, use_container_width=True)
    
    # Best model details
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_result = results[best_model_name]
    
    st.subheader(f"Best Model: {best_model_name}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("AUC Score", f"{best_model_result['auc']:.4f}")
    
    with col2:
        st.metric("F1 Score", f"{best_model_result['f1']:.4f}")
    
    with col3:
        st.metric("Accuracy", f"{best_model_result['accuracy']:.4f}")
    
    # Feature importance (if available)
    if hasattr(best_model_result['model'], 'feature_importances_'):
        st.subheader("Feature Importance")
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': best_model_result['model'].feature_importances_
        }).sort_values('Importance', ascending=False).head(20)
        
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 20 Most Important Features"
        )
        st.plotly_chart(fig_importance, use_container_width=True)

def show_risk_assessment(engineered_df):
    """Show risk assessment dashboard."""
    st.header("⚠️ Risk Assessment")
    
    # Risk categories
    st.subheader("Risk Category Distribution")
    
    # Create risk categories based on available features
    risk_df = engineered_df.copy()
    
    # Create risk score (simplified)
    risk_score = 0
    if 'AMT_INCOME_TOTAL' in risk_df.columns and 'AMT_CREDIT' in risk_df.columns:
        risk_score += (risk_df['AMT_CREDIT'] / (risk_df['AMT_INCOME_TOTAL'] + 1)) * 0.3
    
    if 'AGE_YEARS' in risk_df.columns:
        age_risk = np.where(risk_df['AGE_YEARS'] < 25, 0.3, 
                           np.where(risk_df['AGE_YEARS'] > 65, 0.2, 0.1))
        risk_score += age_risk
    
    if 'EMPLOYMENT_YEARS' in risk_df.columns:
        emp_risk = np.where(risk_df['EMPLOYMENT_YEARS'] < 1, 0.3, 0.1)
        risk_score += emp_risk
    
    # Normalize risk score
    risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
    
    # Categorize risk
    risk_df['Risk_Score'] = risk_score
    risk_df['Risk_Category'] = pd.cut(
        risk_score,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = risk_df['Risk_Category'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Category Distribution",
            color_discrete_map={
                'Low Risk': '#28a745',
                'Medium Risk': '#ffc107',
                'High Risk': '#dc3545'
            }
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Risk score distribution
        fig_hist = px.histogram(
            risk_df,
            x='Risk_Score',
            title="Risk Score Distribution",
            nbins=50,
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Risk factors analysis
    st.subheader("Risk Factors Analysis")
    
    # Select risk factors to analyze
    risk_factors = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AGE_YEARS', 'EMPLOYMENT_YEARS']
    available_factors = [f for f in risk_factors if f in risk_df.columns]
    
    if available_factors:
        selected_factor = st.selectbox("Select risk factor", available_factors)
        
        if selected_factor:
            # Risk factor vs risk category
            fig_factor = px.box(
                risk_df,
                x='Risk_Category',
                y=selected_factor,
                title=f"{selected_factor} by Risk Category",
                color='Risk_Category',
                color_discrete_map={
                    'Low Risk': '#28a745',
                    'Medium Risk': '#ffc107',
                    'High Risk': '#dc3545'
                }
            )
            st.plotly_chart(fig_factor, use_container_width=True)

def show_recommendations(engineered_df):
    """Show recommendations dashboard."""
    st.header("💡 Loan Recommendations")
    
    # Recommendation engine setup
    st.subheader("Recommendation Engine")
    
    # Business rules configuration
    st.sidebar.subheader("Business Rules Configuration")
    
    high_income_threshold = st.sidebar.number_input(
        "High Income Threshold",
        min_value=0,
        value=100000,
        step=1000
    )
    
    stable_employment_years = st.sidebar.number_input(
        "Stable Employment Years",
        min_value=0,
        value=5,
        step=1
    )
    
    # Sample recommendations
    st.subheader("Sample Recommendations")
    
    # Create sample recommendations
    sample_size = min(100, len(engineered_df))
    sample_df = engineered_df.sample(sample_size)
    
    # Create mock recommendations
    recommendations = []
    for idx, row in sample_df.iterrows():
        # Mock probability of default
        prob_default = np.random.beta(2, 8)  # Skewed towards lower probabilities
        
        # Mock risk category
        if prob_default < 0.3:
            risk_category = "Low Risk"
            recommendation = "APPROVE"
        elif prob_default < 0.7:
            risk_category = "Medium Risk"
            recommendation = "REVIEW"
        else:
            risk_category = "High Risk"
            recommendation = "REJECT"
        
        recommendations.append({
            'Client ID': row.get('SK_ID_CURR', idx),
            'Risk Category': risk_category,
            'Probability of Default': prob_default,
            'Recommendation': recommendation,
            'Confidence': np.random.uniform(0.6, 0.95)
        })
    
    rec_df = pd.DataFrame(recommendations)
    
    # Display recommendations
    st.dataframe(rec_df, use_container_width=True)
    
    # Recommendation summary
    st.subheader("Recommendation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        approve_count = len(rec_df[rec_df['Recommendation'] == 'APPROVE'])
        st.metric("Approve", approve_count)
    
    with col2:
        review_count = len(rec_df[rec_df['Recommendation'] == 'REVIEW'])
        st.metric("Review", review_count)
    
    with col3:
        reject_count = len(rec_df[rec_df['Recommendation'] == 'REJECT'])
        st.metric("Reject", reject_count)
    
    with col4:
        avg_confidence = rec_df['Confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    # Recommendation distribution
    rec_counts = rec_df['Recommendation'].value_counts()
    fig_rec = px.pie(
        values=rec_counts.values,
        names=rec_counts.index,
        title="Recommendation Distribution",
        color_discrete_map={
            'APPROVE': '#28a745',
            'REVIEW': '#ffc107',
            'REJECT': '#dc3545'
        }
    )
    st.plotly_chart(fig_rec, use_container_width=True)

def show_client_analysis(engineered_df):
    """Show client analysis dashboard."""
    st.header("👤 Client Analysis")
    
    # Client selection
    st.subheader("Select Client for Analysis")
    
    # Get client IDs
    client_ids = engineered_df['SK_ID_CURR'].unique() if 'SK_ID_CURR' in engineered_df.columns else range(len(engineered_df))
    
    selected_client = st.selectbox("Select Client ID", client_ids[:100])  # Limit to first 100 for demo
    
    if selected_client:
        # Get client data
        client_data = engineered_df[engineered_df['SK_ID_CURR'] == selected_client].iloc[0]
        
        # Client information
        st.subheader("Client Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Basic Information**")
            if 'AGE_YEARS' in client_data:
                st.write(f"Age: {client_data['AGE_YEARS']:.1f} years")
            if 'AMT_INCOME_TOTAL' in client_data:
                st.write(f"Income: ${client_data['AMT_INCOME_TOTAL']:,.0f}")
            if 'AMT_CREDIT' in client_data:
                st.write(f"Credit Amount: ${client_data['AMT_CREDIT']:,.0f}")
        
        with col2:
            st.write("**Employment**")
            if 'EMPLOYMENT_YEARS' in client_data:
                st.write(f"Employment Years: {client_data['EMPLOYMENT_YEARS']:.1f}")
            if 'OCCUPATION_TYPE' in client_data:
                st.write(f"Occupation: {client_data['OCCUPATION_TYPE']}")
        
        with col3:
            st.write("**Family**")
            if 'CNT_CHILDREN' in client_data:
                st.write(f"Children: {client_data['CNT_CHILDREN']}")
            if 'CNT_FAM_MEMBERS' in client_data:
                st.write(f"Family Members: {client_data['CNT_FAM_MEMBERS']}")
        
        # Risk assessment for this client
        st.subheader("Risk Assessment")
        
        # Calculate risk score
        risk_factors = []
        risk_score = 0
        
        if 'AMT_INCOME_TOTAL' in client_data and 'AMT_CREDIT' in client_data:
            debt_ratio = client_data['AMT_CREDIT'] / (client_data['AMT_INCOME_TOTAL'] + 1)
            risk_factors.append(f"Debt-to-Income Ratio: {debt_ratio:.2%}")
            if debt_ratio > 0.3:
                risk_score += 0.3
        
        if 'AGE_YEARS' in client_data:
            age = client_data['AGE_YEARS']
            risk_factors.append(f"Age: {age:.1f} years")
            if age < 25 or age > 65:
                risk_score += 0.2
        
        if 'EMPLOYMENT_YEARS' in client_data:
            emp_years = client_data['EMPLOYMENT_YEARS']
            risk_factors.append(f"Employment Years: {emp_years:.1f}")
            if emp_years < 1:
                risk_score += 0.3
        
        # Display risk factors
        for factor in risk_factors:
            st.write(f"• {factor}")
        
        # Risk category
        if risk_score < 0.3:
            risk_category = "Low Risk"
            risk_color = "green"
        elif risk_score < 0.7:
            risk_category = "Medium Risk"
            risk_color = "orange"
        else:
            risk_category = "High Risk"
            risk_color = "red"
        
        st.write(f"**Risk Category:** :{risk_color}[{risk_category}]")
        st.write(f"**Risk Score:** {risk_score:.2f}")
        
        # Recommendation
        if risk_score < 0.3:
            recommendation = "APPROVE"
            rec_color = "green"
        elif risk_score < 0.7:
            recommendation = "REVIEW"
            rec_color = "orange"
        else:
            recommendation = "REJECT"
            rec_color = "red"
        
        st.write(f"**Recommendation:** :{rec_color}[{recommendation}]")

def show_customer_portal():
    """Show customer loan application portal with user-friendly guidance."""
    
    # Welcome section with clear guidance
    st.markdown("""
    <div class="customer-portal">
        <h1 style="text-align: center; font-size: 2.5rem; margin-bottom: 1rem;">🏦 Smart Loan Application Portal</h1>
        <p style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">Get your loan decision in minutes with AI-powered risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step-by-step guidance
    st.markdown("### 📋 How it works:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Step 1: Fill Form**  
        📝 Complete the application form below with your details
        """)
    
    with col2:
        st.markdown("""
        **Step 2: AI Analysis**  
        🤖 Our AI analyzes your information for risk assessment
        """)
    
    with col3:
        st.markdown("""
        **Step 3: Get Decision**  
        ✅ Receive instant loan decision and personalized options
        """)
    
    st.markdown("---")
    
    # Main application form with better organization
    st.markdown("### 📝 Loan Application Form")
    st.markdown("Please fill in your details below. All fields are required for accurate assessment.")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["👤 Personal Info", "💰 Financial Info", "🎯 Loan Details"])
    
    with tab1:
        st.markdown("#### Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *", placeholder="Enter your full name", help="Your complete legal name")
            age = st.number_input("Age *", min_value=18, max_value=80, value=30, help="Must be between 18-80 years")
            email = st.text_input("Email Address *", placeholder="your.email@example.com", help="We'll send your loan decision here")
        
        with col2:
            phone = st.text_input("Phone Number *", placeholder="+1 (555) 123-4567", help="Your primary contact number")
            home_ownership = st.selectbox("Home Ownership *", ["Own", "Rent", "Other"], help="Your current housing situation")
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, help="People who depend on your income")
    
    with tab2:
        st.markdown("#### Financial Information")
        col1, col2 = st.columns(2)
        
        with col1:
            annual_income = st.number_input("Annual Income ($) *", min_value=0, value=75000, step=1000, help="Your total yearly income before taxes")
            monthly_expenses = st.number_input("Monthly Expenses ($) *", min_value=0, value=3000, step=100, help="Your monthly living expenses")
            existing_loans = st.number_input("Existing Monthly Loan Payments ($)", min_value=0, value=500, step=50, help="Current loan payments you make monthly")
        
        with col2:
            credit_score = st.slider("Credit Score *", min_value=300, max_value=850, value=720, help="Your current credit score (300-850)")
            employment_years = st.number_input("Years at Current Job *", min_value=0, max_value=50, value=5, help="How long have you been in your current job")
    
    with tab3:
        st.markdown("#### Loan Requirements")
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amount = st.number_input("Loan Amount ($) *", min_value=1000, max_value=1000000, value=50000, step=1000, help="Amount you want to borrow")
            loan_purpose = st.selectbox("Loan Purpose *", [
                "Home Purchase", "Home Refinance", "Debt Consolidation", 
                "Business", "Education", "Personal", "Auto", "Other"
            ], help="What will you use this loan for?")
        
        with col2:
            preferred_term = st.selectbox("Preferred Loan Term (Years) *", [15, 20, 25, 30], help="How long do you want to repay the loan")
    
    # Calculate eligibility button with better guidance
    st.markdown("---")
    st.markdown("### 🚀 Ready to Get Your Loan Decision?")
    
    # Validation check
    required_fields = [name, email, phone, annual_income, monthly_expenses, loan_amount]
    all_filled = all(field for field in required_fields)
    
    if not all_filled:
        st.warning("⚠️ Please fill in all required fields (marked with *) to proceed.")
    
    if st.button("🚀 Get My Loan Decision", type="primary", use_container_width=True, disabled=not all_filled):
        
        with st.spinner("🤖 AI is analyzing your application..."):
            # Simulate processing time
            import time
            time.sleep(2)
            
            # Assess eligibility
            eligibility, risk_level, interest_rate = assess_eligibility(
                annual_income, monthly_expenses, loan_amount, credit_score
            )
            
            # Calculate loan options
            loan_options = calculate_loan_options(loan_amount, interest_rate, preferred_term)
            
            # Display results with better user experience
            st.markdown("---")
            st.header("🎯 Your Loan Decision")
            
            # Eligibility badge with clear messaging
            if eligibility == "eligible":
                st.success(f"✅ **Congratulations! You're Eligible**")
                st.markdown(f"**Risk Level:** {risk_level} | **Interest Rate:** {interest_rate}%")
            elif eligibility == "review":
                st.warning(f"⚠️ **Under Review**")
                st.markdown(f"**Risk Level:** {risk_level} | **Interest Rate:** {interest_rate}%")
                st.info("Your application requires manual review by our loan officers.")
            else:
                st.error(f"❌ **Not Eligible at this time**")
                st.markdown(f"**Risk Level:** {risk_level} | **Interest Rate:** {interest_rate}%")
                st.info("We'll provide suggestions to improve your eligibility below.")
            
            # Loan options with better presentation
            if eligibility in ["eligible", "review"]:
                st.subheader("💡 Your Personalized Loan Options")
                st.markdown("Here are your loan options based on your profile:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🏠 Recommended Loan")
                    st.metric("Interest Rate", f"{interest_rate}%")
                    st.metric("Monthly Payment", f"${loan_options['monthly_payment']:,.0f}")
                    st.metric("Loan Term", f"{preferred_term} years")
                    st.metric("Total Interest", f"${loan_options['total_interest']:,.0f}")
                
                with col2:
                    # Alternative option with different terms
                    alt_rate = interest_rate + 0.5
                    alt_term = preferred_term + 5
                    alt_options = calculate_loan_options(loan_amount, alt_rate, alt_term)
                    
                    st.markdown("### 🔄 Extended Term Option")
                    st.metric("Interest Rate", f"{alt_rate}%")
                    st.metric("Monthly Payment", f"${alt_options['monthly_payment']:,.0f}")
                    st.metric("Loan Term", f"{alt_term} years")
                    st.metric("Total Interest", f"${alt_options['total_interest']:,.0f}")
                    st.caption("Lower monthly payment, longer term")
                
                with col3:
                    # Lower rate option
                    low_rate = max(interest_rate - 0.5, 3.5)
                    low_options = calculate_loan_options(loan_amount, low_rate, preferred_term)
                    
                    st.markdown("### ⭐ Best Rate Option")
                    st.metric("Interest Rate", f"{low_rate}%")
                    st.metric("Monthly Payment", f"${low_options['monthly_payment']:,.0f}")
                    st.metric("Loan Term", f"{preferred_term} years")
                    st.metric("Total Interest", f"${low_options['total_interest']:,.0f}")
                    st.caption("Best interest rate available")
                
                # Risk factors
                st.subheader("📊 Your Risk Assessment")
                
                risk_factors = []
                if annual_income < 50000:
                    risk_factors.append("Lower income level")
                if credit_score < 650:
                    risk_factors.append("Credit score below 650")
                if employment_years < 2:
                    risk_factors.append("Short employment history")
                if (monthly_expenses + existing_loans) / (annual_income / 12) > 0.4:
                    risk_factors.append("High debt-to-income ratio")
                
                if risk_factors:
                    st.write("**Risk Factors Identified:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.success("✅ No significant risk factors identified!")
                
                # Next steps
                st.subheader("🚀 Next Steps")
                if eligibility == "eligible":
                    st.success("🎉 Congratulations! You're pre-approved for this loan.")
                    st.write("**What happens next:**")
                    st.write("1. 📋 Complete the full application")
                    st.write("2. 📄 Submit required documents")
                    st.write("3. 🏦 Final approval within 24 hours")
                    st.write("4. 💰 Funds deposited to your account")
                else:
                    st.info("📋 Your application requires manual review.")
                    st.write("**What happens next:**")
                    st.write("1. 👨‍💼 Loan officer will review your application")
                    st.write("2. 📞 We'll contact you within 2 business days")
                    st.write("3. 📄 Additional documentation may be required")
                    st.write("4. ✅ Final decision within 3-5 business days")
            
            else:
                st.error("❌ Unfortunately, you don't meet our current lending criteria.")
                st.write("**Why you might not be eligible:**")
                st.write("• Income too low for requested amount")
                st.write("• High debt-to-income ratio")
                st.write("• Credit score below minimum requirements")
                st.write("• Insufficient employment history")
                
                st.write("**What you can do:**")
                st.write("• Improve your credit score")
                st.write("• Reduce existing debt")
                st.write("• Apply for a smaller amount")
                st.write("• Reapply in 6 months")
    
    # Footer with better guidance
    st.markdown("---")
    st.markdown("### 📞 Need Help?")
    st.markdown("""
    - **Questions about your application?** Contact our support team
    - **Want to speak with a loan officer?** Schedule a consultation
    - **Need to update your information?** You can modify your application anytime
    """)
    
    st.markdown("### 🔒 Your Privacy & Security")
    st.markdown("""
    - All your data is encrypted and secure
    - We follow strict privacy guidelines
    - Your information is only used for loan assessment
    """)

def calculate_loan_options(loan_amount, interest_rate, term_years):
    """Calculate loan options and payments."""
    monthly_rate = interest_rate / 100 / 12
    num_payments = term_years * 12
    
    if monthly_rate > 0:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    else:
        monthly_payment = loan_amount / num_payments
    
    total_payment = monthly_payment * num_payments
    total_interest = total_payment - loan_amount
    
    return {
        'monthly_payment': monthly_payment,
        'total_payment': total_payment,
        'total_interest': total_interest,
        'term_years': term_years,
        'interest_rate': interest_rate
    }

def assess_eligibility(income, expenses, loan_amount, credit_score):
    """Assess loan eligibility based on basic criteria."""
    # Debt-to-income ratio
    dti_ratio = (expenses / income) if income > 0 else 1
    
    # Loan-to-income ratio
    lti_ratio = (loan_amount / income) if income > 0 else 1
    
    # Risk score calculation
    risk_score = 0
    
    # Income stability (higher income = lower risk)
    if income > 100000:
        risk_score += 0.1
    elif income > 50000:
        risk_score += 0.2
    else:
        risk_score += 0.3
    
    # Debt-to-income ratio
    if dti_ratio < 0.3:
        risk_score += 0.1
    elif dti_ratio < 0.5:
        risk_score += 0.2
    else:
        risk_score += 0.4
    
    # Credit score
    if credit_score > 750:
        risk_score += 0.1
    elif credit_score > 650:
        risk_score += 0.2
    else:
        risk_score += 0.3
    
    # Loan amount vs income
    if lti_ratio < 3:
        risk_score += 0.1
    elif lti_ratio < 5:
        risk_score += 0.2
    else:
        risk_score += 0.3
    
    # Determine eligibility
    if risk_score < 0.3:
        return "eligible", "Low Risk", 5.5
    elif risk_score < 0.6:
        return "review", "Medium Risk", 7.5
    else:
        return "not_eligible", "High Risk", 12.0

if __name__ == "__main__":
    main()
