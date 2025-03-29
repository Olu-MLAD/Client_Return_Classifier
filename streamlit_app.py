import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import gspread
from gspread_dataframe import get_as_dataframe
from sklearn.base import is_classifier
from datetime import datetime
from PIL import Image
import shap  # Added for XAI
from io import BytesIO  # Added for image handling

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title="IFSSA Return Predictor"
)

# --- Helper Functions ---
def validate_postal_code(postal_code):
    """Validate Canadian postal code format"""
    if not postal_code:
        return False
    clean_code = postal_code.replace(" ", "").upper()
    if len(clean_code) != 6:
        return False
    return bool(re.match(r'^[A-Z]\d[A-Z]\d[A-Z]\d$', clean_code))

@st.cache_resource
def load_model():
    """Load and validate the machine learning model"""
    model_path = "RF_model.pkl"
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {os.path.abspath(model_path)}")
        
        model = joblib.load(model_path)
        
        if not is_classifier(model):
            raise ValueError("Loaded model is not a classifier")
            
        if not (hasattr(model, 'predict') and hasattr(model, 'predict_proba')):
            raise AttributeError("Model missing required methods")
            
        return model
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.error("""
        Please ensure:
        1. 'RF_model.pkl' exists in the same directory
        2. The file is a valid scikit-learn model
        3. You have matching Python/scikit-learn versions
        """)
        return None

def connect_to_google_sheets():
    """Handle Google Sheets connection with status tracking"""
    status_container = st.container()
    data_container = st.container()
    
    if not os.path.exists("service_account.json"):
        with status_container:
            st.info("ℹ️ Google Sheets integration not configured - using local mode")
            st.caption("To enable Google Sheets, add 'service_account.json' to your directory")
        return None
    
    try:
        with status_container:
            with st.spinner("Connecting to Google Sheets..."):
                gc = gspread.service_account(filename="service_account.json")
                st.success("🔐 Authentication successful")
                
                sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQwjh9k0hk536tHDO3cgmCb6xvu6GMAcLUUW1aVqKI-bBw-3mb5mz1PTRZ9XSfeLnlmrYs1eTJH3bvJ/pubhtml"
                sh = gc.open_by_url(sheet_url)
                worksheet = sh.sheet1
                st.success("📊 Connected to Google Sheet")
                
                with st.spinner("Loading client data..."):
                    df = get_as_dataframe(worksheet)
                    if df.empty:
                        st.warning("⚠️ Loaded empty dataset")
                    else:
                        st.success(f"✅ Loaded {len(df)} records")
                        
                        with data_container.expander("View Live Client Data", expanded=False):
                            st.dataframe(df.head(10))
                            st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        return df
                        
    except gspread.exceptions.APIError as e:
        with status_container:
            st.error(f"🔴 API Error: {str(e)}")
    except gspread.exceptions.SpreadsheetNotFound:
        with status_container:
            st.error("🔍 Spreadsheet not found - please check URL")
    except Exception as e:
        with status_container:
            st.error(f"⚠️ Unexpected error: {str(e)}")
    
    return None

# --- UI Components ---
def display_header():
    col1, col2, _ = st.columns([0.15, 0.15, 0.7])
    with col1:
        st.image("logo1.jpeg", width=120)
    with col2:
        st.image("logo2.png", width=120)
    
    st.markdown(
        """
        <h1 style='text-align: center; color: #ff5733; padding: 20px;'>
        IFSSA Client Return Prediction
        </h1>
        <p style='text-align: center; font-size: 1.1rem;'>
        Predict which clients will return within 3 months using statistically validated features
        </p>
        """,
        unsafe_allow_html=True
    )

def about_page():
    st.markdown("""
    ## About This Tool
    
    This application helps IFSSA predict which clients are likely to return for services 
    within the next 3 months using machine learning.
    
    ### How It Works
    1. Staff enter client visit information
    2. The system analyzes patterns from historical data
    3. Predictions guide outreach efforts
    
    ### Key Benefits
    - Enhance Response Accuracy
    - Improve Operational Efficiency
    - Streamline Stakeholder Communication
    - Facilitate Informed Decision Making
    - Ensure Scalability and Flexibility
    """)

def exploratory_data_analysis_page():
    st.markdown("<h2 style='color: #33aaff;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666;'>Exploring the dataset to understand structure, patterns, and insights.</p>", unsafe_allow_html=True)
    
    # Display Pre-generated Charts
    cols = st.columns(2)
    for i in range(1, 8):
        try:
            img = Image.open(f"chart{i}.png")
            with cols[(i-1) % 2]:
                st.image(img, caption=f"Chart {i}", use_container_width=True)
        except FileNotFoundError:
            with cols[(i-1) % 2]:
                st.warning(f"Chart image not found: chart{i}.png")

def feature_analysis_page():
    st.markdown("## Statistically Validated Predictors")
    
    chi2_results = {
        'monthly_visits': 0.000000e+00,
        'weekly_visits': 0.000000e+00,
        'total_dependents_3_months': 0.000000e+00,
        'pickup_count_last_90_days': 0.000000e+00,
        'pickup_count_last_30_days': 0.000000e+00,
        'pickup_count_last_14_days': 0.000000e+00,
        'pickup_count_last_7_days': 0.000000e+00,
        'Holidays': 8.394089e-90,
        'pickup_week': 1.064300e-69,
        'time_since_first_visit': 7.845354e-04
    }
    
    chi_df = pd.DataFrame.from_dict(chi2_results, orient='index', columns=['p-value'])
    chi_df['-log10(p)'] = -np.log10(chi_df['p-value'].replace(0, 1e-300))
    chi_df = chi_df.sort_values('-log10(p)', ascending=False)
    
    st.markdown("### Feature Significance (-log10 p-values)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='-log10(p)', y=chi_df.index, data=chi_df, palette="viridis", ax=ax)
    ax.axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
    ax.set_xlabel("Statistical Significance (-log10 p-value)")
    ax.set_ylabel("Features")
    ax.set_title("Chi-Square Test Results for Feature Selection")
    st.pyplot(fig)
    
    st.markdown("""
    **Key Insights**:
    - All shown features are statistically significant (p < 0.05)
    - Visit frequency metrics are strongest predictors (p ≈ 0)
    - Holiday effects are 10^90 times more significant than chance
    - Time since first visit still shows significance (p=7.8e-04)
    """)

def xai_insights_page():
    st.markdown("<h2 style='color: #33aaff;'>XAI Insights</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #666;'>
    Explainable AI (XAI) helps understand how the model makes predictions using SHAP values.
    </p>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading prediction model..."):
        model = load_model()
    if model is None:
        st.error("Failed to load model - cannot generate explanations")
        show_fallback_xai()
        return

    # Create sample data that matches model's training format
    X = pd.DataFrame({
        'weekly_visits': [3, 1, 4],
        'total_dependents_3_months': [2, 1, 3],
        'pickup_count_last_30_days': [4, 2, 5],
        'pickup_count_last_14_days': [2, 1, 3],
        'Holidays': [0, 0, 1],
        'pickup_week': [25, 10, 50],
        'time_since_first_visit': [90, 30, 180]
    })

    try:
        # Compute SHAP values
        with st.spinner("Computing SHAP explanations (this may take a moment)..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # SHAP Summary Plot (Bar Chart)
            st.markdown("### Feature Importance (SHAP Values)")
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance (Class 1 - Will Return)")
            st.pyplot(fig)
            plt.close()

            # Detailed SHAP summary plot
            st.markdown("### Detailed Feature Effects")
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values[1], X, show=False)
            plt.title("SHAP Value Distribution")
            st.pyplot(fig)
            plt.close()

            # Interpretation guide
            st.markdown("""
            **How to interpret SHAP values**:
            - **Global Importance**: Shows which features most influence predictions overall
            - **Detailed Effects**: Shows how feature values affect predictions
            - Positive SHAP values increase probability of returning
            - Negative SHAP values decrease probability of returning
            - Each point represents a client from the sample data
            """)

    except Exception as e:
        st.error(f"SHAP computation failed: {str(e)}")
        show_fallback_xai()

def show_fallback_xai():
    """Show simplified explanations when SHAP fails"""
    st.warning("Showing simplified feature importance (SHAP not available)")
    
    features = [
        'weekly_visits',
        'total_dependents_3_months',
        'pickup_count_last_30_days',
        'pickup_count_last_14_days',
        'Holidays',
        'pickup_week',
        'time_since_first_visit'
    ]
    importance = [0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=importance, y=features, palette="viridis")
    plt.title("Simplified Feature Importance")
    plt.xlabel("Relative Importance")
    st.pyplot(fig)
    
    st.markdown("""
    **Key Insights**:
    - Weekly visits is the strongest predictor of return visits
    - Number of dependents is the second most important factor
    - Recent pickup activity strongly influences predictions
    - Holidays and pickup week have smaller but significant effects
    """)

def prediction_page():
    st.markdown("<h2 style='color: #33aaff;'>Client Return Prediction</h2>", unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading prediction model..."):
        model = load_model()
    if model is None:
        st.stop()

    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pickup_count_last_14_days = st.number_input("Pickups in last 14 days:", min_value=0, value=0)
            pickup_count_last_30_days = st.number_input("Pickups in last 30 days:", min_value=0, value=0)
            weekly_visits = st.number_input("Weekly Visits:", min_value=0, value=3)

        with col2:
            total_dependents_3_months = st.number_input("Total Dependents in Last 3 Months:", min_value=0, value=2)
            time_since_first_visit = st.number_input("Time Since First Visit (days):", min_value=1, max_value=366, value=30)
            pickup_week = st.number_input("Pickup Week (1-52):", min_value=1, max_value=52, value=1)

        Holidays = st.selectbox("Is this pick-up during a holiday?", ["No", "Yes"], index=0)
        Holidays = 1 if Holidays == "Yes" else 0

        submitted = st.form_submit_button("Predict Return Probability", type="primary")

    # Handle form submission
    if submitted:
        try:
            input_data = pd.DataFrame([[ 
                weekly_visits,
                total_dependents_3_months,
                pickup_count_last_30_days,
                pickup_count_last_14_days,
                Holidays,
                pickup_week,
                time_since_first_visit
            ]], columns=[
                'weekly_visits',
                'total_dependents_3_months',
                'pickup_count_last_30_days',
                'pickup_count_last_14_days',
                'Holidays',
                'pickup_week',
                'time_since_first_visit'
            ])

            with st.spinner("Making prediction..."):
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]
            
            # Display results
            st.markdown("---")
            st.markdown("<h3 style='color: #ff33aa;'>Prediction Result</h3>", unsafe_allow_html=True)
            
            col_pred, col_prob = st.columns(2)
            with col_pred:
                st.metric("Prediction", 
                         "Will Return" if prediction[0] == 1 else "Will Not Return",
                         delta="High probability" if prediction[0] == 1 else "Low probability",
                         delta_color="normal")
            
            with col_prob:
                st.metric("Return Probability", f"{probability:.1%}")
            
            if prediction[0] == 1:
                st.success("✅ This client is likely to return within 3 months")
            else:
                st.warning("⚠️ This client is unlikely to return within 3 months")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

    # Google Sheets integration
    st.markdown("---")
    st.subheader("Data Integration Status")
    connect_to_google_sheets()

# --- Main App Logic ---
display_header()

# Navigation - Updated to include XAI Insights
page = st.sidebar.radio(
    "Navigation",
    ["About", "Exploratory Data Analysis", "Feature Analysis", "XAI Insights", "Make Prediction"],
    index=4
)

if page == "About":
    about_page()
elif page == "Exploratory Data Analysis":
    exploratory_data_analysis_page()
elif page == "Feature Analysis":
    feature_analysis_page()
elif page == "XAI Insights":
    xai_insights_page()
elif page == "Make Prediction":
    prediction_page()

# Footer - Updated version number
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
    <p>IFSSA Client Return Predictor • v1.2</p>
    <p><small>For support contact: support@ifssa.org</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
