import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
from gspread_dataframe import get_as_dataframe
from sklearn.base import is_classifier
from datetime import datetime
from PIL import Image
import shap
from io import BytesIO

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title="IFSSA Return Predictor"
)

# --- Helper Functions ---
@st.cache_resource
def load_model():
    """Load and validate the machine learning model"""
    model_path = "RF_model_streamlit.pkl"
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
        1. 'RF_model_streamlit.pkl' exists in the same directory
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
                
                sheet_url = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"
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
        <p style='text-align: center; font-size: 0.9rem; color: #666;'>
        <b>Model Output:</b> 1 = Will Return, 0 = Will Not Return
        </p>
        """,
        unsafe_allow_html=True
    )

def about_page():
    st.markdown("""
    ## About This Tool
    
    This application helps IFSSA predict which clients are likely to return for services 
    within the next 3 months using machine learning.
    
    ### Model Interpretation
    - **1**: Client will return within 3 months (probability ≥ 50%)
    - **0**: Client will not return within 3 months (probability < 50%)
    
    ### How It Works
    1. Staff enter client visit information
    2. The system analyzes patterns from historical data
    3. Predictions are made with clear 1/0 outputs
    4. Probability scores show confidence level
    
    ### Key Benefits
    - Clear binary output (1/0) with explanation
    - Probability scores for nuanced understanding
    - Feature importance explanations
    - Easy integration with existing systems
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
                st.image(img, caption=f"Chart {i}", use_column_width=True)
        except FileNotFoundError:
            with cols[(i-1) % 2]:
                st.warning(f"Chart image not found: chart{i}.png")

def xai_insights_page():
    st.markdown("<h2 style='color: #33aaff;'>XAI Insights</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #666;'>
    Explainable AI (XAI) helps understand how the model makes predictions using SHAP values.
    </p>
    <div style='background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <b>Model Output Key:</b><br>
    • <span style='color: green;'>1 = Will Return</span> (probability ≥ 50%)<br>
    • <span style='color: red;'>0 = Will Not Return</span> (probability < 50%)
    </div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading prediction model..."):
        model = load_model()
    if model is None:
        st.error("Failed to load model - cannot generate explanations")
        return

    # Create sample data with features in specified order
    X = pd.DataFrame({
        'pickup_week': [25, 10, 50],
        'pickup_count_last_14_days': [2, 1, 3],
        'pickup_count_last_30_days': [4, 2, 5],
        'pickup_count_last_90_days': [8, 3, 12],
        'time_since_first_visit': [90, 30, 180],
        'weekly_visits': [3, 1, 4]
    })

    try:
        # Compute SHAP values with correct settings
        with st.spinner("Computing SHAP explanations..."):
            explainer = shap.TreeExplainer(
                model,
                feature_perturbation="interventional",
                model_output="probability"
            )
            shap_values = explainer.shap_values(X, check_additivity=False)

            # SHAP Summary Plot (Bar Chart)
            st.markdown("### Feature Importance (SHAP Values)")
            fig, ax = plt.subplots(figsize=(12, 6))
            shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
            plt.title("Which Features Most Influence 'Will Return' Predictions?")
            st.pyplot(fig)
            plt.close()

            # Detailed SHAP summary plot
            st.markdown("### How Feature Values Affect Return Probability")
            fig, ax = plt.subplots(figsize=(12, 6))
            shap.summary_plot(shap_values[1], X, show=False)
            plt.title("Feature Value Impact on Return Probability (1=Return)")
            st.pyplot(fig)
            plt.close()

            st.markdown("""
            **Interpreting the Results**:
            - Features are ordered by their impact on predicting returns (1)
            - Right of center (positive SHAP values) = increases return probability
            - Left of center (negative SHAP values) = decreases return probability
            - Color shows feature value (red=high, blue=low)
            """)

    except Exception as e:
        st.error(f"Detailed explanation failed: {str(e)}")

def prediction_page():
    st.markdown("<h2 style='color: #33aaff;'>Client Return Prediction</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <b>Remember:</b><br>
    • <span style='color: green;'>1 = Will Return</span> (probability ≥ 50%)<br>
    • <span style='color: red;'>0 = Will Not Return</span> (probability < 50%)
    </div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading prediction model..."):
        model = load_model()
    if model is None:
        st.stop()

    # Input form with features in specified order
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pickup_week = st.number_input("Pickup Week (1-52):", min_value=1, max_value=52, value=1)
            pickup_count_last_14_days = st.number_input("Pickups in last 14 days:", min_value=0, value=0)
            pickup_count_last_30_days = st.number_input("Pickups in last 30 days:", min_value=0, value=0)
            
        with col2:
            pickup_count_last_90_days = st.number_input("Pickups in last 90 days:", min_value=0, value=0)
            time_since_first_visit = st.number_input("Time Since First Visit (days):", min_value=1, max_value=366, value=30)
            weekly_visits = st.number_input("Weekly Visits:", min_value=0, value=3)

        submitted = st.form_submit_button("Predict Return Probability", type="primary")

    # Handle form submission
    if submitted:
        try:
            input_data = pd.DataFrame([[ 
                pickup_week,
                pickup_count_last_14_days,
                pickup_count_last_30_days,
                pickup_count_last_90_days,
                time_since_first_visit,
                weekly_visits
            ]], columns=[
                'pickup_week',
                'pickup_count_last_14_days',
                'pickup_count_last_30_days',
                'pickup_count_last_90_days',
                'time_since_first_visit',
                'weekly_visits'
            ])

            with st.spinner("Making prediction..."):
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]
            
            # Display results
            st.markdown("---")
            st.markdown("<h3 style='color: #ff33aa;'>Prediction Result</h3>", unsafe_allow_html=True)
            
            col_pred, col_prob, col_expl = st.columns([1,1,2])
            with col_pred:
                st.metric("Binary Prediction", 
                         f"{prediction} - {'Will Return' if prediction == 1 else 'Will Not Return'}",
                         delta="Positive (1)" if prediction == 1 else "Negative (0)",
                         delta_color="normal")
            
            with col_prob:
                st.metric("Return Probability", 
                         f"{probability:.1%}",
                         delta="Confidence level")
            
            with col_expl:
                st.markdown("""
                **Interpretation**:
                - <span style='color: green;'>1 (Will Return)</span>: Probability ≥ 50%
                - <span style='color: red;'>0 (Will Not Return)</span>: Probability < 50%
                - Threshold can be adjusted for sensitivity
                """, unsafe_allow_html=True)
            
            # Visual indicator
            if prediction == 1:
                st.success("✅ This client is likely to return within 3 months (prediction = 1)")
            else:
                st.warning("⚠️ This client is unlikely to return within 3 months (prediction = 0)")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

    # Google Sheets integration
    st.markdown("---")
    st.subheader("Data Integration Status")
    connect_to_google_sheets()

# --- Main App Logic ---
display_header()

# Navigation
page = st.sidebar.radio(
    "Select a page:",
    ["About", "Exploratory Data Analysis", "XAI Insights", "Make Prediction"]
)

if page == "About":
    about_page()
elif page == "Exploratory Data Analysis":
    exploratory_data_analysis_page()
elif page == "XAI Insights":
    xai_insights_page()
elif page == "Make Prediction":
    prediction_page()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
    <p>IFSSA Client Return Predictor • v1.8</p>
    <p><small>Model outputs: 1 = Return, 0 = No Return | For support contact: support@ifssa.org</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
