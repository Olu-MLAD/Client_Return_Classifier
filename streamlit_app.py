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

def chat_with_rahim_page():
    st.markdown("<h2 style='color: #33aaff;'>Chat with Rahim</h2>", unsafe_allow_html=True)
    st.markdown("""
    Welcome to Chat with Rahim! This section allows you to ask questions about the IFSSA Client Return Prediction system, data insights, and machine learning.
    
    **How to Use:**
    - Type your question below.
    - Rahim will provide relevant insights based on available data and predictive analytics.
    """)
    
    try:
        user_input = st.text_input("Ask Rahim anything about IFSSA predictions:")
        
        if user_input:
            st.write("Rahim's Response:")
            
            # Simulate response based on user input. You can expand this with model logic.
            if "return prediction" in user_input.lower():
                st.success("I can help you understand how we predict if a client will return based on their profile and past behavior!")
            else:
                st.success(f"Great question! Here's what I found about: {user_input}")
    except Exception as e:
        st.error(f"❌ Something went wrong with the chat: {str(e)}")

# --- Pages ---
def about_page():
    st.markdown("<h2>About the IFSSA Client Return Predictor</h2>")
    st.markdown("""
    This application predicts whether a client will return based on past behavior and data insights. 
    The model uses a random forest classifier to make predictions and provide actionable insights.
    """)

def exploratory_data_analysis_page():
    st.markdown("<h2>Exploratory Data Analysis</h2>")
    # Placeholder for EDA visualizations and insights
    st.markdown("This section will showcase the exploratory data analysis of the IFSSA client data.")
    # Example plot
    df = pd.DataFrame(np.random.randn(100, 2), columns=["Feature 1", "Feature 2"])
    st.write(df)
    st.line_chart(df)

def xai_insights_page():
    st.markdown("<h2>XAI Insights</h2>")
    st.markdown("""
    Here we provide insights into the model's predictions using explainable AI techniques such as SHAP (SHapley Additive exPlanations).
    """)
    
    # Example: Displaying an image related to XAI
    image = Image.open("shap_example.png")  # Replace with your image path
    st.image(image, caption="Example SHAP visualization", use_column_width=True)
    
    # Adding more relevant images or plots for the XAI page
    st.markdown("### Feature Importance")
    fig, ax = plt.subplots()
    sns.barplot(x=[0.2, 0.5, 0.3], y=["Feature 1", "Feature 2", "Feature 3"], ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

def prediction_page():
    st.markdown("<h2>Make a Prediction</h2>")
    st.markdown("This section allows users to input client data and receive a prediction on whether they will return.")
    # Add input fields, model prediction, etc. here

# --- Main App Logic ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["About", "Exploratory Data Analysis", "XAI Insights", "Make Prediction", "Chat with Rahim"]
)

if page == "About":
    about_page()
elif page == "Exploratory Data Analysis":
    exploratory_data_analysis_page()
elif page == "XAI Insights":
    xai_insights_page()
elif page == "Make Prediction":
    prediction_page()
elif page == "Chat with Rahim":
    chat_with_rahim_page()

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
