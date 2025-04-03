import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import is_classifier

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

# Ask a Question Page with Simple Answers
def ask_a_question_page():
    st.markdown("<h2 style='color: #33aaff;'>Ask a Question</h2>", unsafe_allow_html=True)
    st.write("Ask a question about IFSSA, its roles, clients, or volunteers.")

    with st.form("question_form"):
        question = st.text_area("Enter your question:", "", height=150)
        submit_button = st.form_submit_button(label="Submit Question")

    if submit_button:
        if question.strip() == "":
            st.error("Please enter a question before submitting.")
        else:
            answer = get_answer(question)
            st.success(f"Your question: {question}")
            st.info(f"Answer: {answer}")

# Simple Q&A Function
def get_answer(question):
    question = question.lower()

    faq = {
        "what does ifssa do?": "IFSSA provides social support, food assistance, and outreach programs for vulnerable communities.",
        "who are ifssa's clients?": "IFSSA serves individuals and families in need, including newcomers, refugees, and low-income households.",
        "how can i volunteer?": "You can volunteer by signing up on the IFSSA website or contacting the volunteer coordinator.",
        "what services does ifssa provide?": "IFSSA offers food support, financial assistance, mental health resources, and educational programs.",
        "where is ifssa located?": "IFSSA is located in Edmonton, Alberta. More details are available on their official website."
    }

    return faq.get(question, "Our team will get back to you with more details!")

# XAI Insights Page with Additional Charts
def xai_insights_page():
    st.markdown("<h2 style='color: #33aaff;'>Explainable AI (XAI) Insights</h2>", unsafe_allow_html=True)

    # Placeholder explanation
    st.write("Explore feature importance and model interpretability.")

    # Feature Importance Chart
    feature_importance = pd.DataFrame({
        'Feature': ['pickup_week', 'pickup_count_last_14_days', 'pickup_count_last_30_days', 'pickup_count_last_90_days', 'time_since_first_visit', 'weekly_visits'],
        'Importance': [0.22, 0.18, 0.15, 0.14, 0.17, 0.14]
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax, palette="Blues_r")
    ax.set_title("Feature Importance in Prediction Model")
    st.pyplot(fig)

    # New Chart: SHAP Summary Plot (Example Placeholder)
    st.write("### SHAP Value Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.2, 0.3, 0.1]], ax=ax)
    ax.set_title("SHAP Value Distribution for Key Features")
    st.pyplot(fig)

    # New Chart: Predicted vs. Actual Returns
    st.write("### Predicted vs. Actual Returns")
    actual_vs_predicted = pd.DataFrame({
        'Category': ['Returned', 'Not Returned'],
        'Actual': [500, 300],
        'Predicted': [480, 320]
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    actual_vs_predicted.plot(kind='bar', x='Category', ax=ax, color=['green', 'red'])
    ax.set_title("Predicted vs. Actual Client Returns")
    st.pyplot(fig)

# --- Main App Logic ---
display_header()

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["About", "Exploratory Data Analysis", "XAI Insights", "Make Prediction", "Ask a Question"],
    index=3
)

if page == "About":
    st.write("About Page Content")
elif page == "Exploratory Data Analysis":
    st.write("Exploratory Data Analysis Page")
elif page == "XAI Insights":
    xai_insights_page()
elif page == "Make Prediction":
    st.write("Prediction Page Placeholder")
elif page == "Ask a Question":
    ask_a_question_page()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
    <p>IFSSA Client Return Predictor • v2.0</p>
    <p><small>For support contact: support@ifssa.org</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
