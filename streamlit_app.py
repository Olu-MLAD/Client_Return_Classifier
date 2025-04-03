import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.base import is_classifier
from datetime import datetime

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
        <p style='text-align: center; font-size: 0.9rem; color: #666;'>
        <b>Model Output:</b> 1 = Will Return, 0 = Will Not Return
        </p>
        """,
        unsafe_allow_html=True
    )

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
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]
            
            # Display results
            st.markdown("---")
            st.markdown("<h3 style='color: #ff33aa;'>Prediction Result</h3>", unsafe_allow_html=True)
            
            col_pred, col_prob, col_expl = st.columns([1,1,2])
            with col_pred:
                st.metric("Binary Prediction", 
                         f"{prediction[0]} - {'Will Return' if prediction[0] == 1 else 'Will Not Return'}",
                         delta="Positive (1)" if prediction[0] == 1 else "Negative (0)",
                         delta_color="normal")
            
            with col_prob:
                st.metric("Return Probability", 
                         f"{probability:.1%}",
                         delta="Confidence level")
            
            # Visual indicator
            if prediction[0] == 1:
                st.success("✅ This client is likely to return within 3 months (prediction = 1)")
            else:
                st.warning("⚠️ This client is unlikely to return within 3 months (prediction = 0)")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Ask a Question Page
def ask_a_question_page():
    st.markdown("<h2 style='color: #33aaff;'>Ask a Question</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <b>If you have any questions or need assistance, feel free to ask below:</b><br>
    </div>
    """, unsafe_allow_html=True)

    with st.form("question_form"):
        # Question form
        question = st.text_area("Enter your question:", "", height=150)
        submit_button = st.form_submit_button(label="Submit Question")

    if submit_button:
        if question.strip() == "":
            st.error("Please enter a question before submitting.")
        else:
            # Simulate the submission of the question
            st.success(f"Your question has been submitted: {question}")
            st.info("Our team will get back to you soon!")

# --- Main App Logic ---
display_header()

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["About", "Exploratory Data Analysis", "XAI Insights", "Make Prediction", "Ask a Question"],
    index=3
)

if page == "About":
    about_page()
elif page == "Exploratory Data Analysis":
    exploratory_data_analysis_page()
elif page == "XAI Insights":
    xai_insights_page()
elif page == "Make Prediction":
    prediction_page()
elif page == "Ask a Question":
    ask_a_question_page()

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
