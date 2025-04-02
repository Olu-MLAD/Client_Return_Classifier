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

# ========== FUNCTION DEFINITIONS ==========

def display_header():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("logo.png", width=100)
    with col2:
        st.markdown("""
        <h1 style='color: #ff5733;'>
        IFSSA Client Return Prediction
        </h1>
        <p style='color: #666;'>
        Predict which clients will return within 3 months
        </p>
        """, unsafe_allow_html=True)

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

# Lazy loading for NLP components
def get_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

def get_transformers_pipeline():
    from transformers import pipeline
    return pipeline

@st.cache_resource
def load_embedder():
    try:
        SentenceTransformer = get_sentence_transformer()
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Embedder loading failed: {e}")
        return None

@st.cache_resource
def load_generator():
    try:
        pipeline = get_transformers_pipeline()
        return pipeline(
            "text2text-generation", 
            model="google/flan-t5-base",
            device="cpu"
        )
    except Exception as e:
        st.error(f"Generator loading failed: {e}")
        return None

def setup_chatbot_knowledge():
    """Set up the chatbot knowledge base"""
    try:
        data_path = "IFSSA_cleaned_dataset.csv"
        df = pd.read_csv(data_path) if os.path.exists(data_path) else None
        
        transaction_narrative = "Recent client transactions:\n"
        if df is not None:
            for _, row in df.head(5).iterrows():
                transaction_narrative += (
                    f"Client {row.get('client_list', 'N/A')} "
                    f"({row.get('sex_new', 'N/A')}, Age {row.get('new_age_years', 'N/A')}) "
                    f"picked up {row.get('quantity', 0)} {row.get('hamper_type', 'N/A')} "
                    f"on {row.get('pickup_date', 'N/A')}\n"
                )
        else:
            transaction_narrative = "No transaction data available"
        
        documents = {
            "doc1": "IFSSA provides food hampers to families in need.",
            "doc2": transaction_narrative,
            "doc3": "Donations accepted via online payments or bank transfers.",
            "doc4": "Volunteers help pack hampers every Saturday.",
            "doc5": "Return prediction model identifies likely returning clients."
        }
        return documents
    except Exception as e:
        st.error(f"Knowledge setup failed: {e}")
        return None

def chat_with_rahim_page():
    st.markdown("<h2 style='color: #33aaff;'>Chat with Rahim</h2>", unsafe_allow_html=True)
    
    embedder = load_embedder()
    generator = load_generator()
    documents = setup_chatbot_knowledge()
    
    if None in [embedder, generator, documents]:
        st.error("Chatbot components not available")
        return
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_input := st.chat_input("Ask Rahim about IFSSA..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.spinner("Thinking..."):
            try:
                from sentence_transformers import util
                query_embedding = embedder.encode(user_input, convert_to_tensor=True)
                scores = {
                    doc_id: util.pytorch_cos_sim(query_embedding, 
                    embedder.encode(text, convert_to_tensor=True)).item()
                    for doc_id, text in documents.items()
                }
                context = documents[max(scores, key=scores.get)]
                
                response = generator(
                    f"Answer this IFSSA question: {user_input}\nContext: {context}",
                    max_length=150
                )[0]['generated_text']
                
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )
                with st.chat_message("assistant"):
                    st.error(error_msg)

def about_page():
    st.markdown("""
    ## About This Tool
    Predicts which clients will return within 3 months.
    - **1**: Will Return
    - **0**: Will Not Return
    """)

def exploratory_data_analysis_page():
    st.markdown("<h2>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.write("Data analysis features coming soon")

def xai_insights_page():
    st.markdown("<h2>XAI Insights</h2>", unsafe_allow_html=True)
    model = load_model()
    if model:
        try:
            X = pd.DataFrame(np.random.rand(10, 6), columns=[
                'pickup_week', 'pickup_count_last_14_days', 
                'pickup_count_last_30_days', 'pickup_count_last_90_days',
                'time_since_first_visit', 'weekly_visits'
            ])
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            st.subheader("Feature Importance")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
            st.pyplot(fig1)
            
            st.subheader("Feature Impact")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values[1], X, show=False)
            st.pyplot(fig2)
            
        except Exception as e:
            st.error(f"Error generating explanations: {str(e)}")

def prediction_page():
    st.markdown("<h2>Client Return Prediction</h2>", unsafe_allow_html=True)
    model = load_model()
    if model is None:
        st.stop()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pickup_week = st.number_input("Pickup Week", 1, 52, 1)
            pickup_14 = st.number_input("Pickups (14 days)", 0, 20, 0)
            pickup_30 = st.number_input("Pickups (30 days)", 0, 20, 0)
        with col2:
            pickup_90 = st.number_input("Pickups (90 days)", 0, 50, 0)
            time_since = st.number_input("Days since first visit", 1, 365, 30)
            weekly = st.number_input("Weekly visits", 0, 10, 1)
        
        if st.form_submit_button("Predict"):
            input_data = [[pickup_week, pickup_14, pickup_30, pickup_90, time_since, weekly]]
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1]
            
            st.metric("Prediction", 
                     f"{prediction} - {'Return' if prediction == 1 else 'No Return'}",
                     f"{proba:.1%} confidence")

# ========== MAIN APP LOGIC ==========

# First display the header
display_header()

# Then set up page navigation
page = st.sidebar.radio(
    "Navigation",
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
st.markdown("""
<div style='text-align: center;'>
<p>IFSSA Client Return Predictor • v2.2</p>
<p><small>For support contact: support@ifssa.org</small></p>
</div>
""", unsafe_allow_html=True)
