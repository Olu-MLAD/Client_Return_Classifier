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
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

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

@st.cache_resource
def load_chatbot_models():
    """Load chatbot models with error handling"""
    try:
        # Initialize Sentence Transformer Model
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FLAN-T5 model (using base model for better compatibility)
        generator = pipeline(
            "text2text-generation", 
            model="google/flan-t5-base",
            device="cpu"
        )
        
        return embedder, generator
    except Exception as e:
        st.error(f"❌ Failed to load chatbot models: {str(e)}")
        return None, None

def setup_chatbot_knowledge():
    """Set up the chatbot knowledge base"""
    try:
        # Try to load the dataset
        data_path = "IFSSA_cleaned_dataset.csv"
        df = pd.read_csv(data_path) if os.path.exists(data_path) else None
        
        # Generate transaction narrative if data exists
        transaction_narrative = "Recent client transactions:\n"
        if df is not None:
            for _, row in df.head(5).iterrows():  # Limit to 5 rows for demo
                transaction_narrative += (
                    f"Client {row.get('client_list', 'N/A')} "
                    f"({row.get('sex_new', 'N/A')}, Age {row.get('new_age_years', 'N/A')}) "
                    f"picked up {row.get('quantity', 0)} {row.get('hamper_type', 'N/A')} "
                    f"on {row.get('pickup_date', 'N/A')}\n"
                )
        else:
            transaction_narrative = "No recent transaction data available"
        
        # Define IFSSA knowledge base
        documents = {
            "doc1": "IFSSA provides food hampers to families in need, ensuring culturally appropriate meals.",
            "doc2": transaction_narrative,
            "doc3": "Donors can contribute via online payments, bank transfers, or in-person donations.",
            "doc4": "Volunteers assist in packing and distributing hampers every Saturday.",
            "doc5": "The return prediction model helps identify clients likely to return within 3 months.",
            "doc6": "Model outputs: 1 = Will Return, 0 = Will Not Return",
            "doc7": "Key factors in return prediction include pickup frequency and time since first visit."
        }
        
        return documents
    except Exception as e:
        st.error(f"Failed to setup chatbot knowledge: {str(e)}")
        return None

def chat_with_rahim_page():
    st.markdown("<h2 style='color: #33aaff;'>Chat with Rahim</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <b>Welcome to Chat with Rahim!</b><br>
    This AI assistant can answer questions about IFSSA operations, client data, and the return prediction model.
    </div>
    """, unsafe_allow_html=True)
    
    # Load models and knowledge base
    embedder, generator = load_chatbot_models()
    documents = setup_chatbot_knowledge()
    
    if embedder is None or generator is None or documents is None:
        st.error("Chatbot functionality is currently unavailable. Please try again later.")
        return
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask Rahim about IFSSA..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get response
        with st.spinner("Rahim is thinking..."):
            try:
                # Create document embeddings
                doc_embeddings = {doc_id: embedder.encode(text, convert_to_tensor=True) 
                                for doc_id, text in documents.items()}
                
                # Retrieve context
                query_embedding = embedder.encode(user_input, convert_to_tensor=True)
                scores = {
                    doc_id: util.pytorch_cos_sim(query_embedding, emb).item() 
                    for doc_id, emb in doc_embeddings.items()
                }
                sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                context = "\n\n".join(documents[doc_id] for doc_id, _ in sorted_docs[:2])
                
                # Generate response
                prompt = (
                    "You are Rahim, an assistant for IFSSA. Answer questions about food distribution, "
                    "donations, volunteer work, and client return predictions.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {user_input}\n\n"
                    "Provide a detailed, helpful response:"
                )
                response = generator(prompt, max_length=200, do_sample=True, temperature=0.7)
                response_text = response[0]['generated_text'].strip()
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                    
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
    
    # Suggested questions
    st.markdown("""
    <div style='margin-top: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px;'>
    <b>Try asking:</b><br>
    • How does the return prediction model work?<br>
    • What factors influence client returns?<br>
    • How can I donate to IFSSA?<br>
    • Tell me about recent client transactions
    </div>
    """, unsafe_allow_html=True)

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
                        
    except Exception as e:
        with status_container:
            st.error(f"⚠️ Error connecting to Google Sheets: {str(e)}")
        return None

# --- UI Components ---
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

def about_page():
    st.markdown("""
    ## About This Tool
    
    This application helps IFSSA predict which clients are likely to return for services 
    within the next 3 months using machine learning.
    
    ### Model Interpretation
    - **1**: Client will return within 3 months (probability ≥ 50%)
    - **0**: Client will not return within 3 months (probability < 50%)
    """)
    st.image("workflow.png", use_column_width=True)

def exploratory_data_analysis_page():
    st.markdown("<h2>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.write("Data analysis features coming soon")

def xai_insights_page():
    st.markdown("<h2 style='color: #33aaff;'>XAI Insights</h2>", unsafe_allow_html=True)
    model = load_model()
    if model is None:
        st.error("Failed to load model - cannot generate explanations")
        return

    # Create sample data
    X = pd.DataFrame(np.random.rand(10, 6), columns=[
        'pickup_week', 'pickup_count_last_14_days', 
        'pickup_count_last_30_days', 'pickup_count_last_90_days',
        'time_since_first_visit', 'weekly_visits'
    ])

    try:
        with st.spinner("Computing SHAP explanations..."):
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
            
            st.markdown("""
            **How to interpret these plots:**
            - Features are ordered by importance
            - Red = higher feature values
            - Blue = lower feature values
            - Right side = increases probability of return
            """)
    except Exception as e:
        st.error(f"Error generating explanations: {str(e)}")

def prediction_page():
    st.markdown("<h2 style='color: #33aaff;'>Client Return Prediction</h2>", unsafe_allow_html=True)
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

# --- Main App Logic ---
display_header()

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
<p>IFSSA Client Return Predictor • v2.0</p>
<p><small>For support contact: support@ifssa.org</small></p>
</div>
""", unsafe_allow_html=True)
