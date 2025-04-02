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
            
        if not (hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
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
        
        # Initialize FLAN-T5 model
        generator = pipeline("text2text-generation", model="google/flan-t5-large")
        
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
        transaction_narrative = "Here are recent client transactions:\n"
        if df is not None:
            for _, row in df.iterrows():
                transaction_narrative += (
                    f"Client {row['client_list']} ({row['sex_new']}, Age {row['new_age_years']}) picked up "
                    f"{row['quantity']} {row['hamper_type']} hamper(s) in {row['pickup_month']} "
                    f"on {row['pickup_date']}. Household size: {row['household']} with {row['dependents_qty']} dependents.\n"
                )
        else:
            transaction_narrative = "No recent transaction data available."
        
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
    
    # Create document embeddings
    doc_embeddings = {doc_id: embedder.encode(text, convert_to_tensor=True) 
                     for doc_id, text in documents.items()}
    
    def retrieve_context(query, top_k=2):
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        scores = {doc_id: util.pytorch_cos_sim(query_embedding, emb).item() 
                for doc_id, emb in doc_embeddings.items()}
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return "\n\n".join(documents[doc_id] for doc_id, _ in sorted_docs[:top_k])
    
    def query_llm(query, context):
        prompt = (
            "You are Rahim, an assistant for IFSSA. Answer questions about food distribution, "
            "donations, volunteer work, and client return predictions.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Provide a detailed, helpful response:"
        )
        response = generator(prompt, max_length=200, do_sample=True, temperature=0.7)
        return response[0]['generated_text'].strip()
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask Rahim about IFSSA...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get response
        with st.spinner("Rahim is thinking..."):
            try:
                context = retrieve_context(user_input)
                response = query_llm(user_input, context)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                    
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

# ... [Keep all your existing functions unchanged until the main app logic] ...

# --- Main App Logic ---
display_header()

# Navigation - Add "Chat with Rahim" to the radio options
page = st.sidebar.radio(
    "Navigation",
    ["About", "Exploratory Data Analysis", "XAI Insights", "Make Prediction", "Chat with Rahim"],
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
elif page == "Chat with Rahim":
    chat_with_rahim_page()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
    <p>IFSSA Client Return Predictor • v1.9</p>
    <p><small>Model outputs: 1 = Return, 0 = No Return | For support contact: support@ifssa.org</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
