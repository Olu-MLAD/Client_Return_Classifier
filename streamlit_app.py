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

# --- RAG Setup ---
@st.cache_resource
def load_rag_components():
    """Load RAG components and sample data"""
    # Sample data - replace with your actual data loading logic
    sample_data = {
        "client_list": ["Client A", "Client B"],
        "sex_new": ["Female", "Male"],
        "new_age_years": [35, 42],
        "quantity": [2, 1],
        "hamper_type": ["Standard", "Vegetarian"],
        "pickup_month": ["January", "February"],
        "pickup_date": ["2023-01-15", "2023-02-20"],
        "household": [4, 2],
        "dependents_qty": [2, 1]
    }
    df = pd.DataFrame(sample_data)
    
    # Generate transaction narrative
    transaction_narrative = "Here are recent client transactions:\n"
    for _, row in df.iterrows():
        transaction_narrative += (
            f"Client {row['client_list']} ({row['sex_new']}, Age {row['new_age_years']}) picked up "
            f"{row['quantity']} {row['hamper_type']} hamper(s) in {row['pickup_month']} "
            f"on {row['pickup_date']}. Household size: {row['household']} with {row['dependents_qty']} dependents.\n"
        )

    # Define IFSSA knowledge base
    documents = {
        "doc1": "IFSSA provides food hampers to families in need, ensuring culturally appropriate meals.",
        "doc2": transaction_narrative,
        "doc3": "Donors can contribute via online payments, bank transfers, or in-person donations.",
        "doc4": "Volunteers assist in packing and distributing hampers every Saturday."
    }

    # Initialize models
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline("text2text-generation", model="google/flan-t5-base")  # Using smaller model for demo
    
    return embedder, generator, documents

def retrieve_context(query, embedder, documents, top_k=1):
    doc_embeddings = {doc_id: embedder.encode(text, convert_to_tensor=True) for doc_id, text in documents.items()}
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = {doc_id: util.pytorch_cos_sim(query_embedding, emb).item() for doc_id, emb in doc_embeddings.items()}
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return "\n\n".join(documents[doc_id] for doc_id, _ in sorted_docs[:top_k])

def rag_chatbot(query, embedder, generator, documents):
    context = retrieve_context(query, embedder, documents, top_k=2)
    prompt = (
        "You are Rahim, an assistant for IFSSA, answering questions about food distribution, donations, and volunteer work.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "Answer politely and professionally:"
    )
    response = generator(prompt, max_length=200, do_sample=True, temperature=0.7)
    return response[0]['generated_text'].strip()

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
                st.image(img, caption=f"Chart {i}", use_container_width=True)
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
        show_fallback_xai()
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
        show_fallback_xai()

def show_fallback_xai():
    """Show simplified explanations when SHAP fails"""
    st.warning("Showing simplified feature importance")
    
    features = [
        'pickup_week',
        'pickup_count_last_14_days',
        'pickup_count_last_30_days',
        'pickup_count_last_90_days',
        'time_since_first_visit',
        'weekly_visits'
    ]
    importance = [0.05, 0.10, 0.15, 0.25, 0.02, 0.35]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=importance, y=features, palette="viridis")
    plt.title("Simplified Feature Importance for Return Prediction (1=Return)")
    plt.xlabel("Relative Importance")
    st.pyplot(fig)
    
    st.markdown("""
    **Key Insights**:
    - Weekly visits is the strongest predictor of return visits (1)
    - Pickups in last 90 days is the second most important factor
    - Recent pickup activity strongly influences predictions
    - Time since first visit has a smaller but significant effect
    """)

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
            
            with col_expl:
                st.markdown("""
                **Interpretation**:
                - <span style='color: green;'>1 (Will Return)</span>: Probability ≥ 50%
                - <span style='color: red;'>0 (Will Not Return)</span>: Probability < 50%
                - Threshold can be adjusted for sensitivity
                """, unsafe_allow_html=True)
            
            # Visual indicator
            if prediction[0] == 1:
                st.success("✅ This client is likely to return within 3 months (prediction = 1)")
            else:
                st.warning("⚠️ This client is unlikely to return within 3 months (prediction = 0)")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

    # Google Sheets integration
    st.markdown("---")
    st.subheader("Data Integration Status")
    connect_to_google_sheets()

def chat_with_rahim_page():
    st.markdown("<h2 style='color: #33aaff;'>Chat with Rahim</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <b>RAD Chat Assistant:</b><br>
    • Ask questions about IFSSA services, donations, or volunteer opportunities<br>
    • Get information about client transactions and food distribution<br>
    • Powered by AI with access to IFSSA knowledge base
    </div>
    """, unsafe_allow_html=True)
    
    # Load RAG components
    with st.spinner("Loading Rahim's knowledge..."):
        embedder, generator, documents = load_rag_components()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask Rahim about IFSSA..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Rahim is thinking..."):
                response = rag_chatbot(prompt, embedder, generator, documents)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Main App Logic ---
display_header()

# Navigation
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
