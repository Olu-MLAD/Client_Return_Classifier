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

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    try:
        # Replace with your actual data loading logic
        sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.randint(0, 5, 100),
            'target': np.random.randint(0, 2, 100)
        })
        return sample_data
    except Exception as e:
        st.error(f"Failed to load sample data: {str(e)}")
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

def xai_insights_page():
    st.markdown("<h2 style='color: #33aaff;'>XAI Insights</h2>", unsafe_allow_html=True)
    st.markdown("""
    Explainable AI (XAI) helps understand the model's decision-making process.
    This section provides various visualizations to interpret the model's behavior.
    """)
    
    model = load_model()
    if model is None:
        return
    
    # Load sample data
    data = load_sample_data()
    if data is None:
        return
    
    # Prepare SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
        sample_for_shap = data.drop(columns=['target']).sample(min(50, len(data)))
        shap_values = explainer.shap_values(sample_for_shap)
        
        # Tab layout for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Feature Importance", 
            "SHAP Summary", 
            "Dependence Plots", 
            "Decision Plots"
        ])
        
        with tab1:
            st.subheader("Global Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            features = sample_for_shap.columns[indices]
            
            ax.barh(range(len(indices)), importances[indices], align='center', color='#33aaff')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Relative Importance')
            ax.set_title('Feature Importance Ranking')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add feature importance boxplot
            st.subheader("Feature Importance Distribution")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=pd.DataFrame(importances.reshape(1, -1), columns=sample_for_shap.columns), 
                       orient='h', palette='viridis', ax=ax2)
            ax2.set_title('Distribution of Feature Importances')
            plt.tight_layout()
            st.pyplot(fig2)
            
        with tab2:
            st.subheader("SHAP Summary Plot")
            fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, sample_for_shap, plot_type="bar", show=False)
            plt.tight_layout()
            st.pyplot(fig_summary)
            
            st.subheader("SHAP Feature Impact")
            fig_detailed, ax_detailed = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values[1], sample_for_shap, show=False)
            plt.tight_layout()
            st.pyplot(fig_detailed)
            
        with tab3:
            st.subheader("Feature Dependence Plots")
            selected_feature = st.selectbox(
                "Select feature to analyze dependence:",
                options=sample_for_shap.columns
            )
            
            if selected_feature:
                fig_dep, ax_dep = plt.subplots(figsize=(10, 6))
                shap.dependence_plot(
                    selected_feature, 
                    shap_values[1], 
                    sample_for_shap, 
                    interaction_index=None,
                    ax=ax_dep,
                    show=False
                )
                plt.title(f"Dependence Plot for {selected_feature}")
                plt.tight_layout()
                st.pyplot(fig_dep)
                
                # Add interaction analysis
                st.subheader("Interaction Analysis")
                interact_feature = st.selectbox(
                    "Select interaction feature:",
                    options=[col for col in sample_for_shap.columns if col != selected_feature]
                )
                
                if interact_feature:
                    fig_interact, ax_interact = plt.subplots(figsize=(10, 6))
                    shap.dependence_plot(
                        selected_feature, 
                        shap_values[1], 
                        sample_for_shap, 
                        interaction_index=interact_feature,
                        ax=ax_interact,
                        show=False
                    )
                    plt.title(f"Interaction between {selected_feature} and {interact_feature}")
                    plt.tight_layout()
                    st.pyplot(fig_interact)
                    
        with tab4:
            st.subheader("SHAP Decision Plot")
            sample_idx = st.slider(
                "Select sample index to explain:",
                min_value=0,
                max_value=len(sample_for_shap)-1,
                value=0
            )
            
            fig_decision, ax_decision = plt.subplots(figsize=(10, 6))
            shap.decision_plot(
                explainer.expected_value[1], 
                shap_values[1][sample_idx], 
                sample_for_shap.iloc[sample_idx],
                show=False
            )
            plt.title(f"Decision Plot for Sample {sample_idx}")
            plt.tight_layout()
            st.pyplot(fig_decision)
            
            # Add force plot
            st.subheader("SHAP Force Plot")
            force_plot = shap.force_plot(
                explainer.expected_value[1],
                shap_values[1][sample_idx],
                sample_for_shap.iloc[sample_idx],
                matplotlib=True,
                show=False
            )
            st.pyplot(force_plot)
            
    except Exception as e:
        st.error(f"Error generating SHAP explanations: {str(e)}")

def about_page():
    st.markdown("<h2 style='color: #33aaff;'>About the IFSSA Client Return Predictor</h2>", unsafe_allow_html=True)
    st.markdown("""
    This application predicts whether a client will return based on past behavior and data insights. 
    The model uses a random forest classifier to make predictions and provide actionable insights.
    """)

# --- Main App Logic ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["About", "Exploratory Data Analysis", "XAI Insights", "Make Prediction", "Chat with Rahim"]
)

if page == "About":
    about_page()
elif page == "Exploratory Data Analysis":
    st.write("Exploratory Data Analysis Page - To be implemented")
elif page == "XAI Insights":
    xai_insights_page()
elif page == "Make Prediction":
    st.write("Prediction Page - To be implemented")
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
