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
import shap
from io import BytesIO

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title="IFSSA Return Predictor"
)

# [Previous helper functions remain unchanged...]

def xai_page():
    st.markdown("<h2 style='color: #33aaff;'>Explainable AI (XAI) Insights</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #666;'>
    Understanding how the model makes predictions using SHAP (SHapley Additive exPlanations) values
    </p>
    """, unsafe_allow_html=True)
    
    # Load model and data for SHAP analysis
    with st.spinner("Loading SHAP explainer..."):
        try:
            model = load_model()
            if model is None:
                st.error("Model not loaded - cannot generate SHAP explanations")
                return
            
            # Load precomputed SHAP values (in a real app, these would be precomputed)
            @st.cache_resource
            def load_shap_values():
                # This is placeholder code - in practice you would load precomputed values
                try:
                    # Example of how you might load precomputed SHAP values
                    explainer = shap.TreeExplainer(model)
                    X_sample = pd.DataFrame(np.random.rand(10, 7), columns=[
                        'weekly_visits',
                        'total_dependents_3_months',
                        'pickup_count_last_30_days',
                        'pickup_count_last_14_days',
                        'Holidays',
                        'pickup_week',
                        'time_since_first_visit'
                    ])
                    return explainer.shap_values(X_sample)
                except Exception as e:
                    st.error(f"SHAP computation failed: {str(e)}")
                    return None
            
            shap_values = load_shap_values()
            
            if shap_values is None:
                st.warning("Could not load SHAP values - using example visualization")
                # Create example visualization
                fig, ax = plt.subplots(figsize=(10, 6))
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
                sns.barplot(x=importance, y=features, palette="viridis", ax=ax)
                ax.set_title("Example Feature Importance (SHAP values)")
                ax.set_xlabel("Mean Absolute SHAP Value")
                ax.set_ylabel("Features")
                st.pyplot(fig)
                st.info("Note: This is an example visualization. In production, real SHAP values would be displayed.")
            else:
                # Actual SHAP visualization
                st.markdown("### Feature Impact on Predictions")
                
                # Create SHAP summary plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, plot_type="bar", show=False)
                plt.tight_layout()
                
                # Display in Streamlit
                st.pyplot(fig)
                
                st.markdown("""
                **How to interpret this chart**:
                - Features are ordered by their impact on model predictions
                - Longer bars indicate stronger influence on the prediction
                - Blue bars show positive impact (increasing return probability)
                - Red bars show negative impact (decreasing return probability)
                """)
                
                # Add download button for SHAP values
                if st.button("Download SHAP Analysis Report"):
                    # Create a PDF or HTML report (placeholder)
                    report_content = "SHAP Analysis Report\n\nFeature Importance:\n"
                    for i, feat in enumerate(features):
                        report_content += f"{feat}: {importance[i]:.2f}\n"
                    
                    # Create download link
                    st.download_button(
                        label="Download report as TXT",
                        data=report_content,
                        file_name="shap_analysis_report.txt",
                        mime="text/plain"
                    )
                
        except Exception as e:
            st.error(f"XAI visualization failed: {str(e)}")

# [Previous page functions remain unchanged...]

# --- Main App Logic ---
display_header()

# Navigation - Added XAI page before Prediction
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
    xai_page()
elif page == "Make Prediction":
    prediction_page()

# [Footer remains unchanged...]
