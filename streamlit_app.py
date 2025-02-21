import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(layout="wide")

# Load and Display Logos
col1, col2 = st.columns([0.2, 0.8])  # Adjust proportions as needed
with col1:
    st.image("logo1.jpeg", width=200)  # Update with actual file path
    st.image("logo2.png", width=200)

# Title and Problem Statement
st.title("Client Retention Prediction App")

st.markdown("## Problem Statement")
st.write("The IFSSA (Islamic Family and Social Services Association) struggles to predict when and how many clients will return to get hampers, leading to challenges in inventory management, resource allocation, and client retention strategies. This uncertainty affects operational efficiency and limits the ability to tailor the organization’s efforts effectively.")

# Project Goals
st.markdown("## Project Goals")
st.write("- Identify patterns in customer behavior and historical data to support decision-making.")
st.write("- Develop a machine learning model to predict whether clients will return within a specified time frame.")
st.write("- Improve operational efficiency by enabling better inventory management and resource planning.")

# Load Model Function
def load_model():
    model_path = "models/model.pkl"  # Update if needed
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Load Model
model = load_model()
if model is None:
    st.error("⚠️ No trained model found. Please upload a trained model to 'models/model.pkl'.")

# Sidebar Inputs
st.sidebar.header("Input Features")
time_since_last_pickup = st.sidebar.number_input("Time Since Last Pickup", min_value=0.0, value=10.0)
hamper_confirmation_type = st.sidebar.number_input("Hamper Confirmation Type", min_value=0.0, value=1.0)
preferred_contact_methods = st.sidebar.number_input("Preferred Contact Methods", min_value=0.0, value=1.0)
status = st.sidebar.number_input("Client Status", min_value=0.0, value=1.0)
sex_new = st.sidebar.number_input("Sex (Encoded)", min_value=0.0, value=1.0)
new_age_years = st.sidebar.number_input("Age in Years", min_value=0.0, value=35.0)
hamper_demand_lag_30 = st.sidebar.number_input("Hamper Demand Lag 30 Days", min_value=0.0, value=2.0)
latest_contact_method = st.sidebar.number_input("Latest Contact Method", min_value=0.0, value=1.0)
dependents_qty = st.sidebar.number_input("Dependents Quantity", min_value=0.0, value=3.0)
household = st.sidebar.number_input("Household Size", min_value=0.0, value=4.0)
contact_frequency = st.sidebar.number_input("Contact Frequency", min_value=0.0, value=5.0)

input_data = pd.DataFrame([[time_since_last_pickup, hamper_confirmation_type, preferred_contact_methods, 
                            status, sex_new, new_age_years, hamper_demand_lag_30, latest_contact_method, 
                            dependents_qty, household, contact_frequency]], 
                          columns=['time_since_last_pickup', 'hamper_confirmation_type', 'preferred_contact_methods', 
                                   'status', 'sex_new', 'new_age_years', 'hamper_demand_lag_30', 'latest_contact_method', 
                                   'dependents_qty', 'household', 'contact_frequency'])

if st.sidebar.button("Predict"):
    if model is None:
        st.error("❌ Prediction failed: No trained model found. Please upload a valid model to 'models/model.pkl'.")
    else:
        prediction = model.predict(input_data)
        st.write("### Prediction Result")
        st.write(f"Predicted Outcome: {prediction[0]}")
