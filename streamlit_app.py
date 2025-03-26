import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(layout="wide")

# Load and Display Logos
col1, col2, _ = st.columns([0.15, 0.15, 0.7])
with col1:
    st.image("logo1.jpeg", width=120)
with col2:
    st.image("logo2.png", width=120)

# Colorful Header
st.markdown(
    "<h1 style='text-align: center; color: #ff5733;'>Client Return Prediction App (MVP)</h1>",
    unsafe_allow_html=True
)

# ================== Navigation Bar ==================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["About the Project", "Exploratory Data Analysis", "Prediction"]
)

# ================== About the Project ==================
if page == "About the Project":
    st.markdown("<h2 style='color: #33aaff;'>Introduction</h2>", unsafe_allow_html=True)
    st.write(
        "The Islamic Family & Social Services Association (IFSSA) is a social service organization based in Edmonton, Alberta, Canada. "
        "It provides a range of community services, such as food hampers, crisis support, and assistance for refugees. "
        "The organization aims to use artificial intelligence to improve operations and enhance support efforts."
    )

    st.markdown("<h2 style='color: #33aaff;'>Problem Statement</h2>", unsafe_allow_html=True)
    st.write(
        "This project focuses on classifying clients to determine if they are likely to return within a 3-month period. "
        "By identifying client behavior patterns, IFSSA can enhance outreach efforts and optimize resource allocation."
    )

# ================== Exploratory Data Analysis ==================
elif page == "Exploratory Data Analysis":
    st.markdown("<h2 style='color: #33aaff;'>Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)
    st.write("### Pre-generated Charts")
    chart_paths = [f"chart{i}.png" for i in range(1, 8)]
    cols = st.columns(2)
    for idx, chart_path in enumerate(chart_paths):
        with cols[idx % 2]:  
            st.image(chart_path, caption=f"Chart {idx + 1}", use_container_width=True)

# ================== Prediction Section ==================
elif page == "Prediction":
    st.markdown("<h2 style='color: #33aaff;'>Prediction Section</h2>", unsafe_allow_html=True)

    # Load Model Function
    def load_model():
        model_path = "RF_churn_model.pkl"
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None

    # Load Model
    model = load_model()
    if model is None:
        st.error("⚠️ No trained model found. Please upload a trained model to 'RF_churn_model.pkl'.")

    # Feature Input Form
    st.markdown("<h3 style='color: #ff5733;'>Input Features</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        holiday = st.radio("1. Is this pick-up during a holiday?", ["No", "Yes"])
        holiday_val = 1 if holiday == "Yes" else 0
        holiday_name = None
        if holiday == "Yes":
            holiday_name = st.selectbox(
                "Choose the holiday:",
                ["Easter Monday", "Heritage Day", "Labour Day", "Thanksgiving Day", "Remembrance Day",
                 "Christmas Day", "Boxing Day", "New Year's Day", "Good Friday", "Mother's Day",
                 "Victoria Day", "Alberta Family Day", "Father's Day", "Canada Day"]
            )

        pickup_week = st.number_input("2. Pickup Week (1-52):", min_value=1, max_value=52, value=1)
        pickup_count_last_14_days = 1 if st.radio("3. Pickup in last 14 days?", ["No", "Yes"]) == "Yes" else 0
        pickup_count_last_30_days = 1 if st.radio("4. Pickup in last 30 days?", ["No", "Yes"]) == "Yes" else 0

    with col2:
        time_since_first_visit = st.number_input("5. Time Since First Visit (days):", min_value=1, max_value=366, value=30)
        total_dependents_3_months = st.number_input("6. Total Dependents in Last 3 Months:", min_value=0, value=2)
        weekly_visits = st.number_input("7. Weekly Visits:", min_value=0, value=3)
        postal_code = st.text_input("8. Postal Code (Canada format: A1A 1A1):")

    # Ensure Model Compatibility
    input_data = pd.DataFrame([[
        holiday_val, holiday_name, pickup_week, pickup_count_last_14_days, pickup_count_last_30_days,
        time_since_first_visit, total_dependents_3_months, weekly_visits, postal_code
    ]], columns=[
        'holiday', 'holiday_name', 'pickup_week', 'pickup_count_last_14_days', 'pickup_count_last_30_days',
        'time_since_first_visit', 'total_dependents_3_months', 'weekly_visits', 'postal_code'
    ])

    if model:
        model_features = model.feature_names_in_
        missing_features = set(model_features) - set(input_data.columns)
        if missing_features:
            st.error(f"⚠️ Missing Features: {missing_features}. Ensure input names match model training.")

    # Prediction Button
    if st.button("🎯 Predict"):
        if model is None:
            st.error("❌ No trained model found. Upload a valid model.")
        else:
            prediction = model.predict(input_data)
            st.markdown("<h3 style='color: #ff33aa;'>Prediction Result</h3>", unsafe_allow_html=True)
            st.write(f"🎉 **Predicted Outcome:** {int(prediction[0])}")
