import streamlit as st
import pandas as pd
import joblib
import os

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
    "<h1 style='text-align: center; color: #ff5733;'>Client Return Prediction App</h1>",
    unsafe_allow_html=True
)

# Sidebar Navigation
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
    st.markdown("<h2 style='color: #33aaff;'>Will This Client Return in 3 Months?</h2>", unsafe_allow_html=True)
    st.write(
        "This prediction helps IFSSA classify clients based on their likelihood of returning for services within the next three months. "
        "By analyzing client behavior patterns, IFSSA can enhance outreach efforts and optimize resource allocation."
    )

    # Load Model Function
    @st.cache_resource
    def load_model():
        model_path = "RF_model.pkl"
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None

    # Load Model
    model = load_model()
    if model is None:
        st.error("⚠️ No trained model found. Please upload a trained model to 'RF_model.pkl'.")

    # Input Features Section
    st.markdown("<h3 style='color: #ff5733;'>Enter Client Data</h3>", unsafe_allow_html=True)

    # Holiday Selection
    holiday = st.radio("Is this pick-up during a holiday?", ["No", "Yes"])
    Holidays = 1 if holiday == "Yes" else 0

    # Pickup Week and Count Inputs
    pickup_week = st.number_input("Pickup Week (1-52):", min_value=1, max_value=52, value=1)
    pickup_count_last_14_days = 1 if st.radio("Pickup in last 14 days?", ["No", "Yes"]) == "Yes" else 0
    pickup_count_last_30_days = 1 if st.radio("Pickup in last 30 days?", ["No", "Yes"]) == "Yes" else 0

    # Additional Features
    time_since_first_visit = st.number_input("Time Since First Visit (days):", min_value=1, max_value=366, value=30)
    total_dependents_3_months = st.number_input("Total Dependents in Last 3 Months:", min_value=0, value=2)
    weekly_visits = st.number_input("Weekly Visits:", min_value=0, value=3)
    postal_code = st.text_input("Postal Code (Canada format: A1A 1A1):")

    # Prepare Input Data (Removing 'holiday_name' to match model features)
    input_data = pd.DataFrame([[Holidays, pickup_week, pickup_count_last_14_days, pickup_count_last_30_days,
        time_since_first_visit, total_dependents_3_months, weekly_visits, postal_code]],
        columns=[
        'Holidays', 'pickup_week', 'pickup_count_last_14_days', 'pickup_count_last_30_days',
        'time_since_first_visit', 'total_dependents_3_months', 'weekly_visits', 'postal_code'
    ])

    if model:
        model_features = model.feature_names_in_
        missing_features = set(model_features) - set(input_data.columns)
        extra_features = set(input_data.columns) - set(model_features)

        if missing_features:
            st.error(f"⚠️ Missing Features: {missing_features}. Ensure input names match model training.")
        if extra_features:
            st.warning(f"⚠️ Extra Features in Input: {extra_features}. These might need to be removed.")

    # Prediction Button
    if st.button("Predict"): 
        if model is None:
            st.error("❌ No trained model found. Upload a valid model.")
        else:
            try:
                prediction = model.predict(input_data)
                result_text = "✔️ This client is likely to return within 3 months." if prediction[0] == 1 else "❌ This client is unlikely to return within 3 months."
                st.markdown("<h3 style='color: #ff33aa;'>Prediction Result</h3>", unsafe_allow_html=True)
                st.write(result_text)
            except Exception as e:
                st.error(f"🚨 Prediction error: {str(e)}")
