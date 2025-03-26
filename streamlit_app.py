import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(layout="wide")

# Load and Display Logos Side by Side
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
        "The organization aims to use artificial intelligence to improve their operations and efficiently tailor their efforts to support "
        "the community by addressing challenges faced in the areas of inventory management, resource allocation, and delayed/inconsistent "
        "information shared with stakeholders."
    )
    st.write(
        "We have received a food hamper dataset consisting of two CSV files (Clients Data Dimension and Food Hampers Fact) to analyze and "
        "build a machine learning model to predict customer churn over a period of time."
    )

    st.markdown("<h2 style='color: #33aaff;'>Problem Statement (Client Retention Classification)</h2>", unsafe_allow_html=True)
    st.write(
        "This problem involves classifying clients to determine if they are likely to return to use IFSSA services within a 3-month time frame. "
        "By identifying client behavior patterns, IFSSA can plan outreach efforts or adjust services to better meet the needs of its clients, "
        "ensuring efficient resource use."
    )

    st.markdown("<h2 style='color: #33aaff;'>Approach</h2>", unsafe_allow_html=True)
    st.write(
        "1. Import datasets into pandas dataframe.\n"
        "2. Visualize datasets to understand structure, patterns, and relationships amongst features.\n"
        "3. Merge dataframes using a column common to both.\n"
        "4. Clean data and prepare for feature engineering and modeling (remove duplicates, outliers, and redundant data; handle missing values by filling or removing)."
    )

    st.markdown("<h2 style='color: #33aaff;'>Project Goals</h2>", unsafe_allow_html=True)
    st.write("✅ Identify patterns in customer behavior and historical data to support decision-making.")
    st.write("✅ Develop a machine learning model to predict whether clients will return within a specified time frame.")
    st.write("✅ Improve operational efficiency by enabling better inventory management and resource planning.")

# ================== Exploratory Data Analysis ==================
elif page == "Exploratory Data Analysis":
    st.markdown("<h2 style='color: #33aaff;'>Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)

    # Introduction to EDA
    st.write(
        "In this section, we explore the dataset to understand its structure, identify patterns, "
        "and visualize key insights. Below are some pre-generated charts to help you get started."
    )

    # Display Pre-generated Charts (chart1.png to chart7.png)
    st.write("### Pre-generated Charts")
    chart_paths = [f"chart{i}.png" for i in range(1, 8)]  # List of chart paths

    # Display charts in a grid layout
    cols = st.columns(2)  # 2 columns for the grid
    for idx, chart_path in enumerate(chart_paths):
        with cols[idx % 2]:  # Alternate between columns
            st.image(chart_path, caption=f"Chart {idx + 1}", use_container_width=True)  # Fixed deprecation warning

# ================== Prediction Section ==================
elif page == "Prediction":
    st.markdown("<h2 style='color: #33aaff;'>Prediction Section</h2>", unsafe_allow_html=True)

    # Load Model Function
    def load_model():
        model_path = "RF_churn_model.pkl"  # Correct model path
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None

    # Load Model
    model = load_model()
    if model is None:
        st.error("⚠️ No trained model found. Please upload a trained model to 'RF_churn_model.pkl'.")

    # Input Features
    st.markdown("<h3 style='color: #ff5733;'>Input Features</h3>", unsafe_allow_html=True)
    holiday = st.radio("1. Is this pick up during a holiday?", ["Yes", "No"])
    holiday_name = None
    if holiday == "Yes":
        holiday_name = st.selectbox(
            "Choose the holiday:",
            ["Easter Monday", "Heritage Day", "Labour Day", "Thanksgiving Day", "Remembrance Day", 
             "Christmas Day", "Boxing Day", "New Year's Day", "Good Friday", "Mother's Day", 
             "Victoria Day", "Alberta Family Day", "Father's Day", "Canada Day"]
        )
    pickup_week = st.number_input("2. What week of the year is the pick up?", min_value=1, max_value=52, value=1)
    pickup_count_last_14_days = st.radio("3. Was there a pick up in the last 14 days?", ["Yes", "No"])
    pickup_count_last_30_days = st.radio("4. Was there a pick up in the last 30 days?", ["Yes", "No"])
    time_since_first_visit = st.number_input("5. Time interval between the first visit and the next visit (in days):", 
                                            min_value=1, max_value=366, value=30)
    total_dependents_3_months = st.number_input("6. Total dependents in the last 3 months:", min_value=0, value=2)
    weekly_visits = st.number_input("7. How many weekly visits?", min_value=0, value=3)
    postal_code = st.text_input("8. Enter Postal Code (Canada format: A1A 1A1):")

    input_data = pd.DataFrame([[holiday, holiday_name, pickup_week, pickup_count_last_14_days, pickup_count_last_30_days, 
                                time_since_first_visit, total_dependents_3_months, weekly_visits, postal_code]],
                              columns=['holiday', 'holiday_name', 'pickup_week', 'pickup_count_last_14_days', 
                                       'pickup_count_last_30_days', 'time_since_first_visit', 'total_dependents_3_months', 
                                       'weekly_visits', 'postal_code'])

    if st.button("🎯 Predict"):
        if model is None:
            st.error("❌ Prediction failed: No trained model found. Please upload a valid model to 'models/RF_churn_model.pkl'.")
        else:
            prediction = model.predict(input_data)
            st.markdown("<h3 style='color: #ff33aa;'>Prediction Result</h3>", unsafe_allow_html=True)
            st.write(f"🎉 **Predicted Outcome:** {int(prediction[0])}")
