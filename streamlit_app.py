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
        model_path = "RF_churn_model.pkl"  # Change this to the correct path
"  # Correct model path
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None

    # Load Model
    model = load_model()
    if model is None:
        st.error("⚠️ No trained model found. Please upload a trained model to 'models/RF_churn_model.pkl'.")

    # Sidebar Inputs
    st.sidebar.markdown("<h2 style='color: #ff5733;'>Input Features</h2>", unsafe_allow_html=True)
    holidays = int(st.sidebar.number_input("Holidays", min_value=0, value=0))
    pickup_week = int(st.sidebar.number_input("Pickup Week", min_value=0, value=1))
    pickup_count_last_14_days = int(st.sidebar.number_input("Pickup Count Last 14 Days", min_value=0, value=5))
    pickup_count_last_30_days = int(st.sidebar.number_input("Pickup Count Last 30 Days", min_value=0, value=10))
    time_since_first_visit = int(st.sidebar.number_input("Time Since First Visit", min_value=0, value=30))
    total_dependents_3_months = int(st.sidebar.number_input("Total Dependents in Last 3 Months", min_value=0, value=2))
    weekly_visits = int(st.sidebar.number_input("Weekly Visits", min_value=0, value=3))
    postal_code = int(st.sidebar.number_input("Postal Code", min_value=0, value=12345))

    # Create DataFrame with Integer Inputs (Removed 'client_return_within_3_months')
    input_data = pd.DataFrame([[holidays, pickup_week, pickup_count_last_14_days, pickup_count_last_30_days, 
                                time_since_first_visit, total_dependents_3_months, weekly_visits, postal_code]], 
                              columns=['holidays', 'pickup_week', 'pickup_count_last_14_days', 'pickup_count_last_30_days', 
                                       'time_since_first_visit', 'total_dependents_3_months', 'weekly_visits', 'postal_code'])

    if st.sidebar.button("🎯 Predict"):
        if model is None:
            st.error("❌ Prediction failed: No trained model found. Please upload a valid model to 'models/RF_churn_model.pkl'.")
        else:
            prediction = model.predict(input_data)
            st.markdown("<h3 style='color: #ff33aa;'>Prediction Result</h3>", unsafe_allow_html=True)
            st.write(f"🎉 **Predicted Outcome:** {int(prediction[0])}")  # Ensures integer output
