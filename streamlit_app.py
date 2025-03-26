import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

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

# ================== About the Project ==================
st.markdown("<h2 style='color: #33aaff;'>About the Project</h2>", unsafe_allow_html=True)
st.write("""
This application predicts whether a client will return within a certain period based on various features such as past visits, household size, and holidays.
It uses a machine learning model trained on historical client interaction data to provide insights that help businesses improve retention strategies.
""")

# ================== Exploratory Data Analysis (EDA) ==================
st.markdown("<h2 style='color: #33aaff;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.dataframe(df.head())

    # Display basic statistics
    st.write("### Basic Statistics")
    st.write(df.describe())

    # Select column for visualization
    selected_column = st.selectbox("Select a numerical column for visualization:", df.select_dtypes(include=['number']).columns)
    if selected_column:
        fig = px.histogram(df, x=selected_column, nbins=30, title=f"Distribution of {selected_column}")
        st.plotly_chart(fig)

# ================== Model Explanation ==================
st.markdown("<h2 style='color: #33aaff;'>Model Explanation</h2>", unsafe_allow_html=True)

def load_model():
    model_path = "RF_churn_model.pkl"  # Correct model path
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()
if model:
    st.write("### Feature Importance (if applicable)")
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": ["holiday", "pickup_week", "pickup_count_last_14_days", "pickup_count_last_30_days", "time_since_first_visit", "total_dependents_3_months", "weekly_visits"],
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        fig = px.bar(importance_df, x="Importance", y="Feature", orientation='h', title="Feature Importance")
        st.plotly_chart(fig)
    else:
        st.warning("Feature importance is not available for this model.")
else:
    st.error("⚠️ No trained model found. Please upload a trained model to 'models/RF_churn_model.pkl'.")

# ================== Prediction Section ==================
st.markdown("<h2 style='color: #33aaff;'>Prediction Section</h2>", unsafe_allow_html=True)

# Sidebar Inputs (Updated for user-friendly format)
st.sidebar.markdown("<h2 style='color: #ff5733;'>Input Features</h2>", unsafe_allow_html=True)

holiday = st.sidebar.radio("1. Is this pick up during a holiday?", ["Yes", "No"])
holiday_name = None
if holiday == "Yes":
    holiday_name = st.sidebar.selectbox("Choose the holiday:",
        ["Easter Monday", "Heritage Day", "Labour Day", "Thanksgiving Day", "Remembrance Day", 
         "Christmas Day", "Boxing Day", "New Year's Day", "Good Friday", "Mother's Day", 
         "Victoria Day", "Alberta Family Day", "Father's Day", "Canada Day"])

pickup_week = st.sidebar.number_input("2. What week of the year is the pick up?", min_value=1, max_value=52, value=1)
pickup_count_last_14_days = st.sidebar.radio("3. Was there a pick up in the last 14 days?", ["Yes", "No"])
pickup_count_last_30_days = st.sidebar.radio("4. Was there a pick up in the last 30 days?", ["Yes", "No"])
time_since_first_visit = st.sidebar.number_input("5. Time interval between first and next visit (in days):", min_value=1, max_value=366, value=30)
total_dependents_3_months = st.sidebar.number_input("6. Total dependents in last 3 months:", min_value=0, value=2)
weekly_visits = st.sidebar.number_input("7. How many weekly visits?", min_value=0, value=3)
postal_code = st.sidebar.text_input("8. Enter Postal Code (Canada format: A1A 1A1):")

# Create DataFrame with the Updated Inputs
input_data = pd.DataFrame([[
    holiday, holiday_name, pickup_week, pickup_count_last_14_days, pickup_count_last_30_days, 
    time_since_first_visit, total_dependents_3_months, weekly_visits, postal_code]],
    columns=['holiday', 'holiday_name', 'pickup_week', 'pickup_count_last_14_days', 
             'pickup_count_last_30_days', 'time_since_first_visit', 'total_dependents_3_months', 
             'weekly_visits', 'postal_code'])

if st.sidebar.button("🎯 Predict"):
    if model is None:
        st.error("❌ Prediction failed: No trained model found. Please upload a valid model to 'models/RF_churn_model.pkl'.")
    else:
        prediction = model.predict(input_data)
        st.markdown("<h3 style='color: #ff33aa;'>Prediction Result</h3>", unsafe_allow_html=True)
        st.write(f"🎉 **Predicted Outcome:** {int(prediction[0])}")  # Ensures integer output
