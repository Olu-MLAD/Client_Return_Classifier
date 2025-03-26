import streamlit as st
import pandas as pd
import numpy as np  # Added this import
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title="IFSSA Return Predictor"
)

# Load and Display Logos
col1, col2, _ = st.columns([0.15, 0.15, 0.7])
with col1:
    st.image("logo1.jpeg", width=120)
with col2:
    st.image("logo2.png", width=120)

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: #ff5733; padding: 20px;'>
    IFSSA Client Return Prediction
    </h1>
    <p style='text-align: center; font-size: 1.1rem;'>
    Predict which clients will return within 3 months using statistically validated features
    </p>
    """,
    unsafe_allow_html=True
)

# ================== Navigation ==================
page = st.sidebar.radio(
    "Navigation",
    ["About", "Feature Analysis", "Make Prediction"],
    index=2
)

# ================== About Page ==================
if page == "About":
    st.markdown("""
    ## About This Tool
    
    This application helps IFSSA predict which clients are likely to return for services 
    within the next 3 months using machine learning.
    
    ### How It Works
    1. Staff enter client visit information
    2. The system analyzes patterns from historical data
    3. Predictions guide outreach efforts
    
    ### Key Benefits
    - Enhance Response Accuracy
    - Improve Operational Efficiency
    - Streamline Stakeholder Communication
    - Facilitate Informed Decision Making
    - Ensure Scalability and Flexibility

    """)

# ================== Feature Analysis ==================
elif page == "Feature Analysis":
    st.markdown("## Statistically Validated Predictors")
    
    # Chi-square test results (from your data)
    chi2_results = {
        'monthly_visits': 0.000000e+00,
        'weekly_visits': 0.000000e+00,
        'total_dependents_3_months': 0.000000e+00,
        'pickup_count_last_90_days': 0.000000e+00,
        'pickup_count_last_30_days': 0.000000e+00,
        'pickup_count_last_14_days': 0.000000e+00,
        'pickup_count_last_7_days': 0.000000e+00,
        'Holidays': 8.394089e-90,
        'pickup_week': 1.064300e-69,
        'postal_code': 2.397603e-16,
        'time_since_first_visit': 7.845354e-04
    }
    
    # Convert to dataframe
    chi_df = pd.DataFrame.from_dict(chi2_results, orient='index', columns=['p-value'])
    chi_df['-log10(p)'] = -np.log10(chi_df['p-value'].replace(0, 1e-300))  # Handle zero p-values
    chi_df = chi_df.sort_values('-log10(p)', ascending=False)
    
    # Visualization
    st.markdown("### Feature Significance (-log10 p-values)")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='-log10(p)', y=chi_df.index, data=chi_df, palette="viridis")
    plt.axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
    plt.xlabel("Statistical Significance (-log10 p-value)")
    plt.ylabel("Features")
    plt.title("Chi-Square Test Results for Feature Selection")
    st.pyplot(plt)
    
    # Interpretation
    st.markdown("""
    **Key Insights**:
    - All shown features are statistically significant (p < 0.05)
    - Visit frequency metrics are strongest predictors (p ≈ 0)
    - Holiday effects are 10^90 times more significant than chance
    - Postal code explains location-based patterns (p=2.4e-16)
    """)
    
    # Feature correlations (placeholder - replace with your actual data)
    st.markdown("### Feature Relationships")
    try:
        # Generate sample correlation data if real data isn't available
        corr_data = pd.DataFrame(np.random.rand(6, 6), 
                               columns=chi_df.index[:6], 
                               index=chi_df.index[:6])
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
        st.pyplot(plt)
    except:
        st.warning("Correlation data not available. Displaying sample visualization.")

# ================== Prediction Page ==================
elif page == "Prediction":
    st.markdown("<h2 style='color: #33aaff;'>Prediction Section</h2>", unsafe_allow_html=True)

    # Load Model Function
    def load_model():
        model_path = "RF_model.pkl"
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None

    # Load Model
    model = load_model()
    if model is None:
        st.error("⚠️ No trained model found. Please upload a trained model to 'RF_churn_model.pkl'.")

    # Input Features Section
    st.markdown("<h3 style='color: #ff5733;'>Input Features</h3>", unsafe_allow_html=True)

    # Holiday Selection
    holiday = st.radio("2. Is this pick-up during a holiday?", ["No", "Yes"])

    # Convert to match expected input format
    Holidays = 1 if holiday == "Yes" else 0

    # Conditional Holiday Name Selection
    holiday_name = "None"
    if holiday == "Yes":
        holiday_name = st.selectbox(
            "3. Select the holiday:",
            [
                "New Year's Day", "Good Friday", "Easter Monday", "Victoria Day",
                "Canada Day", "Heritage Day", "Labour Day", "Thanksgiving Day",
                "Remembrance Day", "Christmas Day", "Boxing Day", "Alberta Family Day",
                "Mother's Day", "Father's Day"
            ]
        )

    # Pickup Week and Count Inputs
    pickup_week = st.number_input("2. Pickup Week (1-52):", min_value=1, max_value=52, value=1)
    pickup_count_last_14_days = 1 if st.radio("3. Pickup in last 14 days?", ["No", "Yes"]) == "Yes" else 0
    pickup_count_last_30_days = 1 if st.radio("4. Pickup in last 30 days?", ["No", "Yes"]) == "Yes" else 0

    with st.container():
        # Additional Features
        time_since_first_visit = st.number_input("5. Time Since First Visit (days):", min_value=1, max_value=366, value=30)
        total_dependents_3_months = st.number_input("6. Total Dependents in Last 3 Months:", min_value=0, value=2)
        weekly_visits = st.number_input("7. Weekly Visits:", min_value=0, value=3)
        postal_code = st.text_input("8. Postal Code (Canada format: A1A 1A1):")

    # Ensure Model Compatibility
    input_data = pd.DataFrame([[Holidays, holiday_name, pickup_week, pickup_count_last_14_days, pickup_count_last_30_days,
        time_since_first_visit, total_dependents_3_months, weekly_visits, postal_code]],
        columns=[
        'Holidays', 'holiday_name', 'pickup_week', 'pickup_count_last_14_days', 'pickup_count_last_30_days',
        'time_since_first_visit', 'total_dependents_3_months', 'weekly_visits', 'postal_code'
    ])

    if model:
        model_features = model.feature_names_in_
        missing_features = set(model_features) - set(input_data.columns)
        if missing_features:
            st.error(f"⚠️ Missing Features: {missing_features}. Ensure input names match model training.")

    # Prediction Button
    if st.button("Predict"):
        if model is None:
            st.error("❌ No trained model found. Upload a valid model.")
        else:
            prediction = model.predict(input_data)
            st.markdown("<h3 style='color: #ff33aa;'>Prediction Result</h3>", unsafe_allow_html=True)
            st.write(f"🎉 **Predicted Outcome:** {int(prediction[0])}")
