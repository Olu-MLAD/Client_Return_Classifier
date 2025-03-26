import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title="IFSSA Return Predictor",
    page_icon="🔮"
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
    3-Month Return Probability Calculator
    </p>
    """,
    unsafe_allow_html=True
)

# ================== Model Loading ==================
@st.cache_resource
def load_model():
    try:
        if os.path.exists("RF_model.pkl"):
            model = joblib.load("RF_model.pkl")
            if hasattr(model, 'feature_names_in_'):
                return model
        return None
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

model = load_model()

# ================== Input Form ==================
st.markdown("## Client Visit Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Holiday Information
        holiday = st.radio(
            "1. Is this pick-up during a holiday?",
            ["No", "Yes"],
            horizontal=True
        )
        Holidays = 1 if holiday == "Yes" else 0
        
        holiday_name = "None"
        if holiday == "Yes":
            holiday_name = st.selectbox(
                "2. Select the holiday:",
                [
                    "New Year's Day", "Good Friday", "Easter Monday", 
                    "Victoria Day", "Canada Day", "Heritage Day",
                    "Labour Day", "Thanksgiving Day", "Remembrance Day",
                    "Christmas Day", "Boxing Day", "Alberta Family Day",
                    "Mother's Day", "Father's Day"
                ]
            )
        
        # Pickup Information
        pickup_week = st.number_input(
            "3. Pickup Week (1-52):",
            min_value=1,
            max_value=52,
            value=datetime.now().isocalendar()[1]  # Current week
        )
        
        pickup_count_last_14_days = st.radio(
            "4. Pickup in last 14 days?",
            ["No", "Yes"],
            horizontal=True
        )
        pickup_count_last_14_days = 1 if pickup_count_last_14_days == "Yes" else 0
        
        pickup_count_last_30_days = st.radio(
            "5. Pickup in last 30 days?",
            ["No", "Yes"],
            horizontal=True
        )
        pickup_count_last_30_days = 1 if pickup_count_last_30_days == "Yes" else 0
    
    with col2:
        # Client History
        time_since_first_visit = st.number_input(
            "6. Days since first visit (1-366):",
            min_value=1,
            max_value=366,
            value=30
        )
        
        total_dependents_3_months = st.number_input(
            "7. Total dependents (last 3 months):",
            min_value=0,
            value=1
        )
        
        weekly_visits = st.number_input(
            "8. Weekly visits:",
            min_value=0,
            value=1
        )
        
        postal_code = st.text_input(
            "9. Postal Code (A1A 1A1 format):",
            placeholder="T5J 2R1",
            max_length=7
        ).upper()
    
    submitted = st.form_submit_button("Predict Return Probability", type="primary")

# ================== Prediction Logic ==================
if submitted:
    if not model:
        st.error("Model not loaded. Please check RF_model.pkl exists.")
        st.stop()
    
    try:
        # Prepare input data
        input_data = pd.DataFrame([{
            'Holidays': Holidays,
            'pickup_week': pickup_week,
            'pickup_count_last_14_days': pickup_count_last_14_days,
            'pickup_count_last_30_days': pickup_count_last_30_days,
            'time_since_first_visit': time_since_first_visit,
            'total_dependents_3_months': total_dependents_3_months,
            'weekly_visits': weekly_visits,
            'postal_code': postal_code[:3]  # Using first 3 characters
        }])
        
        # Handle holiday_name if needed (one-hot encoding example)
        if 'holiday_name_None' in model.feature_names_in_:
            all_holidays = [
                "New Year's Day", "Good Friday", "Easter Monday", 
                "Victoria Day", "Canada Day", "Heritage Day",
                "Labour Day", "Thanksgiving Day", "Remembrance Day",
                "Christmas Day", "Boxing Day", "Alberta Family Day",
                "Mother's Day", "Father's Day", "None"
            ]
            for h in all_holidays:
                col_name = f"holiday_name_{h.replace(' ', '_').replace("'", "")}"
                input_data[col_name] = 1 if holiday_name == h else 0
        
        # Ensure correct feature order
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # Make prediction
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)
        return_prob = proba[0][1]  # Probability of returning
        
        # Display results
        st.markdown("---")
        st.markdown(f"""
        ## Prediction Result
        <div style='background-color:#f0f2f6; padding:20px; border-radius:10px;'>
        <h3 style='color:#33aaff;'>
        Probability of returning within 3 months: <b>{return_prob:.0%}</b>
        </h3>
        """, unsafe_allow_html=True)
        
        # Interpretation
        if return_prob >= 0.75:
            st.success("High likelihood of return - standard follow-up recommended")
        elif return_prob >= 0.5:
            st.warning("Moderate likelihood - consider additional outreach")
        else:
            st.error("Low likelihood - prioritize for proactive engagement")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Key factors
        st.markdown("""
        ### Key Influencing Factors
        - **Recent activity:** {recent_14}d / {recent_30}d pickups
        - **Visit pattern:** {weekly} weekly visits
        - **Client tenure:** {tenure} days since first visit
        - **Family size:** {dependents} dependents
        """.format(
            recent_14="Yes" if pickup_count_last_14_days else "No",
            recent_30="Yes" if pickup_count_last_30_days else "No",
            weekly=weekly_visits,
            tenure=time_since_first_visit,
            dependents=total_dependents_3_months
        ))
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Please check all inputs are valid")

# Footer
st.markdown("---")
st.markdown("*IFSSA Client Services - Predictive Analytics*")
