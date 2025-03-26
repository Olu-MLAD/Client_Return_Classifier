import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title="IFSSA Client Return Prediction",
    page_icon="📊"
)

# Load and Display Logos
col1, col2, _ = st.columns([0.15, 0.15, 0.7])
with col1:
    st.image("logo1.jpeg", width=120)
with col2:
    st.image("logo2.png", width=120)

# Colorful Header
st.markdown(
    """
    <h1 style='text-align: center; color: #ff5733; 
    background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
    Client Return Prediction App
    </h1>
    """,
    unsafe_allow_html=True
)

# ================== Navigation Bar ==================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["About the Project", "Exploratory Data Analysis", "Prediction"],
    index=2  # Default to Prediction page
)

# Add model info to sidebar if on Prediction page
if page == "Prediction":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    try:
        model = joblib.load("RF_model.pkl") if os.path.exists("RF_model.pkl") else None
        if model:
            st.sidebar.success("Model loaded successfully!")
            if hasattr(model, 'feature_names_in_'):
                st.sidebar.write(f"Input features required: {len(model.feature_names_in_)}")
        else:
            st.sidebar.error("Model not found")
    except Exception as e:
        st.sidebar.error(f"Model loading error: {str(e)}")

# ================== About the Project ==================
if page == "About the Project":
    st.markdown("""
    <h2 style='color: #33aaff; border-bottom: 2px solid #33aaff; padding-bottom: 10px;'>
    Introduction
    </h2>
    """, unsafe_allow_html=True)
    
    st.write(
        "The Islamic Family & Social Services Association (IFSSA) is a social service organization based in Edmonton, Alberta, Canada. "
        "It provides a range of community services, such as food hampers, crisis support, and assistance for refugees. "
        "The organization aims to use artificial intelligence to improve operations and enhance support efforts."
    )

    st.markdown("""
    <h2 style='color: #33aaff; border-bottom: 2px solid #33aaff; padding-bottom: 10px;'>
    Problem Statement
    </h2>
    """, unsafe_allow_html=True)
    
    st.write(
        "This project focuses on classifying clients to determine if they are likely to return within a 3-month period. "
        "By identifying client behavior patterns, IFSSA can enhance outreach efforts and optimize resource allocation."
    )

    # Add team/contact info
    st.markdown("---")
    st.markdown("### Project Team")
    st.write("For more information, please contact the IFSSA data team.")

# ================== Exploratory Data Analysis ==================
elif page == "Exploratory Data Analysis":
    st.markdown("""
    <h2 style='color: #33aaff; border-bottom: 2px solid #33aaff; padding-bottom: 10px;'>
    Exploratory Data Analysis (EDA)
    </h2>
    """, unsafe_allow_html=True)
    
    st.write("These charts show patterns in client return behavior based on historical data.")

    # Pre-generated Charts with better organization
    chart_paths = [f"chart{i}.png" for i in range(1, 8)]
    
    # Section for key metrics
    with st.expander("Key Metrics Summary", expanded=True):
        cols = st.columns(3)
        cols[0].metric("Return Rate", "63%", "5% YoY")
        cols[1].metric("Average Visits", "2.7", "-0.3 MoM")
        cols[2].metric("Peak Days", "Monday", "Weekends +15%")
    
    # Main charts
    st.write("### Detailed Analysis")
    cols = st.columns(2)
    for idx, chart_path in enumerate(chart_paths):
        with cols[idx % 2]:  
            st.image(
                chart_path, 
                caption=f"Chart {idx + 1}: {'Demographic' if idx%2 else 'Behavioral'} Patterns",
                use_container_width=True
            )
            if idx == len(chart_paths)-1 and len(chart_paths)%2:
                cols[1].write("")  # Empty space for alignment

# ================== Prediction Section ==================
elif page == "Prediction":
    st.markdown("""
    <h2 style='color: #33aaff; border-bottom: 2px solid #33aaff; padding-bottom: 10px;'>
    Prediction Section
    </h2>
    """, unsafe_allow_html=True)

    # Load Model with better error handling
    @st.cache_resource
    def load_model():
        try:
            if os.path.exists("RF_model.pkl"):
                model = joblib.load("RF_model.pkl")
                if hasattr(model, 'feature_names_in_'):
                    return model
                else:
                    st.error("Model is missing feature names information")
                    return None
            else:
                st.error("Model file not found at 'RF_model.pkl'")
                return None
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    model = load_model()

    # Input Features Section with better organization
    st.markdown("""
    <h3 style='color: #ff5733; border-bottom: 1px solid #ff5733; padding-bottom: 5px;'>
    Client Information
    </h3>
    """, unsafe_allow_html=True)
    
    # Use tabs for different feature categories
    tab1, tab2, tab3 = st.tabs(["Visit Information", "Client History", "Demographics"])

    with tab1:
        # Visit Information
        col1, col2 = st.columns(2)
        with col1:
            pickup_week = st.number_input("Pickup Week (1-52):", 
                                        min_value=1, max_value=52, 
                                        value=25, help="Week of the year when service was requested")
            
            holiday = st.radio("Is this pick-up during a holiday?", 
                             ["No", "Yes"], 
                             horizontal=True)
            
            Holidays = 1 if holiday == "Yes" else 0
            
            holiday_name = "None"
            if holiday == "Yes":
                holiday_name = st.selectbox(
                    "Select the holiday:",
                    [
                        "New Year's Day", "Good Friday", "Easter Monday", "Victoria Day",
                        "Canada Day", "Heritage Day", "Labour Day", "Thanksgiving Day",
                        "Remembrance Day", "Christmas Day", "Boxing Day", "Alberta Family Day",
                        "Mother's Day", "Father's Day"
                    ]
                )
        
        with col2:
            pickup_count_last_14_days = st.radio("Pickup in last 14 days?", 
                                               ["No", "Yes"], 
                                               horizontal=True)
            pickup_count_last_14_days = 1 if pickup_count_last_14_days == "Yes" else 0
            
            pickup_count_last_30_days = st.radio("Pickup in last 30 days?", 
                                              ["No", "Yes"], 
                                              horizontal=True)
            pickup_count_last_30_days = 1 if pickup_count_last_30_days == "Yes" else 0

    with tab2:
        # Client History
        col1, col2 = st.columns(2)
        with col1:
            time_since_first_visit = st.number_input("Time Since First Visit (days):", 
                                                    min_value=1, max_value=366, 
                                                    value=90,
                                                    help="Days since client's first visit")
            
            weekly_visits = st.number_input("Weekly Visits:", 
                                          min_value=0, value=1,
                                          help="Average weekly visits in past month")
        
        with col2:
            total_dependents_3_months = st.number_input("Total Dependents in Last 3 Months:", 
                                                      min_value=0, value=2,
                                                      help="Number of dependents served")

    with tab3:
        # Demographics
        postal_code = st.text_input("Postal Code (Canada format: A1A 1A1):",
                                  placeholder="T5J 2R1",
                                  help="First 3 characters sufficient for regional analysis")

    # Prepare input data
    input_dict = {
        'Holidays': Holidays,
        'holiday_name': holiday_name,
        'pickup_week': pickup_week,
        'pickup_count_last_14_days': pickup_count_last_14_days,
        'pickup_count_last_30_days': pickup_count_last_30_days,
        'time_since_first_visit': time_since_first_visit,
        'total_dependents_3_months': total_dependents_3_months,
        'weekly_visits': weekly_visits,
        'postal_code': postal_code
    }

    # Convert to DataFrame
    input_data = pd.DataFrame([input_dict])

    # Model validation and prediction
    if model:
        # Debug information in expander
        with st.expander("Show Feature Validation"):
            st.write("### Model Expectations")
            st.write(f"Model type: {type(model).__name__}")
            st.write(f"Expected features ({len(model.feature_names_in_)}):")
            st.write(model.feature_names_in_)
            
            st.write("### Provided Features")
            st.write(input_data.columns.tolist())
            
            missing = set(model.feature_names_in_) - set(input_data.columns)
            extra = set(input_data.columns) - set(model.feature_names_in_)
            
            if missing:
                st.error(f"Missing features: {list(missing)}")
            if extra:
                st.warning(f"Extra features provided: {list(extra)}")

        # Prediction button with better styling
        predict_col, _ = st.columns([0.2, 0.8])
        with predict_col:
            predict_btn = st.button("Predict Return Probability", 
                                  type="primary",
                                  use_container_width=True)

        if predict_btn:
            try:
                # Prepare final feature set
                final_features = pd.DataFrame(columns=model.feature_names_in_)
                
                # Handle holiday_name one-hot encoding
                if 'holiday_name' in model.feature_names_in_:
                    # Get all holidays the model knows about
                    holiday_cols = [col for col in model.feature_names_in_ 
                                   if col.startswith('holiday_name_')]
                    
                    # Create one-hot encoded columns
                    for col in holiday_cols:
                        holiday = col.replace('holiday_name_', '').replace('_', ' ')
                        final_features[col] = [1 if holiday_name == holiday else 0]
                
                # Copy other features
                for col in model.feature_names_in_:
                    if col in input_data.columns and col not in final_features.columns:
                        final_features[col] = input_data[col]
                    elif col not in final_features.columns:
                        final_features[col] = 0  # Fill missing with 0
                
                # Make prediction
                prediction = model.predict(final_features)
                proba = model.predict_proba(final_features)
                
                # Display results with better visualization
                st.markdown("---")
                result_col1, result_col2 = st.columns([0.3, 0.7])
                
                with result_col1:
                    st.markdown("""
                    <h3 style='color: #ff33aa; text-align: center;'>
                    Prediction Result
                    </h3>
                    """, unsafe_allow_html=True)
                    
                    # Visual indicator
                    if prediction[0] == 1:
                        st.success("""
                        <div style='text-align: center; font-size: 24px;'>
                        ✅ Likely to Return
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("""
                        <div style='text-align: center; font-size: 24px;'>
                        ❌ Unlikely to Return
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence meter
                    confidence = proba[0][prediction[0]]
                    st.metric("Confidence Level", f"{confidence:.1%}")
                    
                with result_col2:
                    # Probability breakdown
                    st.markdown("""
                    <h4 style='color: #33aaff;'>
                    Probability Breakdown
                    </h4>
                    """, unsafe_allow_html=True)
                    
                    prob_df = pd.DataFrame({
                        'Outcome': ['Will Return', 'Will Not Return'],
                        'Probability': proba[0]
                    })
                    
                    fig, ax = plt.subplots(figsize=(8, 3))
                    sns.barplot(data=prob_df, x='Outcome', y='Probability', 
                               palette=['#33aaff', '#ff5733'], ax=ax)
                    ax.set_ylim(0, 1)
                    ax.set_title("Return Probability Distribution")
                    st.pyplot(fig)
                    
                    # Key influencing factors
                    if hasattr(model, 'feature_importances_'):
                        st.markdown("""
                        <h4 style='color: #33aaff;'>
                        Top Influencing Factors
                        </h4>
                        """, unsafe_allow_html=True)
                        
                        importance_df = pd.DataFrame({
                            'Feature': model.feature_names_in_,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False).head(5)
                        
                        st.dataframe(importance_df.style.format({'Importance': '{:.2%}'}))
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.write("### Debug Information")
                st.write("Final features sent to model:")
                st.write(final_features.columns.tolist())
                st.write("Sample of feature values:")
                st.write(final_features.iloc[0].to_dict())
    else:
        st.warning("Please upload a valid model file to enable predictions")
