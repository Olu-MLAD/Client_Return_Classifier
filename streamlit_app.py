import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title="IFSSA 3-Month Return Predictor",
    page_icon="🔮"
)

# Load and Display Logos
col1, col2, _ = st.columns([0.15, 0.15, 0.7])
with col1:
    st.image("logo1.jpeg", width=120)
with col2:
    st.image("logo2.png", width=120)

# Mission-focused Header
st.markdown(
    """
    <h1 style='text-align: center; color: #ff5733; 
    background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
    IFSSA 3-Month Client Return Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

# ================== Navigation Bar ==================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Project Overview", "Client Insights", "Return Prediction", "Resource Planner"],
    index=2  # Default to Prediction page
)

# ================== Project Overview ==================
if page == "Project Overview":
    st.markdown("""
    <h2 style='color: #33aaff; border-bottom: 2px solid #33aaff; padding-bottom: 10px;'>
    Project Mission
    </h2>
    """, unsafe_allow_html=True)
    
    st.write(
        "This predictive analytics tool helps IFSSA identify which clients are most likely to return "
        "for services within 3 months, enabling proactive outreach and optimized resource allocation."
    )
    
    # Key objectives
    st.markdown("""
    <h3 style='color: #33aaff;'>Key Objectives</h3>
    <ul>
    <li>Predict 3-month client return probability with 85%+ accuracy</li>
    <li>Identify high-need clients for targeted outreach</li>
    <li>Optimize food hamper inventory and staff scheduling</li>
    <li>Reduce client churn through data-driven interventions</li>
    </ul>
    """, unsafe_allow_html=True)
    
    # Methodology
    st.markdown("""
    <h3 style='color: #33aaff;'>Methodology</h3>
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
    <p>Our machine learning model analyzes:</p>
    <ol>
    <li><b>Visit patterns</b> - Frequency, recency, and seasonality</li>
    <li><b>Client characteristics</b> - Family size, location, and needs</li>
    <li><b>Service history</b> - Types of assistance previously received</li>
    <li><b>External factors</b> - Holidays, weather, and economic conditions</li>
    </ol>
    <p>The model outputs a return probability score and key influencing factors.</p>
    </div>
    """, unsafe_allow_html=True)

# ================== Client Insights ==================
elif page == "Client Insights":
    st.markdown("""
    <h2 style='color: #33aaff; border-bottom: 2px solid #33aaff; padding-bottom: 10px;'>
    Client Behavior Patterns
    </h2>
    """, unsafe_allow_html=True)
    
    # Dynamic date selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", 
                                 value=datetime.now() - timedelta(days=365),
                                 max_value=datetime.now())
    with col2:
        end_date = st.date_input("End date", 
                               value=datetime.now(),
                               max_value=datetime.now())
    
    # Key metrics
    st.markdown("### 3-Month Return Rates")
    cols = st.columns(4)
    cols[0].metric("Overall Return Rate", "62%", "3% YoY")
    cols[1].metric("New Clients", "41%", "-2% YoY")
    cols[2].metric("Repeat Clients", "78%", "5% YoY")
    cols[3].metric("High-Need Clients", "85%", "8% YoY")
    
    # Visualizations
    st.markdown("### Return Pattern Analysis")
    tab1, tab2, tab3 = st.tabs(["By Neighborhood", "By Service Type", "By Family Size"])
    
    with tab1:
        # Mock data - replace with actual data
        neighborhood_data = pd.DataFrame({
            'Area': ['North', 'South', 'East', 'West', 'Central'],
            'Return Rate': [0.55, 0.68, 0.72, 0.61, 0.59],
            'Visits': [120, 85, 92, 78, 105]
        })
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=neighborhood_data, x='Area', y='Return Rate', palette='viridis')
        ax.set_title("Return Rates by Neighborhood")
        st.pyplot(fig)
    
    with tab2:
        service_data = pd.DataFrame({
            'Service': ['Food', 'Clothing', 'Counseling', 'Employment', 'Housing'],
            'Return Rate': [0.65, 0.58, 0.42, 0.71, 0.63]
        })
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=service_data, x='Return Rate', y='Service', palette='rocket')
        ax.set_title("Return Rates by Service Type")
        st.pyplot(fig)
    
    with tab3:
        family_data = pd.DataFrame({
            'Family Size': ['1', '2-3', '4-5', '6+'],
            'Return Rate': [0.48, 0.67, 0.72, 0.81]
        })
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=family_data, x='Family Size', y='Return Rate', 
                    marker='o', linewidth=2.5)
        ax.set_title("Return Rates by Family Size")
        st.pyplot(fig)

# ================== Return Prediction ==================
elif page == "Return Prediction":
    st.markdown("""
    <h2 style='color: #33aaff; border-bottom: 2px solid #33aaff; padding-bottom: 10px;'>
    3-Month Return Prediction
    </h2>
    """, unsafe_allow_html=True)
    
    # Load Model with enhanced caching
    @st.cache_resource
    def load_model():
        try:
            if os.path.exists("RF_model.pkl"):
                model = joblib.load("RF_model.pkl")
                if hasattr(model, 'feature_names_in_'):
                    return model
                raise AttributeError("Model missing feature names")
            raise FileNotFoundError("Model file not found")
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
            return None

    model = load_model()
    
    if model:
        st.success("✅ Predictive model loaded successfully")
        
        # Client Information Section
        with st.form("client_info"):
            st.markdown("""
            <h3 style='color: #ff5733; border-bottom: 1px solid #ff5733; padding-bottom: 5px;'>
            Client Details
            </h3>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Basic info
                client_id = st.text_input("Client ID*")
                last_visit = st.date_input("Last Visit Date*", 
                                         value=datetime.now() - timedelta(days=30))
                visit_frequency = st.selectbox(
                    "Typical Visit Frequency*",
                    ["Weekly", "Bi-Weekly", "Monthly", "Quarterly", "First Time"]
                )
                
                # Family information
                dependents = st.number_input("Number of Dependents*", 
                                           min_value=0, max_value=10, value=2)
                housing_status = st.selectbox(
                    "Housing Status",
                    ["Stable", "Temporary", "Unstable", "Homeless"]
                )
            
            with col2:
                # Visit details
                services = st.multiselect(
                    "Services Used*",
                    ["Food Hamper", "Clothing", "Counseling", 
                     "Employment Support", "Housing Assistance"],
                    default=["Food Hamper"]
                )
                
                # External factors
                holiday_visit = st.checkbox("Visit during holiday period")
                temperature = st.slider(
                    "Temperature at Visit (°C)", 
                    min_value=-30, max_value=30, value=10
                )
                
                # Special circumstances
                crisis_support = st.checkbox("Received crisis support")
                referral_source = st.selectbox(
                    "Referral Source",
                    ["Self", "Community Org", "Government", "Healthcare", "Other"]
                )
            
            # Form submission
            submitted = st.form_submit_button("Predict Return Probability", type="primary")
        
        if submitted and client_id:
            try:
                # Prepare feature vector (mock - adapt to your actual model)
                features = {
                    'days_since_last_visit': (datetime.now().date() - last_visit).days,
                    'visit_frequency_encoded': ["Weekly", "Bi-Weekly", "Monthly", "Quarterly", "First Time"].index(visit_frequency),
                    'num_dependents': dependents,
                    'num_services': len(services),
                    'holiday_visit': int(holiday_visit),
                    'crisis_support': int(crisis_support),
                    'temperature': temperature
                }
                
                # Mock prediction - replace with actual model prediction
                return_prob = 0.65  # Replace with model.predict_proba()
                prediction = return_prob > 0.5
                
                # Display results
                st.markdown("---")
                st.markdown("""
                <h3 style='color: #ff33aa; text-align: center;'>
                Prediction Results
                </h3>
                """, unsafe_allow_html=True)
                
                # Result columns
                col1, col2 = st.columns([0.4, 0.6])
                
                with col1:
                    # Prediction card
                    prediction_card = st.container()
                    prediction_card.markdown("""
                    <div style='background-color: #f0f2f6; border-radius: 10px; padding: 20px;'>
                    <h4 style='color: #333; text-align: center;'>3-Month Return Prediction</h4>
                    """, unsafe_allow_html=True)
                    
                    if prediction:
                        prediction_card.success("""
                        <div style='text-align: center; font-size: 24px;'>
                        🎯 Likely to Return ({(return_prob*100):.1f}%)
                        </div>
                        """, unsafe_allow_html=True)
                        prediction_card.write("**Recommended Action:** Schedule follow-up in 2 months")
                    else:
                        prediction_card.error("""
                        <div style='text-align: center; font-size: 24px;'>
                        ⚠️ Unlikely to Return ({(return_prob*100):.1f}%)
                        </div>
                        """, unsafe_allow_html=True)
                        prediction_card.write("**Recommended Action:** Proactive outreach needed")
                    
                    prediction_card.markdown("</div>", unsafe_allow_html=True)
                    
                    # Key factors
                    st.markdown("""
                    <h4 style='color: #33aaff;'>Key Influencing Factors</h4>
                    <ul>
                    <li>Visit frequency: {visit_frequency}</li>
                    <li>Days since last visit: {(datetime.now().date() - last_visit).days}</li>
                    <li>Number of dependents: {dependents}</li>
                    <li>Services used: {", ".join(services)}</li>
                    </ul>
                    """.format(**locals()), unsafe_allow_html=True)
                
                with col2:
                    # Probability visualization
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(['Return', 'Not Return'], 
                           [return_prob, 1-return_prob], 
                           color=['#33aaff', '#ff5733'])
                    ax.set_xlim(0, 1)
                    ax.set_title('Return Probability Distribution')
                    st.pyplot(fig)
                    
                    # Action plan
                    st.markdown("""
                    <h4 style='color: #33aaff;'>Suggested Outreach Plan</h4>
                    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px;'>
                    <p><b>Timing:</b> {optimal_timing}</p>
                    <p><b>Channel:</b> {optimal_channel}</p>
                    <p><b>Message:</b> {message_content}</p>
                    </div>
                    """.format(
                        optimal_timing="2-4 weeks before predicted return window" if prediction else "Immediate contact recommended",
                        optimal_channel="Phone call" if dependents > 3 else "Text message",
                        message_content="Check if additional support needed for family" if dependents > 0 else "Inform about new services"
                    ), unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# ================== Resource Planner ==================
elif page == "Resource Planner":
    st.markdown("""
    <h2 style='color: #33aaff; border-bottom: 2px solid #33aaff; padding-bottom: 10px;'>
    Resource Allocation Planner
    </h2>
    """, unsafe_allow_html=True)
    
    st.write("This tool helps plan resources based on predicted client returns")
    
    # Time horizon selection
    horizon = st.slider(
        "Planning Horizon (weeks)", 
        min_value=1, max_value=12, value=4
    )
    
    # Mock data - replace with actual predictions
    return_data = pd.DataFrame({
        'Week': [f"Week {i}" for i in range(1, horizon+1)],
        'Predicted Returns': [25, 32, 28, 35][:horizon],
        'Food Hampers Needed': [38, 45, 42, 50][:horizon],
        'Staff Hours Required': [120, 140, 130, 150][:horizon]
    })
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    return_data.set_index('Week').plot(kind='bar', ax=ax)
    ax.set_title(f"Predicted Resource Needs for Next {horizon} Weeks")
    ax.legend(loc='upper right')
    st.pyplot(fig)
    
    # Detailed planning
    st.markdown("### Detailed Resource Planning")
    st.dataframe(return_data.style.format({
        'Predicted Returns': '{:.0f}',
        'Food Hampers Needed': '{:.0f}',
        'Staff Hours Required': '{:.0f}'
    }))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
<p>IFSSA Client Return Prediction System | Data Last Updated: {date}</p>
</div>
""".format(date=datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
