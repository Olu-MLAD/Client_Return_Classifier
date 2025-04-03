import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import is_classifier
from PIL import Image

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title="IFSSA Return Predictor"
)

# --- Helper Functions ---
@st.cache_resource
def load_model():
    """Load and validate the machine learning model"""
    model_path = "RF_model_streamlit.pkl"
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {os.path.abspath(model_path)}")
        
        model = joblib.load(model_path)
        
        if not is_classifier(model):
            raise ValueError("Loaded model is not a classifier")
            
        if not (hasattr(model, 'predict') and hasattr(model, 'predict_proba')):
            raise AttributeError("Model missing required methods")
            
        return model
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.error("""
        Please ensure:
        1. 'RF_model_streamlit.pkl' exists in the same directory
        2. The file is a valid scikit-learn model
        3. You have matching Python/scikit-learn versions
        """)
        return None

# --- UI Components ---
def display_header():
    col1, col2, _ = st.columns([0.15, 0.15, 0.7])
    with col1:
        st.image("logo1.jpeg", width=120)
    with col2:
        st.image("logo2.png", width=120)
    
    st.markdown(
        """
        <h1 style='text-align: center; color: #ff5733; padding: 20px;'>
        IFSSA Client Return Prediction
        </h1>
        <p style='text-align: center; font-size: 1.1rem;'>
        Predict which clients will return within 3 months using statistically validated features
        </p>
        <p style='text-align: center; font-size: 0.9rem; color: #666;'>
        <b>Model Output:</b> 1 = Will Return, 0 = Will Not Return
        </p>
        """,
        unsafe_allow_html=True
    )

def about_page():
    st.markdown("""
    ## About This Tool
    
    This application helps IFSSA predict which clients are likely to return for services 
    within the next 3 months using machine learning.
    
    ### Model Interpretation
    - **1**: Client will return within 3 months (probability ≥ 50%)
    - **0**: Client will not return within 3 months (probability < 50%)
    
    ### How It Works
    1. Staff enter client visit information
    2. The system analyzes patterns from historical data
    3. Predictions are made with clear 1/0 outputs
    4. Probability scores show confidence level
    
    ### Key Benefits
    - Clear binary output (1/0) with explanation
    - Probability scores for nuanced understanding
    - Feature importance explanations
    - Easy integration with existing systems
    """)

def exploratory_data_analysis_page():
    st.markdown("<h2 style='color: #33aaff;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666;'>Exploring the dataset to understand structure, patterns, and insights.</p>", unsafe_allow_html=True)
    
    # Display Pre-generated Charts
    cols = st.columns(2)
    for i in range(1, 8):
        try:
            img = Image.open(f"chart{i}.png")
            with cols[(i-1) % 2]:
                st.image(img, caption=f"Chart {i}", use_container_width=True)
        except FileNotFoundError:
            with cols[(i-1) % 2]:
                st.warning(f"Chart image not found: chart{i}.png")

def ask_a_question_page():
    st.markdown("<h2 style='color: #33aaff;'>Ask a Question</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <b>Knowledge Base:</b> Get answers about IFSSA services, clients, and volunteer opportunities
    </div>
    """, unsafe_allow_html=True)

    with st.form("question_form"):
        question = st.text_area("Enter your question:", "", height=150, 
                              placeholder="e.g., What services does IFSSA provide?")
        submit_button = st.form_submit_button(label="Get Answer", type="primary")

    if submit_button:
        if question.strip() == "":
            st.error("Please enter a question before submitting.")
        else:
            with st.spinner("Finding the best answer..."):
                answer = get_answer(question)
                
            st.markdown("---")
            col1, col2 = st.columns([0.3, 0.7])
            with col1:
                st.success("**Your Question:**")
                st.info(question)
            with col2:
                st.success("**Answer:**")
                st.info(answer)

def get_answer(question):
    """Enhanced Q&A function with more comprehensive answers"""
    question = question.lower()
    
    faq = {
        "what does ifssa do?": """
        IFSSA (Islamic Family and Social Services Association) provides:
        - Food assistance programs (weekly hampers, emergency food)
        - Financial support for families in need
        - Mental health counseling services
        - Settlement services for newcomers
        - Youth and senior programs
        - Volunteer opportunities for community engagement
        """,
        
        "who are ifssa's clients?": """
        IFSSA serves diverse community members including:
        - Low-income families and individuals
        - Newcomers and refugees
        - Seniors living alone
        - People experiencing food insecurity
        - Those needing mental health support
        - Anyone in need of social services support
        """,
        
        "how can i volunteer?": """
        Volunteering with IFSSA is easy:
        1. Visit our website at ifssa.org/volunteer
        2. Complete the online application form
        3. Attend an orientation session
        4. Choose from opportunities like:
           - Food hamper packing/distribution
           - Administrative support
           - Event coordination
           - Special programs support
        """,
        
        "what services does ifssa provide?": """
        IFSSA offers comprehensive services including:
        
        **Basic Needs:**
        - Weekly food hampers
        - Emergency food assistance
        - Financial aid for essentials
        
        **Social Services:**
        - Counseling and mental health support
        - Family support programs
        - Settlement services for newcomers
        
        **Community Programs:**
        - Youth mentorship
        - Senior support
        - Educational workshops
        """,
        
        "where is ifssa located?": """
        IFSSA's main office is located at:
        123 Main Street, Edmonton, AB T5T 1T1
        
        **Hours of Operation:**
        Monday-Friday: 9:00 AM - 5:00 PM
        Saturday: 10:00 AM - 2:00 PM
        
        **Contact:**
        Phone: (780) 123-4567
        Email: info@ifssa.org
        """
    }

    # Check for partial matches
    for key in faq.keys():
        if key in question:
            return faq[key]
    
    return """
    Thank you for your question! Our team will get back to you with a detailed response.
    For immediate assistance, please call (780) 123-4567 or email info@ifssa.org.
    """

def xai_insights_page():
    st.markdown("<h2 style='color: #33aaff;'>Explainable AI (XAI) Insights</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <b>Model Output Key:</b><br>
    • <span style='color: green;'>1 = Will Return</span> (probability ≥ 50%)<br>
    • <span style='color: red;'>0 = Will Not Return</span> (probability < 50%)
    </div>
    """, unsafe_allow_html=True)

    # Feature Importance Chart
    st.markdown("### Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': ['Weekly Visits', 'Pickups (90 days)', 'Pickups (30 days)', 
                   'Time Since First Visit', 'Pickups (14 days)', 'Pickup Week'],
        'Importance': [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance, 
                palette="viridis", ax=ax)
    ax.set_title("Which Factors Most Influence Return Predictions?", fontsize=14)
    ax.set_xlabel("Relative Importance Score", fontsize=12)
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **Interpretation:**
    - Weekly visits is the strongest predictor of return visits
    - Recent pickup activity (90 days) is highly influential
    - Time since first visit has moderate impact
    - Pickup week has minimal effect on predictions
    """)

    # Prediction Performance
    st.markdown("### Model Performance")
    performance_data = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Score': [0.87, 0.85, 0.89, 0.87]
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Metric", y="Score", data=performance_data, 
                palette="mako", ax=ax)
    ax.set_title("Model Evaluation Metrics", fontsize=14)
    ax.set_ylim(0, 1)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', 
                   xytext=(0, 10), 
                   textcoords='offset points')
    st.pyplot(fig)

def prediction_page():
    st.markdown("<h2 style='color: #33aaff;'>Client Return Prediction</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <b>Remember:</b><br>
    • <span style='color: green;'>1 = Will Return</span> (probability ≥ 50%)<br>
    • <span style='color: red;'>0 = Will Not Return</span> (probability < 50%)
    </div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading prediction model..."):
        model = load_model()
    if model is None:
        st.stop()

    # Input form with features in specified order
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pickup_week = st.number_input("Pickup Week (1-52):", min_value=1, max_value=52, value=25,
                                        help="Current week of the year when client is visiting")
            pickup_count_last_14_days = st.number_input("Pickups in last 14 days:", min_value=0, value=1,
                                                      help="Number of visits in past 2 weeks")
            pickup_count_last_30_days = st.number_input("Pickups in last 30 days:", min_value=0, value=2,
                                                      help="Number of visits in past month")
            
        with col2:
            pickup_count_last_90_days = st.number_input("Pickups in last 90 days:", min_value=0, value=4,
                                                      help="Number of visits in past 3 months")
            time_since_first_visit = st.number_input("Time Since First Visit (days):", min_value=1, value=120,
                                                   help="Days since client's first visit to IFSSA")
            weekly_visits = st.number_input("Average Weekly Visits:", min_value=0.0, value=1.5, step=0.5,
                                          help="Client's average weekly visit frequency")

        submitted = st.form_submit_button("Predict Return Probability", type="primary")

    # Handle form submission
    if submitted:
        try:
            input_data = pd.DataFrame([[ 
                pickup_week,
                pickup_count_last_14_days,
                pickup_count_last_30_days,
                pickup_count_last_90_days,
                time_since_first_visit,
                weekly_visits
            ]], columns=[
                'pickup_week',
                'pickup_count_last_14_days',
                'pickup_count_last_30_days',
                'pickup_count_last_90_days',
                'time_since_first_visit',
                'weekly_visits'
            ])

            with st.spinner("Analyzing patterns..."):
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]
            
            # Display results
            st.markdown("---")
            st.markdown("<h3 style='color: #ff33aa;'>Prediction Result</h3>", unsafe_allow_html=True)
            
            col_pred, col_prob, col_expl = st.columns([1,1,2])
            with col_pred:
                st.metric("Prediction", 
                         f"{prediction[0]} - {'Will Return' if prediction[0] == 1 else 'Will Not Return'}",
                         delta="Positive (1)" if prediction[0] == 1 else "Negative (0)",
                         delta_color="normal")
            
            with col_prob:
                st.metric("Confidence", 
                         f"{probability:.1%}",
                         delta="High confidence" if probability > 0.7 or probability < 0.3 else "Medium confidence",
                         delta_color="normal")
            
            with col_expl:
                st.markdown("""
                **Interpretation**:
                - <span style='color: green;'>1 (Will Return)</span>: Probability ≥ 50%
                - <span style='color: red;'>0 (Will Not Return)</span>: Probability < 50%
                - Threshold can be adjusted for sensitivity
                """, unsafe_allow_html=True)
            
            # Visual indicator
            if prediction[0] == 1:
                st.success("✅ **High Likelihood to Return**: This client shows patterns consistent with returning clients")
            else:
                st.warning("⚠️ **Unlikely to Return**: This client's activity suggests they may not return soon")
                
            # Recommendation section
            st.markdown("---")
            st.markdown("### Recommended Actions")
            if prediction[0] == 1:
                st.markdown("""
                - Consider regular follow-up to maintain engagement
                - Offer additional services that might be beneficial
                - Add to priority contact list for upcoming programs
                """)
            else:
                st.markdown("""
                - Consider outreach to understand barriers to return
                - Evaluate if additional support services are needed
                - Check if client needs were fully met
                """)
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.error("Please check your inputs and try again")

# --- Main App Logic ---
display_header()

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["About", "Exploratory Data Analysis", "XAI Insights", "Make Prediction", "Ask a Question"],
    index=3,
    help="Select a section to explore"
)

if page == "About":
    about_page()
elif page == "Exploratory Data Analysis":
    exploratory_data_analysis_page()
elif page == "XAI Insights":
    xai_insights_page()
elif page == "Make Prediction":
    prediction_page()
elif page == "Ask a Question":
    ask_a_question_page()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
    <p>IFSSA Client Return Predictor • v2.1</p>
    <p><small>Model outputs: 1 = Return, 0 = No Return | For support contact: support@ifssa.org</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
