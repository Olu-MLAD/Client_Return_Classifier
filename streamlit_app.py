import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(layout="wide", page_title="Client Retention Prediction", page_icon="📊")

# Custom CSS for improved styling
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        color: #FF5733;
        font-size: 36px;
        font-weight: bold;
    }
    .section-title {
        color: #001F3F;
        font-size: 28px;
        font-weight: bold;
    }
    .sidebar-title {
        color: #FF5733;
        font-size: 22px;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #001F3F;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
    }
    .prediction-result {
        color: #FF5733;
        font-size: 24px;
        font-weight: bold;
    }
    .section-background {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and Display Logos Side by Side
col1, col2, _ = st.columns([0.15, 0.15, 0.7])  
with col1:
    st.image("logo1.jpeg", width=120)  
with col2:
    st.image("logo2.png", width=120)

st.markdown("<h1 class='main-title'>Client Retention Prediction App (MVP)</h1>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("<h2 class='sidebar-title'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Go to",
    ["About the Project", "Exploratory Data Analysis", "Prediction"]
)

# ================== About the Project ==================
if page == "About the Project":
    st.markdown("<div class='section-background'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Introduction</h2>", unsafe_allow_html=True)
    st.write(
        "The Islamic Family & Social Services Association (IFSSA) is a social service organization based in Edmonton, Alberta, Canada. "
        "It provides a range of community services, such as food hampers, crisis support, and assistance for refugees. "
        "The organization aims to use artificial intelligence to improve their operations and efficiently tailor their efforts to support "
        "the community by addressing challenges faced in the areas of inventory management, resource allocation, and delayed/inconsistent "
        "information shared with stakeholders."
    )
    st.markdown("<h2 class='section-title'>Problem Statement</h2>", unsafe_allow_html=True)
    st.write(
        "This project aims to classify clients based on whether they are likely to return to use IFSSA services within a 3-month timeframe. "
        "By identifying client behavior patterns, IFSSA can improve outreach efforts and optimize resource planning."
    )
    st.markdown("<h2 class='section-title'>Project Goals</h2>", unsafe_allow_html=True)
    st.write("✅ Identify patterns in client behavior for data-driven decision-making.")
    st.write("✅ Develop a predictive model to forecast client return likelihood.")
    st.write("✅ Enhance resource allocation and operational efficiency.")
    st.markdown("</div>", unsafe_allow_html=True)

# ================== Exploratory Data Analysis ==================
elif page == "Exploratory Data Analysis":
    st.markdown("<div class='section-background'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.write("Exploring the dataset to understand structure, patterns, and insights.")
    
    # Display Pre-generated Charts
    chart_paths = [f"chart{i}.png" for i in range(1, 8)]
    cols = st.columns(2)
    for idx, chart_path in enumerate(chart_paths):
        with cols[idx % 2]:
            st.image(chart_path, caption=f"Chart {idx + 1}", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ================== Prediction ==================
elif page == "Prediction":
    st.markdown("<div class='section-background'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Prediction</h2>", unsafe_allow_html=True)
    
    def load_model():
        model_path = "models/model.pkl"
        return joblib.load(model_path) if os.path.exists(model_path) else None
    
    model = load_model()
    if model is None:
        st.error("⚠️ No trained model found. Please upload a trained model to 'models/model.pkl'.")
    
    st.sidebar.markdown("<h2 class='sidebar-title'>Input Features</h2>", unsafe_allow_html=True)
    
    inputs = {
        "Time Since Last Pickup": (0, 10),
        "Hamper Confirmation Type": (0, 1),
        "Preferred Contact Methods": (0, 1),
        "Client Status": (0, 1),
        "Sex": (0, 1),
        "Age in Years": (0, 35),
        "Hamper Demand Lag 30 Days": (0, 2),
        "Latest Contact Method": (0, 1),
        "Dependents Quantity": (0, 3),
        "Household Size": (0, 4),
        "Contact Frequency": (0, 5),
    }
    
    input_data = pd.DataFrame(
        [[st.sidebar.number_input(label, min_value=values[0], value=values[1]) for label, values in inputs.items()]],
        columns=[label.replace(" ", "_").lower() for label in inputs.keys()]
    )
    
    if st.sidebar.button("🎯 Predict"):
        if model is None:
            st.error("❌ No trained model found. Please upload a valid model.")
        else:
            prediction = model.predict(input_data)
            st.markdown("<h3 class='prediction-result'>🎉 Predicted Outcome: {}</h3>".format(int(prediction[0])), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
