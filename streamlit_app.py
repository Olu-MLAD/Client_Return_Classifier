import streamlit as st
import pandas as pd
import joblib
import os 

# Load Model Function
def load_model():
    model_path = "models/model.pkl"  # Update if needed
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Load Data Function
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Streamlit UI
st.title("Client Retention Prediction App")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    st.write("### Data Summary")
    st.write(df.describe())
    
    st.write("### Missing Values")
    st.write(df.isnull().sum())
    
    model = load_model()
    if model:
        st.sidebar.header("Make Predictions")
        if st.sidebar.button("Predict"):
            X = df.drop(columns=['target'])  # Adjust column names as needed
            predictions = model.predict(X)
            df['Prediction'] = predictions
            st.write("### Predictions")
            st.dataframe(df[['Prediction']])
    else:
        st.warning("No trained model found. Please upload a trained model to 'models/model.pkl'.")

