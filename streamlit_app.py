import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.base import is_classifier

# Basic app setup
st.set_page_config(layout="wide", page_title="IFSSA Predictor")

@st.cache_resource
def load_model():
    try:
        model = joblib.load("RF_model_streamlit.pkl")
        if not is_classifier(model):
            raise ValueError("Model is not a classifier")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

def main():
    st.title("IFSSA Predictor")
    model = load_model()
    
    if model:
        st.success("Model loaded successfully!")
        # Add your app pages here

if __name__ == "__main__":
    main()import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.base import is_classifier

# Basic app setup
st.set_page_config(layout="wide", page_title="IFSSA Predictor")

@st.cache_resource
def load_model():
    try:
        model = joblib.load("RF_model_streamlit.pkl")
        if not is_classifier(model):
            raise ValueError("Model is not a classifier")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

def main():
    st.title("IFSSA Predictor")
    model = load_model()
    
    if model:
        st.success("Model loaded successfully!")
        # Add your app pages here

if __name__ == "__main__":
    main()
