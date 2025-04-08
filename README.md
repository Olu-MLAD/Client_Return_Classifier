<p align="center" draggable="false"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR8HNB-ex4xb4H3-PXRcywP5zKC_3U8VzQTPA&usqp=CAU" 
     width="200px"
     height="auto"/>
</p>

# <h1 align="center" id="heading">IFSSA Client Return Prediction</h1>

### PROJECT TITLE: IFSSA Client Return Prediction

Welcome to the repository for our Capstone project at Norquest College. This project aims to predict which clients of the Islamic Family and Social Services Association (IFSSA) will return for services within 3 months using machine learning.

### Problem Statement

The IFSSA (Islamic Family and Social Services Association) struggles to predict when and how many clients will return to get hampers, leading to challenges in inventory management, resource allocation, and client retention strategies. This uncertainty affects operational efficiency and limits the ability to tailor the organization’s efforts effectively.

### Solution

We developed a Random Forest classifier that:
- Processes 6 key client visit features
- Provides binary predictions (1=return, 0=won't return)
- Includes explainable AI insights

### Live Application

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clientreturnclassifier.streamlit.app/)

### Repository Structure

The repository contains:
- `RF_model_streamlit.pkl`: Trained prediction model
- `app.py`: Streamlit web application
- `requirements.txt`: Python dependencies
- `data/`:IFSSA dataset
- `images/`: Logos and visualizations

### Team Members
Our team consists of the following members:

[Gift Ajayi](https://www.linkedin.com/in/gift-ajayi-036329ba/)
[Moyosore Akinola](http://linkedin.com/in/moyosore-akinola2022/)
[Abiola Bakare](https://www.linkedin.com/in/theabiolabakare/)
[Oluwaseun Ademokun](https://www.linkedin.com/in/oluwaseun-ademokun/)
[Abi Afolabi](http://linkedin.com/in/abimbola-adebowale-25a3a135/)

### Getting Started

1. Clone repository:
```bash
git clone [repository-url]
cd ifssa-return-predictor

