# python -m streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load Model
with open('knn_credit_score_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load Encoders
le_gender = LabelEncoder()
le_education = LabelEncoder()
le_marital_status = LabelEncoder()
le_home_ownership = LabelEncoder()
le_credit_score = LabelEncoder()

# Define classes for each encoder
le_gender.classes_ = np.array(['Female', 'Male'])
le_education.classes_ = np.array(['Associate\'s Degree', 'Bachelor\'s Degree', 'Doctorate', 'High School Diploma', 'Master\'s Degree'])
le_marital_status.classes_ = np.array(['Married', 'Single'])
le_home_ownership.classes_ = np.array(['Owned', 'Rented'])
le_credit_score.classes_ = np.array(['Average', 'High', 'Low'])

st.title('Credit Score Classification')

# Input Fields
age = st.number_input('Age', min_value=18, max_value=100, value=30)
gender = st.selectbox('Gender', le_gender.classes_)
income = st.number_input('Income', min_value=0)
education = st.selectbox('Education', le_education.classes_)
marital_status = st.selectbox('Marital Status', le_marital_status.classes_)
number_of_children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
home_ownership = st.selectbox('Home Ownership', le_home_ownership.classes_)

# Encode Input
gender_encoded = le_gender.transform([gender])[0]
education_encoded = le_education.transform([education])[0]
marital_status_encoded = le_marital_status.transform([marital_status])[0]
home_ownership_encoded = le_home_ownership.transform([home_ownership])[0]

input_data = np.array([[age, gender_encoded, income, education_encoded, marital_status_encoded, number_of_children, home_ownership_encoded]])

# Scale Input
input_data_scaled = scaler.transform(input_data)

# Predict
if st.button('Predict'):
    prediction = knn_model.predict(input_data_scaled)
    credit_score = le_credit_score.inverse_transform(prediction)[0]
    st.write(f'The predicted credit score is: {credit_score}')
