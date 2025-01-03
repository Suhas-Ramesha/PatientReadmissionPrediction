import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('./data/diabetes_readmission_model.pkl')

st.title('Patient Readmission Prediction')

# Input fields
age = st.number_input('Age')
gender = st.selectbox('Gender', ['Male', 'Female'])  # Add relevant features

if st.button('Predict'):
    input_data = pd.DataFrame({'age': [age], 'gender': [gender]})  # Adjust for your dataset
    prediction = model.predict(input_data)
    st.write(f'Prediction: {prediction[0]}')
