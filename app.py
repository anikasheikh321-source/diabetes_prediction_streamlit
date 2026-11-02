
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("Diabetes Prediction App")

st.write("Enter the details below:")

# User inputs
Pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
Glucose = st.number_input("Glucose", min_value=1, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=1, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=1, value=20)
Insulin = st.number_input("Insulin", min_value=1, value=79)
BMI = st.number_input("BMI", min_value=1.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
Age = st.number_input("Age", min_value=1, value=30)

# Prediction button
if st.button("Predict"):

    input_data = np.array([[
        Pregnancies, Glucose, BloodPressure,
        SkinThickness, Insulin, BMI,
        DiabetesPedigreeFunction, Age
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Prediction: Diabetes")
    else:
        st.success("Prediction: No Diabetes")

    st.warning("This is an educational prediction, not a medical diagnosis.")
