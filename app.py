import streamlit as st
import pandas as pd
import joblib

# Load model and encoders using joblib
model = joblib.load("extra_trees_credit_model.pkl")
encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking account"]}

# App title
st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict if the credit risk is good or bad: ")

# input fields
age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.select_box("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox["Saving Accounts", ["little", "moderate", "rich", "quite rich"]]
checking_account = st.selectbox["Checking Account", ["little", "moderate", "rich"]]
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

# Preparing input for the model
input_df = pd.DataFrame({
    "Age": [age], 
    "Sex": [encoders["Sex"].transform([sex])[0]], 
    "Job": [job], 
    "Housing": [encoders["Housing"].transform([housing])[0]], 
    "Saving Accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]], 
    "Checking Account": [encoders["Checking account"].transform([checking_account])[0]], 
    "Credit Amount": [credit_amount], 
    "Duration": [duration], 
})

