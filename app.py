import pandas as pd
import streamlit as st
import mlflow.sklearn

# Load model from MLflow registry or saved model path
try:
     
    model = mlflow.sklearn.load_model("D:/Azure/GENAI/EXP/mlflow-churn-pipeline/mlruns/450911256782553228/53f9dbc8e77d4c78b7c10ea26ce00650/artifacts/model")

except:
    st.error(" Could not load model from MLflow registry.")
    st.stop()

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title(" Customer Churn Prediction App")

# User input form
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 80, 30)
    income = st.number_input("Annual Income", value=50000)
    submitted = st.form_submit_button("Predict")

# Convert user input to model input
if submitted:
    # Define full feature set as during training
    #feature_columns = ["age", "income", "gender_Female", "gender_Male"]

# Construct input row in the correct order
    # Build input row with exact feature order and names
    input_data = pd.DataFrame([[
    age,
    income,
    1 if gender == "Female" else 0,  # gender_Female
    1 if gender == "Male" else 0     # gender_Male
    ]], columns=["age", "income", "gender_Female", "gender_Male"])

    # Create DataFrame with correct column names
    #input_data = pd.DataFrame([input_data], columns=feature_columns)
    # Predict
    prediction = model.predict(input_data)[0]
    label = "Churned" if prediction == 1 else "Not Churned"
    st.success(f"Prediction: {label}")
