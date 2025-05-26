import pandas as pd
import streamlit as st
import mlflow.sklearn

# Load model from MLflow registry or saved model path
try:
    model = mlflow.sklearn.load_model("models:/ChurnPrediction/1")
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
    input_data = pd.DataFrame([{
        "gender_Female": 1 if gender == "Female" else 0,
        "gender_Male": 1 if gender == "Male" else 0,
        "age": age,
        "income": income
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    label = "Churned" if prediction == 1 else "Not Churned"
    st.success(f"Prediction: {label}")
