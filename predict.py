import mlflow.sklearn
import pandas as pd

# Example input (should match training format)
data = pd.DataFrame({
    "gender_Female": [1],
    "gender_Male": [0],
    "age": [35],
    "income": [58000]
})

# Load model (latest version from MLflow registry)
model = mlflow.sklearn.load_model("models:/ChurnPrediction/1")
pred = model.predict(data)
print("Churn prediction:", pred[0])