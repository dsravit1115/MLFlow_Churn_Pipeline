#  Customer Churn Prediction with MLflow
A production-ready customer churn prediction system using Scikit-learn and MLflow.
Includes model training with experiment logging, detailed evaluation reports, and an interactive Streamlit app to predict churn based on user input.
Emphasizes reproducibility, deployment-readiness, and modular ML pipeline architecture.



This project demonstrates a complete machine learning pipeline using Scikit-learn and MLflow. It includes training, logging parameters and metrics, and evaluating models.

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start MLflow UI:
```bash
mlflow ui
```
Open `http://localhost:5000` in your browser.

3. Train the model:
```bash
python train.py
```

4. View metrics and logged model in MLflow UI.

5. Predict using logged model:
```bash
python app.py
```

6. Evaluate manually:
```bash
python evaluate.py
```