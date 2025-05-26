# 📈 Customer Churn Prediction with MLflow

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
python predict.py
```

6. Evaluate manually:
```bash
python evaluate.py
```