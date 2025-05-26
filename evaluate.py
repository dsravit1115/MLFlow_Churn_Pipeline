import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv("data/churn_data.csv")

# Prepare features and labels
X = pd.get_dummies(df.drop(columns=["customer_id", "churned"]))
y = df["churned"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation report
print("Classification Report:")
print(classification_report(y_test, y_pred))


