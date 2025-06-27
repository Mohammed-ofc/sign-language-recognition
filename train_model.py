import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load data
df = pd.read_csv("data/sign_data.csv", header=None)

# Features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the MLP classifier
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[INFO] Model Accuracy: {acc:.2f}")
print(classification_report(y_test, y_pred))

# Save the model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/sign_model.pkl")
print("[INFO] Model saved to model/sign_model.pkl")
