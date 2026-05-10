import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("gesture_data.csv", header=None)

# Features (hand landmark coordinates)
X = data.iloc[:, :-1]

# Labels (gesture names)
y = data.iloc[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

# Save trained model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")