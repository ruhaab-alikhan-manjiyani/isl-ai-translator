import pandas as pd

# Load CSV
data = pd.read_csv("gesture_data.csv", header=None)

# Show labels
print("Labels found:")
print(data.iloc[:, -1].value_counts())

# Ask which label to delete
label_to_delete = input("Enter label to delete: ")

# Remove rows
data = data[data.iloc[:, -1] != label_to_delete]

# Save updated CSV
data.to_csv("gesture_data.csv", header=False, index=False)

print(f"{label_to_delete} deleted successfully.")