import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Load dataset
df = pd.read_csv("data/ngo_fake_data.csv")

# Process labels
# Modify the lambda function to handle NaN or non-string values gracefully
df['Supplies_Short'] = df['Supplies_Short'].apply(
    lambda x: x.split(', ') if isinstance(x, str) and x != 'None' else []
)

# Encode categorical features
le_location = LabelEncoder()
df['NGO_Location'] = le_location.fit_transform(df['NGO_Location'])

le_crisis = LabelEncoder()
df['Crisis_Level'] = le_crisis.fit_transform(df['Crisis_Level'])

X = df.drop(columns=['Supplies_Short'])

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Supplies_Short'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
multi_model = MultiOutputClassifier(base_model)
multi_model.fit(X_train, y_train)

# Save model and encoders
pipeline = {
    "model": multi_model,
    "scaler": scaler,
    "mlb": mlb,
    "le_location": le_location,
    "le_crisis": le_crisis
}
joblib.dump(pipeline, "model/supply_model.pkl")
print("Model saved successfully.")
