import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
pipeline = joblib.load("../model/supply_model.pkl")

model = pipeline["model"]
scaler = pipeline["scaler"]
mlb = pipeline["mlb"]
le_location = pipeline["le_location"]
le_crisis = pipeline["le_crisis"]

# Set the title of the app
st.title("NGO Supply Shortage Predictor")

# Display an introductory paragraph
st.markdown("""
    *Welcome to the NGO Supply Shortage Predictor!*
    
    This tool helps predict the shortages of various supplies in an NGO based on factors such as the location, crisis level, and available resources. 
    Fill in the details below, and click "Predict Shortages" to see if there are any expected shortages.
""")

# Create two columns for better layout of inputs
col1, col2 = st.columns(2)

# Select NGO Location and Crisis Level
with col1:
    location = st.selectbox("NGO Location", le_location.classes_.tolist(), key="location")

with col2:
    crisis = st.selectbox("Crisis Level", le_crisis.classes_.tolist(), key="crisis")

# Input section for numerical data, organized in columns
with col1:
    days = st.number_input("Days Since Last Supply", min_value=0, max_value=30, value=5, key="days")
    people = st.number_input("People Count", min_value=0, max_value=2000, value=500, key="people")
    water = st.number_input("Water Stock (Liters)", min_value=0, max_value=5000, value=1000, key="water")

with col2:
    food = st.number_input("Food Stock (Kg)", min_value=0, max_value=5000, value=1000, key="food")
    medicine = st.number_input("Medicine Units", min_value=0, max_value=1000, value=300, key="medicine")
    blankets = st.number_input("Blankets Count", min_value=0, max_value=1000, value=400, key="blankets")
    fuel = st.number_input("Fuel (Liters)", min_value=0, max_value=2000, value=200, key="fuel")

# Add an icon and description above the prediction button
st.markdown("""
    ### Ready to predict shortages? 
    Press the button below to get your prediction.
""")

# Prediction button with loading animation
if st.button("🔍 Predict Shortages"):
    with st.spinner("Making the prediction..."):
        input_data = np.array([[le_location.transform([location])[0],
                                le_crisis.transform([crisis])[0],
                                days, people, water, food, medicine, blankets, fuel]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        predicted_labels = mlb.inverse_transform(prediction)

        # Display the prediction
        if predicted_labels and predicted_labels[0]:
            st.warning(f"⚠ *Predicted shortages*: {', '.join(predicted_labels[0])}")
        else:
            st.success("✅ *No supply shortages predicted*.")

# Load the dataset for visualization
sample_data = pd.read_csv("../data/ngo_fake_data.csv")

# Generate a simple bar plot for average stock levels by location
st.markdown("""
    ### 📊 Average Stock Levels by Location
""")
plt.figure(figsize=(10, 6))
stock_columns = ["Water_Stock_Liters", "Food_Stock_Kg", "Medicine_Units", "Blankets_Count", "Fuel_Liters"]
stock_means = sample_data.groupby("NGO_Location")[stock_columns].mean().reset_index()
stock_means.set_index("NGO_Location").plot(kind="bar", figsize=(12, 8), colormap="viridis")
plt.title("Average Stock Levels by Location")
plt.ylabel("Average Quantity")
plt.xticks(rotation=45)
st.pyplot(plt)

# Additional information (optional)
st.markdown("""
    #### About the Model
    This model uses data such as the location of the NGO, the crisis level, and available supplies to predict potential shortages. The predictions are based on historical data and various machine learning techniques.
    
    *If you need further assistance, feel free to reach out to us!*
""")


