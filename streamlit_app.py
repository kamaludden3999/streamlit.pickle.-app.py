import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title of the app
st.title("Logistic Regression Model Deployment")

# Description
st.write("This app allows users to input data and get predictions from the trained logistic regression model.")

# Input fields for user data
# Adjust the input fields based on your model's feature requirements
feature_1 = st.number_input("Enter value for Feature 1:")
feature_2 = st.number_input("Enter value for Feature 2:")
# Add more input fields as needed

# Create a button for prediction
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[feature_1, feature_2]])  # Update based on the number of features
    prediction = model.predict(input_data)
    st.write(f"The predicted value is: {prediction[0]}")
