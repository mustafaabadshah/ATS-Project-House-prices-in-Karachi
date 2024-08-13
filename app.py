import streamlit as st
import joblib
import json
import numpy as np

# Load column names from JSON file
with open('columns-v1.json', 'r') as file:
    columns = json.load(file)

# Load the trained model
model = joblib.load('decision_tree_regressor.pkl')

def validate_inputs(sqft, bedrooms, baths):
    """Validate if the given area can accommodate the specified number of bedrooms and bathrooms."""
    min_area_per_bedroom = 100  # square yards
    min_area_per_bathroom = 50  # square yards
    
    required_area = bedrooms * min_area_per_bedroom + baths * min_area_per_bathroom
    
    if required_area > sqft:
        return False, f"{bedrooms} bedrooms and {baths} bathrooms are not possible in {sqft} square yards."
    return True, ""

def predict_price(model, location, sqft, bedrooms, baths):
    loc_index = columns.index(location) if location in columns else -1
    
    x = np.zeros(len(columns))
    x[0] = baths
    x[1] = sqft
    x[2] = bedrooms
    if loc_index >= 0:
        x[loc_index] = 1
    
    return model.predict([x])[0] / 100000

# Page Title with Subheading
st.title("🏠 Karachi House Price Prediction")
st.markdown("Use this tool to estimate the price of houses in various locations within Karachi.")

# Use Columns for Layout
col1, col2 = st.columns(2)

with col1:
    # Select Location Dropdown
    location_columns = columns[3:]
    location = st.selectbox("Select Location", location_columns)

    # Area in Square Yards Input
    area_sq_yards = st.number_input("Area in Square Yards", min_value=0, step=1, help="Enter the total area of the house in square yards.")

with col2:
    # Number of Bedrooms Input
    no_of_bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1, help="Enter the number of bedrooms in the house.")

    # Number of Bathrooms Input
    no_of_bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1, help="Enter the number of bathrooms in the house.")

# Validate inputs
valid, message = validate_inputs(area_sq_yards, no_of_bedrooms, no_of_bathrooms)

# Add a Predict Button and Display the Result
if st.button("🔍 Predict Price"):
    if valid:
        price = predict_price(model, location, area_sq_yards, no_of_bedrooms, no_of_bathrooms)
        st.success(f"🏷️ The estimated house price is **{price:.2f} Lakhs**")
    else:
        st.error(message)

# Additional Information or Footer
st.markdown("---")
st.markdown("### About this App")
st.info("This app is a tool to predict house prices in Karachi using machine learning. The predictions are based on factors like location, area, and number of rooms. For more accurate results, ensure that the inputs are realistic.")

# Add a Custom Footer
st.markdown("<style>footer {visibility: hidden;} .stApp {bottom: 0; position: fixed; width: 100%; color: gray; background-color: #f0f2f6; padding: 10px; text-align: center; font-size: small;}</style>", unsafe_allow_html=True)
st.markdown("Developed by [Mustafa Badshah](https://github.com/mustafaabadshah) | © 2024 All rights reserved.", unsafe_allow_html=True)
