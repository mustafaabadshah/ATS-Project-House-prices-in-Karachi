import streamlit as st
import joblib
import json
import numpy as np

# Load column names from JSON file
@st.cache_resource
def load_model():
    return joblib.load('random_forest_regressor_model.pkl')

@st.cache_data
def load_columns():
    with open('columns-v1.json', 'r') as file:
        return json.load(file)

model = load_model()
columns = load_columns()

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

# Page Styling
st.markdown(
    """
    <style>
    .main {background-color: #f5f5f5; padding: 20px;}
    .title {font-size: 36px; font-weight: bold; color: #333;}
    .subheader {font-size: 24px; color: #555;}
    .button {background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px;}
    .button:hover {background-color: #45a049;}
    .error {color: #e57373;}
    .success {color: #81c784;}
    .footer {background-color: #333; color: white; padding: 10px; text-align: center; font-size: small;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏠 Karachi House Price Prediction")
st.markdown("<p class='subheader'>Estimate the price of houses in various locations within Karachi using our machine learning model.</p>", unsafe_allow_html=True)

# Layout with Columns
col1, col2 = st.columns(2)

with col1:
    location_columns = columns[3:]
    location = st.selectbox("Select Location", location_columns, key='location')

    area_sq_yards = st.number_input("Area in Square Yards", min_value=0, step=1, help="Enter the total area of the house in square yards.")

with col2:
    no_of_bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1, help="Enter the number of bedrooms in the house.")
    no_of_bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1, help="Enter the number of bathrooms in the house.")

# Validate inputs
valid, message = validate_inputs(area_sq_yards, no_of_bedrooms, no_of_bathrooms)

if st.button("🔍 Predict Price", key='predict_button'):
    if valid:
        price = predict_price(model, location, area_sq_yards, no_of_bedrooms, no_of_bathrooms)
        st.markdown(f"<p class='success'>🏷️ The estimated house price is **{price:.2f} Lakhs**</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='error'>{message}</p>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p class='subheader'>About this App</p>", unsafe_allow_html=True)
st.info("This app is a tool to predict house prices in Karachi using machine learning. Ensure that the inputs are realistic for accurate predictions.")


# Add a Custom Footer
st.markdown("<style>footer {visibility: hidden;} .stApp {bottom: 0; position: fixed; width: 100%; color: gray; background-color: #f0f2f6; padding: 10px; text-align: center; font-size: small;}</style>", unsafe_allow_html=True)
st.markdown("Developed by [Mustafa Badshah](https://github.com/mustafaabadshah) | © 2024 All rights reserved.", unsafe_allow_html=True)
