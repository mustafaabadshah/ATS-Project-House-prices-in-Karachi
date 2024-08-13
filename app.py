import streamlit as st
import joblib
import json
import numpy as np

# Load column names from JSON file
with open('columns-v1.json', 'r') as file:
    columns = json.load(file)

# Example: Load the trained model
model = joblib.load('decision_tree_regressor.pkl')  # Use joblib to load your model

def predict_price(model, location, sqft, bedrooms, baths):
    loc_index = columns.index(location) if location in columns else -1
    
    x = np.zeros(len(columns))
    x[0] = baths
    x[1] = sqft
    x[2] = bedrooms
    if loc_index >= 0:
        x[loc_index] = 1
    
    return model.predict([x])[0] / 100000

st.title("House Price Prediction in Karachi")

# Extract location columns (assumed to be all columns after the first three)
location_columns = columns[3:]

# User inputs
location = st.selectbox("Location", location_columns)
area_sq_yards = st.number_input("Area in Square Yards", min_value=0)
no_of_bedrooms = st.number_input("Number of Bedrooms", min_value=1)
no_of_bathrooms = st.number_input("Number of Bathrooms", min_value=1)

if st.button("Predict Price"):
    price = predict_price(model, location, area_sq_yards, no_of_bedrooms, no_of_bathrooms)
    st.success(f"The estimated house price is {price:.2f} Lakhs")
