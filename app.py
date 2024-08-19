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

def validate_inputs(area, bedrooms, baths, unit='Square Yards'):
    """Validate if the given area can accommodate the specified number of bedrooms and bathrooms."""
    min_area_per_bedroom = 16  # square yards
    min_area_per_bathroom = 6  # square yards
    
    if unit == 'Square Feet':
        area = area / 9  # Convert square feet to square yards
    
    required_area = bedrooms * min_area_per_bedroom + baths * min_area_per_bathroom
    
    if required_area > area:
        # Calculate the minimum area required in square feet for user information
        min_area_sqft = (required_area * 9)
        return False, f"{bedrooms} bedrooms and {baths} bathrooms are not possible in {area:.2f} square yards. Minimum required area is {min_area_sqft:.2f} square feet."
    return True, ""

def predict_price(model, location, area, bedrooms, baths):
    loc_index = columns.index(location) if location in columns else -1
    
    x = np.zeros(len(columns))
    x[0] = baths
    x[1] = area
    x[2] = bedrooms
    if loc_index >= 0:
        x[loc_index] = 1
    
    return model.predict([x])[0] / 1000000

# Page Stylings
st.markdown(
    """
    <style>
    /* Main Background */
    .main {
        background-image: url('https://img.freepik.com/free-vector/man-telescope-looking-chart-rating-architecture-elements_126523-2932.jpg?w=740&t=st=1723633448~exp=1723634048~hmac=d3f3cb10ef7a696e740bd791ac2d9bab771238242327d579b03c9ae0dbf8691d');
        background-size: cover;
        background-position: center;
        color: #333333;
        padding: 20px;
        min-height: 100vh;
    }

    .subheader {
        font-size: 20px;
        color: #ffffff;
        background-color: rgba(0, 0, 0, 0.7);
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        margin: 0 auto;
        width: fit-content;
    }

    /* Paragraphs and Info Boxes */
    .paragraph, .stText {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 5px;
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Form Labels */
    .stSelectbox, .stNumberInput, .stTextInput {
        color: white !important;
    }

    /* Buttons */
    .button {
        background-color: #2596be;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    
    .button:hover {
        background-color: #006994;
    }

    /* Error and Success Messages */
    .error {
        color: ##ff1c1c;
        background-color: rgba(255, 28, 28,0.6);
        padding: 10px;
        border-radius: 5px;
    }
    
    .success {
        color: #ffffff;
        background-color: rgba(0, 255, 0, 0.6);
        padding: 10px;
        border-radius: 5px;
    }

    .lasttxt {
        background-color: rgba(255, 255, 255, 0.8);
        border: 2px solid #000000;
        border-radius: 5px;
    }

    /* Footer */
    .footer {
        background-color: #2596be;
        color: white;
        padding: 10px;
        text-align: center;
        font-size: small;
        position: fixed;
        width: 100%;
        bottom: 0;
        left: 0;
    }

    /* Responsive Design */
    @media only screen and (max-width: 768px) {
        .title {
            font-size: 28px;
        }

        .button {
            font-size: 14px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏠 Karachi House Price Prediction")
st.markdown("<p class='subheader'>Estimate the price of houses in various locations within Karachi using our machine learning model.</p>", unsafe_allow_html=True)

# Session State Initialization
if 'input_values' not in st.session_state:
    st.session_state.input_values = {
        'location': columns[3] if len(columns) > 3 else '',
        'area': 0,  # Initialize area
        'area_unit': 'Square Yards',
        'no_of_bedrooms': 1,
        'no_of_bathrooms': 1
    }

# Layout with Columns
col1, col2 = st.columns(2)

with col1:
    location_columns = columns[3:]
    st.session_state.input_values['location'] = st.selectbox("Select Location", location_columns, index=location_columns.index(st.session_state.input_values['location']), key='location')

    st.session_state.input_values['area_unit'] = st.selectbox("Area Unit", ['Square Yards', 'Square Feet'], index=0, key='area_unit')
    st.session_state.input_values['area'] = st.number_input(f"Area in {st.session_state.input_values['area_unit']}", min_value=0, step=1, value=st.session_state.input_values['area'], help="Enter the total area of the house.", key='area')

with col2:
    st.session_state.input_values['no_of_bedrooms'] = st.number_input("Number of Bedrooms", min_value=1, step=1, value=st.session_state.input_values['no_of_bedrooms'], help="Enter the number of bedrooms in the house.", key='no_of_bedrooms')
    st.session_state.input_values['no_of_bathrooms'] = st.number_input("Number of Bathrooms", min_value=1, step=1, value=st.session_state.input_values['no_of_bathrooms'], help="Enter the number of bathrooms in the house.", key='no_of_bathrooms')

# Validate inputs
valid, message = validate_inputs(st.session_state.input_values['area'], st.session_state.input_values['no_of_bedrooms'], st.session_state.input_values['no_of_bathrooms'], st.session_state.input_values['area_unit'])

# Display minimum area requirement
if st.session_state.input_values['area_unit'] == 'Square Feet':
    min_area_sqft = (st.session_state.input_values['no_of_bedrooms'] * 16 + st.session_state.input_values['no_of_bathrooms'] * 6) * 9
    st.markdown(f"<p class='subheader'>For {st.session_state.input_values['no_of_bedrooms']} bedrooms and {st.session_state.input_values['no_of_bathrooms']} bathrooms, the minimum required area is {min_area_sqft:.2f} square feet.</p>", unsafe_allow_html=True)

if st.button("🔍 Predict Price", key='predict_button'):
    if valid:
        area_in_yards = st.session_state.input_values['area'] if st.session_state.input_values['area_unit'] == 'Square Yards' else st.session_state.input_values['area'] / 9
        price = predict_price(model, st.session_state.input_values['location'], area_in_yards, st.session_state.input_values['no_of_bedrooms'], st.session_state.input_values['no_of_bathrooms'])
        st.markdown(f"<p class='success'>🏷️ The estimated house price is {price:.2f} Lakhs</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='error'>{message}</p>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p class='subheader'>About this App</p>", unsafe_allow_html=True)
st.markdown("<p class='lasttxt'>This app is a tool to predict house prices in Karachi using machine learning. Ensure that the inputs are realistic for accurate predictions.</p>", unsafe_allow_html=True)

# Add a Custom Footer
st.markdown("<style>footer {visibility: hidden;} .stApp {bottom: 0; position: fixed; width: 100%;}</style>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Developed by [Mustafa Badshah] | © 2024 All rights reserved.</div>", unsafe_allow_html=True)
