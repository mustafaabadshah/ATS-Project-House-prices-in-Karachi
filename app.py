import streamlit as st
import joblib
import json
import base64
import numpy as np

# Authentication management
users = {"admin": "admin"}  # Default credentials (username: password)
session_state = st.session_state

# Load model and column names
@st.cache_resource
def load_model():
    return joblib.load('random_forest_regressor_model.pkl')

@st.cache_data
def load_columns():
    with open('columns-v1.json', 'r') as file:
        return json.load(file)

model = load_model()
columns = load_columns()

# Function to convert image to base64
def get_base64_image(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Add background image CSS using Base64 encoding
def add_bg_image(img_path):
    base64_img = get_base64_image(img_path)
    css_code = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(to bottom, rgba(255, 255, 255, 0.5), rgba(200, 200, 200, 0.5)), 
                          url("data:image/jpg;base64,{base64_img}");
        background-size: cover;
        background-position: center;
        font-family: 'Arial', sans-serif;
    }}
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)

# Add background image
add_bg_image("img/bgImg2.jpg")  # Replace with your image path

# Login page
def login_page():
    st.markdown("<h1 style='text-align: center;'>🔐 Login</h1>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("Login"):
            if username in users and users[username] == password:
                session_state.logged_in = True
                session_state.current_user = username
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password!")
    
    with col2:
        if st.button("Forgot Password?"):
            st.info("Feature not implemented yet. Please contact admin.")

# Add new user
def add_user():
    st.markdown("<h2 style='text-align: center;'>➕ Add New User</h2>", unsafe_allow_html=True)
    if session_state.current_user == "admin":
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Create User"):
            if new_username in users:
                st.error("User already exists!")
            elif new_username and new_password:
                users[new_username] = new_password
                st.success(f"User {new_username} created successfully!")
            else:
                st.warning("Please fill in all fields!")
    else:
        st.warning("Only admin can create new users.")

# Main App (House Price Prediction)
def main_page():
    st.markdown("<h1 class='main-title'>🏠 Karachi House Price Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Estimate the price of houses in Karachi using an advanced machine learning model.</p>", unsafe_allow_html=True)
    
    # Input Section
    with st.container():
        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox("📍 Select Location", columns[3:])
            area_unit = st.selectbox("📐 Area Unit", ['Square Yards', 'Square Feet'])
            area = st.number_input(f"📏 Area in {area_unit}", min_value=0, step=1)
        with col2:
            bedrooms = st.number_input("🛏️ Number of Bedrooms", min_value=1, step=1)
            baths = st.number_input("🛁 Number of Bathrooms", min_value=1, step=1)
        st.markdown("</div>", unsafe_allow_html=True)

    # Validate Inputs
    def validate_inputs(area, bedrooms, baths, unit='Square Yards'):
        min_area_per_bedroom = 16  # square yards
        min_area_per_bathroom = 6  # square yards

        if unit == 'Square Feet':
            area = area / 9  # Convert square feet to square yards

        required_area = bedrooms * min_area_per_bedroom + baths * min_area_per_bathroom

        if required_area > area:
            min_area_sqft = required_area * 9
            return False, f"{bedrooms} bedrooms and {baths} bathrooms are not possible in {area:.2f} square yards. Minimum required area is {min_area_sqft:.2f} square feet."
        return True, ""

    valid, message = validate_inputs(area, bedrooms, baths, area_unit)

    # Predict Price
    if st.button("🔍 Predict Price"):
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        if valid:
            area_in_yards = area if area_unit == 'Square Yards' else area / 9
            loc_index = columns.index(location) if location in columns else -1

            x = np.zeros(len(columns))
            x[0] = baths
            x[1] = area_in_yards
            x[2] = bedrooms
            if loc_index >= 0:
                x[loc_index] = 1

            price = model.predict([x])[0] / 1000000
            st.markdown(f"<p class='success'>🏷️ The estimated house price is <b>{price:.2f} Lakhs</b></p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='error'>{message}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Control navigation
if "logged_in" not in session_state:
    session_state.logged_in = False

if session_state.logged_in:
    st.sidebar.markdown(f"👤 Logged in as: {session_state.current_user}")
    page = st.sidebar.selectbox("Navigation", ["Home", "Add User", "Logout"])
    
    if page == "Home":
        main_page()
    elif page == "Add User":
        add_user()
    elif page == "Logout":
        session_state.logged_in = False
        session_state.current_user = None
        st.sidebar.success("Logged out!")
else:
    login_page()
