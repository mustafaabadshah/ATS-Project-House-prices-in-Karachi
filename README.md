# House Price Prediction in Karachi

## Overview

This repository contains a project for predicting house prices in Karachi, Pakistan. The project utilizes machine learning techniques to estimate property values based on various features such as area, number of bedrooms, and bathrooms. The app is built using Streamlit for a user-friendly interface and leverages a Decision Tree Regressor model for predictions.

## Project Components

1. **Data Preprocessing**
   - Cleansing and transforming the dataset.
   - Handling missing values and outliers.
   - Feature engineering and dimensionality reduction.

2. **Model Training**
   - Training various regression models, including Linear Regression, Lasso, Decision Tree Regressor, and Random Forest Regressor.
   - Hyperparameter tuning using GridSearchCV.

3. **Streamlit Application**
   - A web application that allows users to input property details and get price predictions.
   - Displays results with visual appeal using custom CSS styling.

4. **Data**
   - A dataset containing house prices in Karachi with features such as address, price, number of bedrooms, number of bathrooms, and area in square yards.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

- `streamlit`
- `joblib`
- `json`
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `seaborn`
- `matplotlib`

You can install these packages using pip:

```bash
pip install streamlit joblib numpy pandas scikit-learn xgboost seaborn matplotlib
```

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/house-price-prediction-karachi.git
   cd house-price-prediction-karachi
   ```

2. **Download the dataset:**

   Ensure you have the dataset `house-prices-in-karachi-pakistan-2023.csv` and place it in the project directory.

3. **Model Files:**

   Place the pre-trained model files (`decision_tree_regressor.pkl` and `columns-v1.json`) in the project directory.

4. **Run the Streamlit App:**

   Execute the following command to start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## File Descriptions

- `app.py`: The main Streamlit application script for user interaction and prediction.

## Code Highlights

### Data Preprocessing

- Cleans data by removing outliers and transforming features.
- Encodes categorical features and performs dimensionality reduction.

### Model Training

- Trains models using `LinearRegression`, `Lasso`, `DecisionTreeRegressor`, and `RandomForestRegressor`.
- Performs hyperparameter tuning with `GridSearchCV` and selects the best model based on cross-validation scores.

### Streamlit Application

- Provides a user-friendly interface for entering property details.
- Displays predictions and ensures input validation.

## Examples

To get price predictions, input details like location, area, number of bedrooms, and bathrooms into the Streamlit app.

Example Predictions:

```python
print(str(int(predict_price('Nazimabad', 1800, 4, 3))) + " Lakhs")
print(str(int(predict_price('Scheme 33', 1080, 3, 2))) + " Lakhs")
```

## Contributions

Feel free to fork the repository and submit pull requests. For any issues or suggestions, please create an issue or contact the repository owner.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Developed by [Mustafa Badshah](https://github.com/mustafaabadshah) | © 2024 All rights reserved.

---

Feel free to modify or expand upon this template based on your project’s specific details and requirements!
