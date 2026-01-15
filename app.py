import streamlit as st
import numpy as np
import pickle
import os
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*unpickle estimator.*')

# Set page config first
st.set_page_config(page_title="Bank Churn Predictor", layout="wide")

# Load trained model and scaler
rfc = None
scaler = None
model_error = None

try:
    model_path = 'random_forest_model.pkl'
    scaler_path = 'scaler.pkl'
    
    # Check if files exist
    if not os.path.exists(model_path):
        model_error = f"Model file not found: {model_path}"
    elif not os.path.exists(scaler_path):
        model_error = f"Scaler file not found: {scaler_path}"
    else:
        # Load the files with warnings suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            with open(model_path, 'rb') as f:
                rfc = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
except Exception as e:
    model_error = f"Error loading model files: {str(e)}"

# Prediction function
def predict_churn(credit_score, country, gender, age, tenure, balance,
                  products_number, credit_card, active_member, estimated_salary):

    features = np.array([[credit_score, country, gender, age, tenure,
                          balance, products_number, credit_card,
                          active_member, estimated_salary]])

    features = scaler.transform(features)
    prediction = rfc.predict(features)

    return prediction[0]

# Streamlit UI
st.title("Bank Customer Churn Prediction")

# Show error if model failed to load
if model_error:
    st.error(f"⚠️ {model_error}")
    st.info("Please check that the model files exist in the application directory.")
else:
    st.success("✅ Model loaded successfully!")

st.markdown("---")
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=0, value=600)
    country = st.selectbox("Country", [0, 1, 2], format_func=lambda x: ["France", "Germany", "Spain"][x])
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: ["Female", "Male"][x])
    age = st.number_input("Age", min_value=18, value=30)
    tenure = st.number_input("Tenure (Years)", min_value=0, value=5)

with col2:
    balance = st.number_input("Account Balance", min_value=0, value=50000)
    products_number = st.number_input("Products Number", min_value=1, value=1)
    credit_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    active_member = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    estimated_salary = st.number_input("Estimated Salary", min_value=0, value=75000)

st.markdown("---")

if st.button("🔮 Predict Churn", key="predict_button"):
    if model_error or rfc is None or scaler is None:
        st.error("Cannot make prediction: Model not loaded properly")
    else:
        try:
            result = predict_churn(
                credit_score, country, gender, age, tenure,
                balance, products_number, credit_card,
                active_member, estimated_salary
            )

            if result == 1:
                st.error("🚨 Customer is likely to LEAVE the bank")
                st.markdown("""
                ### Recommendation:
                - Consider offering special retention packages
                - Review customer service quality
                - Offer loyalty rewards or better interest rates
                """)
            else:
                st.success("✅ Customer is likely to STAY with the bank")
                st.markdown("""
                ### Recommendation:
                - Maintain current service level
                - Consider cross-selling opportunities
                - Continue good customer relationship
                """)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
