import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(page_title="Fraud Detection API", page_icon="🚨", layout="centered")

# Load the trained model
# Using caching so the model doesn't reload every time a user clicks a button
@st.cache_resource
def load_model():
    return joblib.load('deployment/xgboost_fraud_model.joblib')

model = load_model()

# Build the UI
st.title("🚨 Real-Time Fraud Detection System")
st.markdown("Enter the transaction details below to evaluate the risk score.")

st.divider()

# Create input columns for a cleaner UI
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=1000.0)
    step = st.number_input("Time Step (Hour of Month)", min_value=1, max_value=744, value=1)

with col2:
    oldbalanceOrg = st.number_input("Origin - Old Balance", min_value=0.0, value=5000.0)
    newbalanceOrig = st.number_input("Origin - New Balance", min_value=0.0, value=4000.0)
    oldbalanceDest = st.number_input("Destination - Old Balance", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("Destination - New Balance", min_value=0.0, value=1000.0)

st.divider()

# Prediction Logic
if st.button("Evaluate Transaction", type="primary", use_container_width=True):
    
    # 1. Encode the categorical variable exactly as we did in training
    type_encoded = 1 if transaction_type == "TRANSFER" else 0
    
    # 2. Package the inputs into a DataFrame
    # MUST be in the exact same column order as your X_train data
    input_data = pd.DataFrame({
        'step': [step],
        'type': [type_encoded],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })
    
    # 3. Make the prediction
    prediction = model.predict(input_data)
    
    # 4. Display the result
    if prediction[0] == 1:
        st.error("🚨 HIGH RISK: Fraudulent Activity Detected. Transaction Blocked.")
    else:
        st.success("✅ LOW RISK: Transaction Approved.")