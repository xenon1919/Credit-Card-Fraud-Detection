import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Load the saved model
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler and transformer
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('transformer.pkl', 'rb') as transformer_file:
    transformer = pickle.load(transformer_file)

# Streamlit App Configuration
st.title("Credit Card Fraud Detection")
st.write("This app predicts whether a transaction is fraudulent or not based on the provided inputs.")

# Input Form
def user_input_features():
    V1 = st.number_input("V1", value=0.0, format="%.2f")
    V2 = st.number_input("V2", value=0.0, format="%.2f")
    V3 = st.number_input("V3", value=0.0, format="%.2f")
    V4 = st.number_input("V4", value=0.0, format="%.2f")
    V5 = st.number_input("V5", value=0.0, format="%.2f")
    V6 = st.number_input("V6", value=0.0, format="%.2f")
    V7 = st.number_input("V7", value=0.0, format="%.2f")
    V8 = st.number_input("V8", value=0.0, format="%.2f")
    V9 = st.number_input("V9", value=0.0, format="%.2f")
    V10 = st.number_input("V10", value=0.0, format="%.2f")
    V11 = st.number_input("V11", value=0.0, format="%.2f")
    V12 = st.number_input("V12", value=0.0, format="%.2f")
    V13 = st.number_input("V13", value=0.0, format="%.2f")
    V14 = st.number_input("V14", value=0.0, format="%.2f")
    V15 = st.number_input("V15", value=0.0, format="%.2f")
    V16 = st.number_input("V16", value=0.0, format="%.2f")
    V17 = st.number_input("V17", value=0.0, format="%.2f")
    V18 = st.number_input("V18", value=0.0, format="%.2f")
    V19 = st.number_input("V19", value=0.0, format="%.2f")
    V20 = st.number_input("V20", value=0.0, format="%.2f")
    V21 = st.number_input("V21", value=0.0, format="%.2f")
    V22 = st.number_input("V22", value=0.0, format="%.2f")
    V23 = st.number_input("V23", value=0.0, format="%.2f")
    V24 = st.number_input("V24", value=0.0, format="%.2f")
    V25 = st.number_input("V25", value=0.0, format="%.2f")
    V26 = st.number_input("V26", value=0.0, format="%.2f")
    V27 = st.number_input("V27", value=0.0, format="%.2f")
    V28 = st.number_input("V28", value=0.0, format="%.2f")
    Amount = st.number_input("Amount", value=0.0, format="%.2f")

    data = {
        'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6, 'V7': V7, 'V8': V8,
        'V9': V9, 'V10': V10, 'V11': V11, 'V12': V12, 'V13': V13, 'V14': V14, 'V15': V15,
        'V16': V16, 'V17': V17, 'V18': V18, 'V19': V19, 'V20': V20, 'V21': V21, 'V22': V22,
        'V23': V23, 'V24': V24, 'V25': V25, 'V26': V26, 'V27': V27, 'V28': V28, 'Amount': Amount
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Preprocess Input Data
scaled_input = scaler.transform(input_df[['Amount']])
input_df[['Amount']] = scaled_input
transformed_input = transformer.transform(input_df)

# Prediction
if st.button("Predict"): 
    prediction = model.predict(transformed_input)
    prediction_proba = model.predict_proba(transformed_input)[:, 1]

    if prediction[0] == 1:
        st.error(f"⚠️ This transaction is predicted to be FRAUDULENT with a probability of {prediction_proba[0]:.2f}.")
    else:
        st.success(f"✅ This transaction is predicted to be NON-FRAUDULENT with a probability of {1 - prediction_proba[0]:.2f}.")
