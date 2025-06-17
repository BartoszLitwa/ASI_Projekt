import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../data/06_models/loan_model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '../data/06_models/model_features.pkl')

def load_model():
    return joblib.load(MODEL_PATH)

def load_features():
    return joblib.load(FEATURES_PATH)

def main():
    st.title('Loan Approval Prediction')
    model = load_model()
    features = load_features()

    st.header('Enter Applicant Information')
    user_input = {}
    for feature in features:
        user_input[feature] = st.text_input(f"{feature}")

    if st.button('Predict'):
        input_df = pd.DataFrame([user_input])
        # Convert numeric columns
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except Exception:
                pass
        prediction = model.predict(input_df)[0]
        st.success(f'Loan Status Prediction: {"Approved" if prediction == 1 else "Rejected"}')

if __name__ == '__main__':
    main() 