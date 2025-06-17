import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join('data/06_models', 'loan_model.pkl')
FEATURES_PATH = os.path.join('data/06_models', 'model_features.pkl')

# Sensible defaults for each feature
DEFAULTS = {
    'Income': 5000000,
    'Age': 35,
    'Experience': 10,
    'CURRENT_JOB_YRS': 5,
    'CURRENT_HOUSE_YRS': 7,
    'Married/Single_single': 0,
    'House_Ownership_owned': 1,
    'House_Ownership_rented': 0,
    'Car_Ownership_yes': 1,
    'income_per_year_of_job': 500000,
    'age_bucket_middle': 1,
    'age_bucket_senior': 0,
}

# Feature types for concise UI
NUMERIC_FEATURES = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS', 'income_per_year_of_job']
CHECKBOX_FEATURES = ['Married/Single_single', 'House_Ownership_owned', 'House_Ownership_rented', 'Car_Ownership_yes', 'age_bucket_middle', 'age_bucket_senior']
# For age bucket, only one can be 1 at a time, so we can use a dropdown for age bucket
AGE_BUCKETS = ['young', 'middle', 'senior']
INT_FEATURES = ['Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']
FLOAT_FEATURES = ['Income', 'income_per_year_of_job']

def safe_float(val, default):
    try:
        f = float(val)
        return f
    except (ValueError, TypeError):
        return float(default)

def safe_int(val, default):
    try:
        i = int(float(val))
        return i
    except (ValueError, TypeError):
        return int(default)

def safe_bool(val, default):
    try:
        return bool(int(val))
    except (ValueError, TypeError):
        return bool(int(default))

def load_model():
    return joblib.load(MODEL_PATH)

def load_features():
    return joblib.load(FEATURES_PATH)

def get_feature_importances(model, features):
    if hasattr(model, 'feature_importances_'):
        return dict(sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]))
    return {}

def main():
    st.set_page_config(page_title='Loan Approval Prediction', layout='centered')
    st.title('Loan Approval Prediction')
    st.markdown('---')
    model = load_model()
    features = load_features()
    importances = get_feature_importances(model, features)

    # Sidebar for threshold, instructions, and feature importances
    st.sidebar.header('Settings')
    threshold = st.sidebar.slider('Prediction Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    st.sidebar.markdown('Adjust the threshold to set how certain the model must be to approve a loan.')
    st.sidebar.markdown('---')
    st.sidebar.markdown('**Instructions:**\n- Fill in the applicant information.\n- Click "Fill Example" for sample data.\n- Click "Predict" to see the result.')
    st.sidebar.markdown('---')
    st.sidebar.subheader('Feature Importances')
    if importances:
        st.sidebar.bar_chart(pd.Series(importances))
    else:
        st.sidebar.write('No importances available.')

    st.header('Applicant Information')
    # Use session state to allow filling defaults
    if 'inputs' not in st.session_state:
        st.session_state['inputs'] = {f: '' for f in features}
        st.session_state['age_bucket'] = 'middle'

    def fill_defaults():
        for f in features:
            st.session_state['inputs'][f] = str(DEFAULTS.get(f, ''))
        # Set age bucket dropdown
        if DEFAULTS.get('age_bucket_middle', 0) == 1:
            st.session_state['age_bucket'] = 'middle'
        elif DEFAULTS.get('age_bucket_senior', 0) == 1:
            st.session_state['age_bucket'] = 'senior'
        else:
            st.session_state['age_bucket'] = 'young'

    st.button('Fill Example', on_click=fill_defaults)

    with st.form(key='input_form'):
        user_input = {}
        # Numeric features
        for feature in NUMERIC_FEATURES:
            if feature in INT_FEATURES:
                user_input[feature] = st.number_input(
                    feature,
                    value=safe_int(st.session_state['inputs'].get(feature, DEFAULTS.get(feature, 0)), DEFAULTS.get(feature, 0)),
                    step=1
                )
            else:
                user_input[feature] = st.number_input(
                    feature,
                    value=safe_float(st.session_state['inputs'].get(feature, DEFAULTS.get(feature, 0)), DEFAULTS.get(feature, 0)),
                    step=1000.0
                )
            st.session_state['inputs'][feature] = str(user_input[feature])
        # Age bucket as dropdown (only one can be 1)
        age_bucket = st.selectbox('Age Bucket', AGE_BUCKETS, index=AGE_BUCKETS.index(st.session_state.get('age_bucket', 'middle')))
        st.session_state['age_bucket'] = age_bucket
        # Set one-hot encoding for age buckets
        user_input['age_bucket_middle'] = 1 if age_bucket == 'middle' else 0
        user_input['age_bucket_senior'] = 1 if age_bucket == 'senior' else 0
        # Checkbox features (except age buckets)
        for feature in [f for f in CHECKBOX_FEATURES if not f.startswith('age_bucket')]:
            user_input[feature] = st.checkbox(
                feature.replace('_', ' '),
                value=safe_bool(st.session_state['inputs'].get(feature, DEFAULTS.get(feature, 0)), DEFAULTS.get(feature, 0))
            )
            st.session_state['inputs'][feature] = str(int(user_input[feature]))
        submit = st.form_submit_button('Predict')

    if submit:
        # Prepare input for model
        input_df = pd.DataFrame([user_input])
        # Convert numeric columns
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except Exception:
                pass
        # Ensure columns are in the same order and names as model features
        input_df = input_df.reindex(features, axis=1, fill_value=0)
        proba = model.predict_proba(input_df)[0][1]  # Probability of approval
        prediction = int(proba >= threshold)
        st.markdown(f"**Prediction Probability:** {proba:.2f}")
        st.markdown(f"**Threshold Used:** {threshold:.2f}")
        if prediction == 1:
            st.success('Loan Status Prediction: Approved')
        else:
            st.error('Loan Status Prediction: Rejected')

if __name__ == '__main__':
    main() 