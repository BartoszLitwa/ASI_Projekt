import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = os.path.join('data/06_models', 'loan_model.pkl')
FEATURES_PATH = os.path.join('data/06_models', 'model_features.pkl')

@st.cache_resource
def load_model_and_features():
    """Load model and features with caching"""
    try:
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, features
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

def create_features_from_input(age, income, loan_amount, credit_score, months_employed, 
                              num_credit_lines, interest_rate, loan_term, dti_ratio,
                              education, employment_type, marital_status, 
                              has_mortgage, has_dependents, loan_purpose, has_cosigner):
    """
    Create the exact same features as the training pipeline for the new dataset
    """
    # Base numerical features from the form
    data = {
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'MonthsEmployed': months_employed,
        'NumCreditLines': num_credit_lines,
        'InterestRate': interest_rate,
        'LoanTerm': loan_term,
        'DTIRatio': dti_ratio,
    }
    
    # Convert boolean inputs to binary
    data['HasMortgage'] = 1 if has_mortgage else 0
    data['HasDependents'] = 1 if has_dependents else 0
    data['HasCoSigner'] = 1 if has_cosigner else 0
    
    # Education level encoding (ordinal)
    education_order = {
        'High School': 1,
        "Bachelor's": 2,
        "Master's": 3,
        'PhD': 4
    }
    data['education_level'] = education_order.get(education, 1)
    
    # Employment risk scores (based on typical risk patterns)
    employment_risk_map = {
        'Full-time': 0.08,
        'Part-time': 0.15,
        'Self-employed': 0.18,
        'Unemployed': 0.35
    }
    data['employment_risk_score'] = employment_risk_map.get(employment_type, 0.15)
    
    # Marital status encoding
    data['is_married'] = 1 if marital_status == 'Married' else 0
    data['is_divorced'] = 1 if marital_status == 'Divorced' else 0
    data['is_single'] = 1 if marital_status == 'Single' else 0
    
    # Loan purpose risk scores (based on typical default patterns)
    purpose_risk_map = {
        'Auto': 0.12,
        'Business': 0.22,
        'Education': 0.08,
        'Home': 0.10,
        'Other': 0.18
    }
    data['purpose_risk_score'] = purpose_risk_map.get(loan_purpose, 0.15)
    
    # Engineered features (matching the feature engineering pipeline)
    # 1. Loan-to-Income ratio
    data['loan_to_income_ratio'] = loan_amount / (income + 1)
    
    # 2. Credit utilization and capacity
    data['credit_per_line'] = credit_score / (num_credit_lines + 1)
    data['credit_density'] = num_credit_lines / (age + 1)
    
    # 3. Employment stability
    employment_years = months_employed / 12
    data['employment_years'] = employment_years
    data['employment_stability'] = months_employed / (age * 12 + 1)
    data['income_per_employment_year'] = income / (employment_years + 1)
    
    # 4. Loan characteristics
    monthly_payment = (loan_amount * (interest_rate / 100 / 12)) / (
        1 - (1 + interest_rate / 100 / 12) ** (-loan_term)
    )
    data['monthly_payment_estimate'] = monthly_payment
    data['payment_to_income_ratio'] = monthly_payment / ((income / 12) + 1)
    
    # 5. Age-based features
    data['age_group_young'] = 1 if age < 30 else 0
    data['age_group_middle'] = 1 if 30 <= age < 50 else 0
    data['age_group_senior'] = 1 if age >= 50 else 0
    
    # 6. Credit score categories
    data['credit_poor'] = 1 if credit_score < 580 else 0
    data['credit_fair'] = 1 if 580 <= credit_score < 670 else 0
    data['credit_good'] = 1 if 670 <= credit_score < 740 else 0
    data['credit_very_good'] = 1 if 740 <= credit_score < 800 else 0
    data['credit_excellent'] = 1 if credit_score >= 800 else 0
    
    # 7. Income brackets (using approximate training data quartiles)
    income_quartiles = [40000, 75000, 120000]  # Approximate based on US data
    data['income_low'] = 1 if income <= income_quartiles[0] else 0
    data['income_medium_low'] = 1 if income_quartiles[0] < income <= income_quartiles[1] else 0
    data['income_medium_high'] = 1 if income_quartiles[1] < income <= income_quartiles[2] else 0
    data['income_high'] = 1 if income > income_quartiles[2] else 0
    
    # 8. Loan amount categories
    loan_quartiles = [20000, 50000, 100000]  # Approximate
    data['loan_small'] = 1 if loan_amount <= loan_quartiles[0] else 0
    data['loan_medium'] = 1 if loan_quartiles[0] < loan_amount <= loan_quartiles[2] else 0
    data['loan_large'] = 1 if loan_amount > loan_quartiles[2] else 0
    
    # 9. Interest rate risk levels
    data['high_interest_rate'] = 1 if interest_rate > 15 else 0
    data['medium_interest_rate'] = 1 if 10 <= interest_rate <= 15 else 0
    data['low_interest_rate'] = 1 if interest_rate < 10 else 0
    
    # 10. Loan term features
    data['short_term_loan'] = 1 if loan_term <= 24 else 0
    data['medium_term_loan'] = 1 if 24 < loan_term <= 48 else 0
    data['long_term_loan'] = 1 if loan_term > 48 else 0
    
    # 11. Combined risk indicators
    data['high_dti_high_interest'] = 1 if (dti_ratio > 0.4 and interest_rate > 15) else 0
    data['low_credit_high_loan'] = 1 if (credit_score < 650 and data['loan_to_income_ratio'] > 0.5) else 0
    data['young_high_risk'] = 1 if (age < 25 and data['loan_to_income_ratio'] > 0.4) else 0
    
    # 12. Financial capacity score
    income_normalized = min((income - 20000) / (200000 - 20000), 1.0)  # Normalize income
    credit_normalized = (credit_score - 300) / (850 - 300)  # Standard credit range
    dti_normalized = 1 - dti_ratio  # Lower DTI is better
    
    data['income_normalized'] = max(income_normalized, 0)
    data['credit_normalized'] = max(credit_normalized, 0)
    data['dti_normalized'] = max(dti_normalized, 0)
    
    data['financial_capacity_score'] = (
        data['income_normalized'] * 0.4 +
        data['credit_normalized'] * 0.4 +
        data['dti_normalized'] * 0.2
    )
    
    # 13. Employment-based features
    data['underemployed'] = 1 if (months_employed < 12 and age > 25) else 0
    data['job_hopper'] = 1 if (months_employed < 24 and age > 30) else 0
    data['stable_employment'] = 1 if months_employed > 60 else 0
    
    # 14. Log transformations
    data['log_income'] = np.log1p(income)
    data['log_loan_amount'] = np.log1p(loan_amount)
    data['sqrt_credit_score'] = np.sqrt(credit_score)
    
    # 15. Percentile-based features (using training data approximations)
    data['high_income_applicant'] = 1 if income > 150000 else 0
    data['large_loan_applicant'] = 1 if loan_amount > 150000 else 0
    data['poor_credit_applicant'] = 1 if credit_score < 400 else 0
    data['high_dti_applicant'] = 1 if dti_ratio > 0.6 else 0
    
    return data

def main():
    st.set_page_config(page_title='🏦 Loan Default Predictor', layout='wide')
    st.title('🏦 Loan Default Risk Assessment System')
    st.markdown('*Advanced ML-based loan default prediction*')
    
    # Load model
    model, expected_features = load_model_and_features()
    
    # Main form
    st.header('📋 Loan Application Form')
    
    with st.form('loan_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Personal Information")
            age = st.number_input('Age', min_value=18, max_value=80, value=35, 
                                help="Applicant's age in years")
            income = st.number_input('Annual Income ($)', 
                                   min_value=10000, 
                                   max_value=1000000, 
                                   value=75000, 
                                   step=5000,
                                   help="Annual gross income in USD")
            credit_score = st.number_input('Credit Score', 
                                         min_value=300, 
                                         max_value=850, 
                                         value=650,
                                         help="FICO credit score (300-850)")
            months_employed = st.number_input('Months Employed', 
                                            min_value=0, 
                                            max_value=480, 
                                            value=24,
                                            help="Months at current job")
        
        with col2:
            st.subheader("💰 Loan Details")
            loan_amount = st.number_input('Loan Amount ($)', 
                                        min_value=1000, 
                                        max_value=500000, 
                                        value=50000, 
                                        step=1000,
                                        help="Requested loan amount")
            interest_rate = st.number_input('Interest Rate (%)', 
                                          min_value=1.0, 
                                          max_value=30.0, 
                                          value=8.5, 
                                          step=0.1,
                                          help="Annual interest rate")
            loan_term = st.number_input('Loan Term (months)', 
                                      min_value=6, 
                                      max_value=84, 
                                      value=36,
                                      help="Loan repayment period in months")
            dti_ratio = st.number_input('Debt-to-Income Ratio', 
                                      min_value=0.0, 
                                      max_value=1.0, 
                                      value=0.3, 
                                      step=0.01,
                                      help="Total monthly debt / monthly income")
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("📊 Financial Profile")
            num_credit_lines = st.number_input('Number of Credit Lines', 
                                             min_value=0, 
                                             max_value=20, 
                                             value=3,
                                             help="Total number of credit accounts")
            education = st.selectbox('Education Level', [
                'High School', "Bachelor's", "Master's", 'PhD'
            ])
            employment_type = st.selectbox('Employment Type', [
                'Full-time', 'Part-time', 'Self-employed', 'Unemployed'
            ])
            marital_status = st.selectbox('Marital Status', [
                'Single', 'Married', 'Divorced'
            ])
            
        with col4:
            st.subheader("🏠 Additional Information")
            has_mortgage = st.checkbox('Has Mortgage', help="Currently has a mortgage")
            has_dependents = st.checkbox('Has Dependents', help="Has dependents to support")
            has_cosigner = st.checkbox('Has Co-signer', help="Loan application has a co-signer")
            loan_purpose = st.selectbox('Loan Purpose', [
                'Auto', 'Business', 'Education', 'Home', 'Other'
            ])
        
        submitted = st.form_submit_button('🔍 Assess Default Risk', use_container_width=True)
    
    if submitted:
        # Create features using the same logic as training
        features_dict = create_features_from_input(
            age, income, loan_amount, credit_score, months_employed,
            num_credit_lines, interest_rate, loan_term, dti_ratio,
            education, employment_type, marital_status,
            has_mortgage, has_dependents, loan_purpose, has_cosigner
        )
        
        # Convert to DataFrame and align with model features
        input_df = pd.DataFrame([features_dict])
        input_df = input_df.reindex(expected_features, axis=1, fill_value=0)
        
        # Make prediction
        default_proba = model.predict_proba(input_df)[0][1]  # Probability of default
        
        # Display results
        st.markdown('---')
        st.header('🎯 Default Risk Assessment Results')
        
        # Risk level determination with business-focused thresholds
        if default_proba < 0.10:
            risk_level = "Very Low"
            color = "green"
            decision = "✅ APPROVED"
            recommendation = "Excellent candidate - approve with standard terms"
        elif default_proba < 0.25:
            risk_level = "Low"
            color = "lightgreen"
            decision = "✅ APPROVED"
            recommendation = "Good candidate - approve with standard terms"
        elif default_proba < 0.40:
            risk_level = "Moderate"
            color = "orange"
            decision = "⚠️ CONDITIONAL APPROVAL"
            recommendation = "Consider approval with higher interest rate or shorter term"
        elif default_proba < 0.60:
            risk_level = "High"
            color = "red"
            decision = "❌ REQUIRES MANUAL REVIEW"
            recommendation = "High risk - require additional documentation and collateral"
        else:
            risk_level = "Very High"
            color = "darkred"
            decision = "❌ RECOMMEND REJECTION"
            recommendation = "Very high default risk - recommend rejection"
        
        # Main result display
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric(
                "Default Probability", 
                f"{default_proba:.1%}",
                help="Probability that this loan will default"
            )
        with col_res2:
            st.metric("Risk Level", risk_level)
        with col_res3:
            if "APPROVED" in decision and "CONDITIONAL" not in decision:
                st.success(decision)
            elif "CONDITIONAL" in decision or "REVIEW" in decision:
                st.warning(decision)
            else:
                st.error(decision)
        
        # Recommendation
        st.info(f"💡 **Recommendation**: {recommendation}")
        
        # Key factors analysis
        st.subheader('📊 Key Risk Factors Analysis')
        col_factor1, col_factor2 = st.columns(2)
        
        with col_factor1:
            st.metric("Loan-to-Income Ratio", f"{features_dict['loan_to_income_ratio']:.2f}")
            st.metric("Credit Utilization", f"{features_dict['credit_per_line']:.0f}")
            st.metric("Employment Stability", f"{features_dict['employment_stability']:.3f}")
            st.metric("Monthly Payment", f"${features_dict['monthly_payment_estimate']:,.0f}")
            
        with col_factor2:
            st.metric("Financial Capacity Score", f"{features_dict['financial_capacity_score']:.2f}")
            st.metric("Employment Risk", f"{features_dict['employment_risk_score']:.1%}")
            st.metric("Purpose Risk", f"{features_dict['purpose_risk_score']:.1%}")
            credit_tier = "Excellent" if credit_score >= 800 else "Very Good" if credit_score >= 740 else "Good" if credit_score >= 670 else "Fair" if credit_score >= 580 else "Poor"
            st.metric("Credit Tier", credit_tier)
        
        # Detailed risk breakdown
        with st.expander("📈 Detailed Risk Breakdown"):
            st.write("**High Risk Indicators:**")
            risk_factors = []
            if features_dict['high_dti_high_interest']: risk_factors.append("High DTI + High Interest Rate")
            if features_dict['low_credit_high_loan']: risk_factors.append("Low Credit Score + High Loan Amount")
            if features_dict['young_high_risk']: risk_factors.append("Young Age + High Loan-to-Income")
            if features_dict['underemployed']: risk_factors.append("Short Employment History")
            if features_dict['job_hopper']: risk_factors.append("Frequent Job Changes")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("• No major red flags detected")
                
            st.write("**Positive Factors:**")
            positive_factors = []
            if features_dict['stable_employment']: positive_factors.append("Stable Employment (5+ years)")
            if features_dict['credit_excellent']: positive_factors.append("Excellent Credit Score")
            if features_dict['low_interest_rate']: positive_factors.append("Low Interest Rate Qualified")
            if features_dict['high_income_applicant']: positive_factors.append("High Income Earner")
            
            if positive_factors:
                for factor in positive_factors:
                    st.write(f"• {factor}")
            else:
                st.write("• Standard risk profile")
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.metric("Model Type", "Random Forest")
        st.metric("Features Used", len(expected_features))
        
        st.subheader("📈 Risk Thresholds")
        st.write("• Very Low: < 10%")
        st.write("• Low: 10% - 25%")
        st.write("• Moderate: 25% - 40%")
        st.write("• High: 40% - 60%")
        st.write("• Very High: > 60%")
        
        st.subheader("💡 Tips for Better Terms")
        st.write("**To improve approval odds:**")
        st.write("• Improve credit score")
        st.write("• Lower debt-to-income ratio")
        st.write("• Increase down payment")
        st.write("• Consider a co-signer")
        st.write("• Choose shorter loan term")
        
        if st.checkbox("Show Feature Debug Info"):
            if submitted:
                st.subheader("Key Feature Values")
                key_features = ['loan_to_income_ratio', 'financial_capacity_score', 
                               'credit_per_line', 'payment_to_income_ratio']
                for feat in key_features:
                    if feat in features_dict:
                        st.write(f"{feat}: {features_dict[feat]:.3f}")

if __name__ == '__main__':
    main() 