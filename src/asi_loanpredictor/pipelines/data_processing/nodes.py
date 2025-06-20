import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def feature_engineering(train: pd.DataFrame, test: pd.DataFrame):
    """
    Create essential engineered features for the new loan dataset.
    Available features: Age, Income, LoanAmount, CreditScore, MonthsEmployed, 
                       NumCreditLines, InterestRate, LoanTerm, DTIRatio, Default
    Plus encoded features from data preparation step.
    """
    logger.info(f"Starting feature engineering. Train: {train.shape}, Test: {test.shape}")
    
    train_feat = train.copy()
    test_feat = test.copy()
    
    # Validate required columns exist
    required_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore']
    missing_cols = [col for col in required_cols if col not in train_feat.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
        return train_feat, test_feat
    
    for df in [train_feat, test_feat]:
        # 1. Loan-to-Income ratio (key metric for loan approval)
        df['loan_to_income_ratio'] = df['LoanAmount'] / (df['Income'] + 1)
        
        # 2. Credit utilization and capacity
        df['credit_per_line'] = df['CreditScore'] / (df['NumCreditLines'] + 1)
        df['credit_density'] = df['NumCreditLines'] / (df['Age'] + 1)  # Credit lines per year of life
        
        # 3. Employment stability indicators
        if 'MonthsEmployed' in df.columns:
            df['employment_years'] = df['MonthsEmployed'] / 12
            df['employment_stability'] = df['MonthsEmployed'] / (df['Age'] * 12 + 1)  # % of life employed
            df['income_per_employment_year'] = df['Income'] / (df['employment_years'] + 1)
        
        # 4. Loan characteristics
        df['monthly_payment_estimate'] = (df['LoanAmount'] * (df['InterestRate'] / 100 / 12)) / (
            1 - (1 + df['InterestRate'] / 100 / 12) ** (-df['LoanTerm'])
        )
        df['payment_to_income_ratio'] = df['monthly_payment_estimate'] / ((df['Income'] / 12) + 1)
        
        # 5. Age-based features
        df['age_group_young'] = (df['Age'] < 30).astype(int)
        df['age_group_middle'] = ((df['Age'] >= 30) & (df['Age'] < 50)).astype(int)
        df['age_group_senior'] = (df['Age'] >= 50).astype(int)
        
        # 6. Credit score categories (industry standard)
        df['credit_poor'] = (df['CreditScore'] < 580).astype(int)
        df['credit_fair'] = ((df['CreditScore'] >= 580) & (df['CreditScore'] < 670)).astype(int)
        df['credit_good'] = ((df['CreditScore'] >= 670) & (df['CreditScore'] < 740)).astype(int)
        df['credit_very_good'] = ((df['CreditScore'] >= 740) & (df['CreditScore'] < 800)).astype(int)
        df['credit_excellent'] = (df['CreditScore'] >= 800).astype(int)
        
        # 7. Income brackets
        income_25 = np.percentile(df['Income'], 25)
        income_50 = np.percentile(df['Income'], 50)
        income_75 = np.percentile(df['Income'], 75)
        
        df['income_low'] = (df['Income'] <= income_25).astype(int)
        df['income_medium_low'] = ((df['Income'] > income_25) & (df['Income'] <= income_50)).astype(int)
        df['income_medium_high'] = ((df['Income'] > income_50) & (df['Income'] <= income_75)).astype(int)
        df['income_high'] = (df['Income'] > income_75).astype(int)
        
        # 8. Loan amount categories
        loan_25 = np.percentile(df['LoanAmount'], 25)
        loan_50 = np.percentile(df['LoanAmount'], 50)
        loan_75 = np.percentile(df['LoanAmount'], 75)
        
        df['loan_small'] = (df['LoanAmount'] <= loan_25).astype(int)
        df['loan_medium'] = ((df['LoanAmount'] > loan_25) & (df['LoanAmount'] <= loan_75)).astype(int)
        df['loan_large'] = (df['LoanAmount'] > loan_75).astype(int)
        
        # 9. Interest rate risk levels
        df['high_interest_rate'] = (df['InterestRate'] > 15).astype(int)
        df['medium_interest_rate'] = ((df['InterestRate'] >= 10) & (df['InterestRate'] <= 15)).astype(int)
        df['low_interest_rate'] = (df['InterestRate'] < 10).astype(int)
        
        # 10. Loan term features
        df['short_term_loan'] = (df['LoanTerm'] <= 24).astype(int)
        df['medium_term_loan'] = ((df['LoanTerm'] > 24) & (df['LoanTerm'] <= 48)).astype(int)
        df['long_term_loan'] = (df['LoanTerm'] > 48).astype(int)
        
        # 11. Combined risk indicators
        df['high_dti_high_interest'] = ((df['DTIRatio'] > 0.4) & (df['InterestRate'] > 15)).astype(int)
        df['low_credit_high_loan'] = ((df['CreditScore'] < 650) & (df['loan_to_income_ratio'] > 0.5)).astype(int)
        df['young_high_risk'] = ((df['Age'] < 25) & (df['loan_to_income_ratio'] > 0.4)).astype(int)
        
        # 12. Financial capacity score (composite metric)
        # Normalize key metrics to 0-1 scale then combine
        df['income_normalized'] = (df['Income'] - df['Income'].min()) / (df['Income'].max() - df['Income'].min() + 1)
        df['credit_normalized'] = (df['CreditScore'] - 300) / (850 - 300)  # Standard credit score range
        df['dti_normalized'] = 1 - df['DTIRatio']  # Lower DTI is better, so invert
        
        df['financial_capacity_score'] = (
            df['income_normalized'] * 0.4 +
            df['credit_normalized'] * 0.4 +
            df['dti_normalized'] * 0.2
        )
        
        # 13. Experience-derived features (if employment data available)
        if 'MonthsEmployed' in df.columns:
            df['underemployed'] = ((df['MonthsEmployed'] < 12) & (df['Age'] > 25)).astype(int)
            df['job_hopper'] = ((df['MonthsEmployed'] < 24) & (df['Age'] > 30)).astype(int)
            df['stable_employment'] = (df['MonthsEmployed'] > 60).astype(int)
        
        # 14. Log transformations for skewed features
        df['log_income'] = np.log1p(df['Income'])
        df['log_loan_amount'] = np.log1p(df['LoanAmount'])
        df['sqrt_credit_score'] = np.sqrt(df['CreditScore'])
    
    # Calculate percentile-based features from training data only (to avoid data leakage)
    train_percentiles = {
        'income_90th': train_feat['Income'].quantile(0.9),
        'loan_90th': train_feat['LoanAmount'].quantile(0.9),
        'credit_10th': train_feat['CreditScore'].quantile(0.1),
        'dti_90th': train_feat['DTIRatio'].quantile(0.9)
    }
    
    for df in [train_feat, test_feat]:
        df['high_income_applicant'] = (df['Income'] > train_percentiles['income_90th']).astype(int)
        df['large_loan_applicant'] = (df['LoanAmount'] > train_percentiles['loan_90th']).astype(int)
        df['poor_credit_applicant'] = (df['CreditScore'] < train_percentiles['credit_10th']).astype(int)
        df['high_dti_applicant'] = (df['DTIRatio'] > train_percentiles['dti_90th']).astype(int)
    
    # Clean up any inf/nan values created during feature engineering
    for df in [train_feat, test_feat]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill any new NaN values with median for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop('Default', errors='ignore')
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                fill_value = df[col].median()
                df[col].fillna(fill_value, inplace=True)
    
    logger.info(f"Feature engineering complete. Train: {train_feat.shape}, Test: {test_feat.shape}")
    logger.info(f"New features created: {len(set(train_feat.columns) - set(train.columns))}")
    
    return train_feat, test_feat
