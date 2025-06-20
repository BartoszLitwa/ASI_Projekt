import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

def feature_engineering(train: pd.DataFrame, test: pd.DataFrame, parameters: dict = None):
    """
    Enhanced feature engineering for loan prediction.
    Creates meaningful features from the raw data.
    """
    # Get feature engineering parameters
    if parameters is None:
        parameters = {}
    fe_params = parameters.get('feature_engineering', {})
    
    logger.info(f"Starting feature engineering. Train shape: {train.shape}, Test shape: {test.shape}")
    
    # Work on copies to avoid modifying original data
    train_processed = train.copy()
    test_processed = test.copy()
    
    for df in [train_processed, test_processed]:
        # 1. Income-based features
        if 'Income' in df.columns and 'CURRENT_JOB_YRS' in df.columns:
            df['income_per_year_of_job'] = df['Income'] / (df['CURRENT_JOB_YRS'] + 1)
            df['income_stability'] = df['Income'] * df['CURRENT_JOB_YRS']
        
        # 2. Age-based features
        if 'Age' in df.columns:
            # Age buckets
            age_bins = fe_params.get('age_bins', [18, 30, 45, 60, 100])
            df['age_bucket'] = pd.cut(df['Age'], bins=age_bins, labels=['young', 'middle', 'senior', 'elderly'][:len(age_bins)-1])
            
            # Age-Experience relationship
            if 'Experience' in df.columns:
                df['experience_age_ratio'] = df['Experience'] / df['Age']
                df['early_career'] = (df['Experience'] < 5).astype(int)
                df['experienced'] = (df['Experience'] > 15).astype(int)
        
        # 3. Experience-based features
        if 'Experience' in df.columns:
            exp_bins = fe_params.get('experience_bins', [0, 2, 5, 10, 20, 50])
            df['experience_level'] = pd.cut(df['Experience'], bins=exp_bins, 
                                          labels=['novice', 'junior', 'mid', 'senior', 'expert'][:len(exp_bins)-1])
        
        # 4. Housing-related features
        if 'CURRENT_HOUSE_YRS' in df.columns:
            df['housing_stability'] = df['CURRENT_HOUSE_YRS'] > 5
            
        # 5. Income bins for better categorical representation
        if 'Income' in df.columns:
            income_bins = fe_params.get('income_bins', [0, 30000, 60000, 100000, 150000, float('inf')])
            df['income_bracket'] = pd.cut(df['Income'], bins=income_bins, 
                                        labels=['low', 'medium', 'high', 'very_high', 'wealthy'][:len(income_bins)-1])
        
        # 6. Marital status and ownership interactions
        if 'Married/Single' in df.columns and 'House_Ownership' in df.columns:
            df['married_homeowner'] = ((df['Married/Single'] == 'married') & 
                                     (df['House_Ownership'] == 'owned')).astype(int)
        
        # 7. Risk indicators
        if 'Age' in df.columns and 'Experience' in df.columns:
            df['age_experience_mismatch'] = (df['Age'] - df['Experience'] < 18).astype(int)
        
        if 'Income' in df.columns and 'Age' in df.columns:
            df['income_age_ratio'] = df['Income'] / df['Age']
    
    # Handle categorical variables with proper encoding
    categorical_cols = []
    
    # One-hot encode key categorical variables
    for col in ['Married/Single', 'House_Ownership', 'Car_Ownership']:
        if col in train_processed.columns:
            categorical_cols.append(col)
    
    # Add the new categorical features
    for col in ['age_bucket', 'experience_level', 'income_bracket']:
        if col in train_processed.columns:
            categorical_cols.append(col)
    
    # Apply one-hot encoding
    train_encoded = pd.get_dummies(train_processed, columns=categorical_cols, drop_first=True)
    test_encoded = pd.get_dummies(test_processed, columns=categorical_cols, drop_first=True)
    
    # Align columns between train and test
    train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)
    
    # Handle any remaining missing values
    numeric_cols = train_encoded.select_dtypes(include=[np.number]).columns
    train_encoded[numeric_cols] = train_encoded[numeric_cols].fillna(train_encoded[numeric_cols].median())
    test_encoded[numeric_cols] = test_encoded[numeric_cols].fillna(train_encoded[numeric_cols].median())
    
    # Log feature engineering results
    logger.info(f"Feature engineering completed. Train shape: {train_encoded.shape}, Test shape: {test_encoded.shape}")
    logger.info(f"New features created: {set(train_encoded.columns) - set(train.columns)}")
    
    # Ensure no infinite values
    train_encoded = train_encoded.replace([np.inf, -np.inf], np.nan)
    test_encoded = test_encoded.replace([np.inf, -np.inf], np.nan)
    
    # Fill any remaining NaN values
    train_encoded = train_encoded.fillna(0)
    test_encoded = test_encoded.fillna(0)
    
    return train_encoded, test_encoded
