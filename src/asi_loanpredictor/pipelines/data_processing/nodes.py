import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def feature_engineering(train: pd.DataFrame, test: pd.DataFrame):
    """
    Create essential engineered features that are most predictive for loan risk.
    Focus on simplicity and interpretability.
    """
    logger.info(f"Starting feature engineering. Train: {train.shape}, Test: {test.shape}")
    
    train_feat = train.copy()
    test_feat = test.copy()
    
    # Validate required columns exist
    required_cols = ['Income', 'Age', 'Experience']
    missing_cols = [col for col in required_cols if col not in train_feat.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
        return train_feat, test_feat
    
    for df in [train_feat, test_feat]:
        # 1. Income-related features (most important for loan decisions)
        df['income_thousands'] = df['Income'] / 1000  # Scale down for better model performance
        df['log_income'] = np.log1p(df['Income'])  # Log transform for skewed income distribution
        
        # 2. Age and experience ratios
        df['experience_age_ratio'] = df['Experience'] / (df['Age'] + 1)  # Add 1 to avoid division by zero
        df['income_per_age'] = df['Income'] / (df['Age'] + 1)
        
        # 3. Employment stability indicators
        if 'CURRENT_JOB_YRS' in df.columns:
            df['job_stability'] = df['CURRENT_JOB_YRS'] / (df['Experience'] + 1)
            df['income_per_job_year'] = df['Income'] / (df['CURRENT_JOB_YRS'] + 1)
        
        # 4. Life stage indicators (simple and interpretable)
        df['young_professional'] = ((df['Age'] < 30) & (df['Experience'] > 2)).astype(int)
        df['experienced_worker'] = (df['Experience'] > 15).astype(int)
        df['early_career'] = (df['Experience'] < 3).astype(int)
        
        # 5. Financial capacity indicators
        df['high_income'] = (df['Income'] > 8000000).astype(int)  # Above 8L
        df['low_income'] = (df['Income'] < 3000000).astype(int)   # Below 3L
        
        # 6. Experience-age consistency check
        df['experience_age_gap'] = df['Age'] - df['Experience'] - 18  # Assuming work starts at 18
        df['unusual_experience'] = (df['experience_age_gap'] < 0).astype(int)  # Started work too early?
        
        # 7. Housing stability (if available)
        if 'CURRENT_HOUSE_YRS' in df.columns:
            df['housing_stability'] = (df['CURRENT_HOUSE_YRS'] > 5).astype(int)
            df['recently_moved'] = (df['CURRENT_HOUSE_YRS'] < 2).astype(int)
    
    # 8. Income percentiles (calculated from training data only to avoid leakage)
    income_percentiles = train_feat['Income'].quantile([0.25, 0.5, 0.75]).values
    
    for df in [train_feat, test_feat]:
        df['income_quartile_1'] = (df['Income'] <= income_percentiles[0]).astype(int)
        df['income_quartile_2'] = ((df['Income'] > income_percentiles[0]) & 
                                 (df['Income'] <= income_percentiles[1])).astype(int)
        df['income_quartile_3'] = ((df['Income'] > income_percentiles[1]) & 
                                 (df['Income'] <= income_percentiles[2])).astype(int)
        df['income_quartile_4'] = (df['Income'] > income_percentiles[2]).astype(int)
    
    # 9. Clean up any inf/nan values created during feature engineering
    for df in [train_feat, test_feat]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill any new NaN values with median/mode
        for col in df.columns:
            if df[col].dtype == 'object':
                continue  # Skip non-numeric columns
            if df[col].isnull().sum() > 0:
                fill_value = df[col].median() if df[col].dtype in ['int64', 'float64'] else 0
                df[col].fillna(fill_value, inplace=True)
    
    logger.info(f"Feature engineering complete. Train: {train_feat.shape}, Test: {test_feat.shape}")
    logger.info(f"New features created: {set(train_feat.columns) - set(train.columns)}")
    
    return train_feat, test_feat
