from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def split_train_test(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split the raw data into train and test sets with stratification.
    """
    logger.info(f"Original data shape: {data.shape}")
    logger.info(f"Columns: {data.columns.tolist()}")
    
    # Check target distribution
    target_col = 'Risk_Flag'
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    
    target_dist = data[target_col].value_counts()
    logger.info(f"Target distribution: {target_dist.to_dict()}")
    logger.info(f"Target balance: {data[target_col].mean():.3f}")
    
    # Split data
    train, test = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=data[target_col]
    )
    
    logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test


def clean_loan_data(train: pd.DataFrame, test: pd.DataFrame):
    """
    Clean and prepare data with a simplified approach focused on core features.
    """
    logger.info(f"Starting data cleaning. Train: {train.shape}, Test: {test.shape}")
    
    # Work on copies
    train_clean = train.copy()
    test_clean = test.copy()
    
    # 1. Remove ID column (not predictive)
    for df in [train_clean, test_clean]:
        if 'Id' in df.columns:
            df.drop('Id', axis=1, inplace=True)
    
    # 2. Handle high-cardinality geographic features carefully
    # Create aggregated risk scores ONLY from training data to avoid leakage
    if 'STATE' in train_clean.columns:
        # Calculate state-level statistics from training data only
        state_stats = train_clean.groupby('STATE').agg({
            'Risk_Flag': ['count', 'mean'],
            'Income': 'median'
        }).round(4)
        
        state_stats.columns = ['state_count', 'state_risk_rate', 'state_median_income']
        
        # Only keep states with sufficient data (>=50 samples)
        valid_states = state_stats[state_stats['state_count'] >= 50].index
        
        # Create state features
        for df in [train_clean, test_clean]:
            # Binary feature: is from a high-risk state?
            high_risk_states = state_stats[state_stats['state_risk_rate'] > 0.15].index
            df['is_high_risk_state'] = df['STATE'].isin(high_risk_states).astype(int)
            
            # Income relative to state median
            df['state_income_ratio'] = df['Income'] / df['STATE'].map(state_stats['state_median_income']).fillna(6000000)
    
    # 3. Handle profession more intelligently
    if 'Profession' in train_clean.columns:
        # Group rare professions
        prof_counts = train_clean['Profession'].value_counts()
        common_profs = prof_counts[prof_counts >= 100].index
        
        for df in [train_clean, test_clean]:
            df['profession_grouped'] = df['Profession'].where(
                df['Profession'].isin(common_profs), 'Other'
            )
        
        # Create profession risk encoding from training data only
        prof_risk = train_clean.groupby('profession_grouped')['Risk_Flag'].mean()
        
        for df in [train_clean, test_clean]:
            df['profession_risk_score'] = df['profession_grouped'].map(prof_risk).fillna(prof_risk.mean())
    
    # 4. Handle city more simply
    if 'CITY' in train_clean.columns:
        # Just mark top 10 cities, drop the rest
        top_cities = train_clean['CITY'].value_counts().head(10).index
        for df in [train_clean, test_clean]:
            df['is_major_city'] = df['CITY'].isin(top_cities).astype(int)
    
    # 5. Drop the original high-cardinality columns
    drop_cols = ['STATE', 'CITY', 'Profession', 'profession_grouped']
    for df in [train_clean, test_clean]:
        df.drop([col for col in drop_cols if col in df.columns], axis=1, inplace=True)
    
    # 6. Handle missing values intelligently
    for df in [train_clean, test_clean]:
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col].fillna(mode_val[0], inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
        
        # Numerical columns - use median (more robust than mean)
        num_cols = df.select_dtypes(include=[np.number]).columns
        num_cols = num_cols.drop('Risk_Flag', errors='ignore')  # Don't fill target
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
    
    # 7. One-hot encode categorical variables
    categorical_features = ['Married/Single', 'House_Ownership', 'Car_Ownership']
    categorical_features = [col for col in categorical_features if col in train_clean.columns]
    
    if categorical_features:
        train_encoded = pd.get_dummies(train_clean, columns=categorical_features, drop_first=True)
        test_encoded = pd.get_dummies(test_clean, columns=categorical_features, drop_first=True)
        
        # Ensure both have the same columns
        train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)
    else:
        train_encoded, test_encoded = train_clean, test_clean
    
    # 8. Final validation
    logger.info(f"Final shapes - Train: {train_encoded.shape}, Test: {test_encoded.shape}")
    logger.info(f"Missing values - Train: {train_encoded.isnull().sum().sum()}, Test: {test_encoded.isnull().sum().sum()}")
    
    # Check for any infinite values
    train_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill any remaining NaNs
    train_encoded.fillna(0, inplace=True)
    test_encoded.fillna(0, inplace=True)
    
    return train_encoded, test_encoded


# Keeping the legacy functions for compatibility but they won't be used
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if "LoanID" in df.columns:
        df = df.drop(columns=["LoanID"])
    bool_cols = [col for col in ["HasMortgage", "HasDependents", "HasCoSigner"] if col in df.columns]
    if bool_cols:
        df[bool_cols] = df[bool_cols].replace({"Yes": 1, "No": 0})
    cat_cols = [col for col in ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"] if col in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype("category")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(thresh=int(0.5 * len(df.columns)))
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if all(col in df.columns for col in ["loanamount", "income"]):
        df["loan_to_income"] = df["loanamount"] / (df["income"] + 1)
    if all(col in df.columns for col in ["income", "numcreditlines"]):
        df["income_per_credit_line"] = df["income"] / (df["numcreditlines"] + 1)
    if "creditscore" in df.columns:
        df["credit_score_band"] = pd.cut(
            df["creditscore"],
            bins=[300, 580, 670, 740, 800, 850],
            labels=["Poor", "Fair", "Good", "Very Good", "Excellent"]
        ).astype("category")
    return df


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    target_col = 'risk_flag' if 'risk_flag' in df.columns else None
    if not target_col:
        raise KeyError("'risk_flag' not found in data for balancing.")
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Check for and clean missing values or infs
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.dropna()
    y = y.loc[X.index]  # Make sure y aligns with cleaned X

    X_encoded = pd.get_dummies(X, drop_first=True)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_encoded, y)

    df_balanced = pd.DataFrame(X_res, columns=X_encoded.columns)
    df_balanced[target_col] = y_res
    return df_balanced


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_col = 'risk_flag' if 'risk_flag' in df.columns else None
    if not target_col:
        raise KeyError("'risk_flag' not found in data for splitting.")
    return train_test_split(df, test_size=0.2, stratify=df[target_col], random_state=42)