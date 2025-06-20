from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def split_train_test(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split the raw data into train and test sets with stratification.
    Updated for new dataset with 'Default' target column.
    """
    logger.info(f"Original data shape: {data.shape}")
    logger.info(f"Columns: {data.columns.tolist()}")
    
    # Check target distribution - updated target column name
    target_col = 'Default'
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    
    target_dist = data[target_col].value_counts()
    logger.info(f"Target distribution: {target_dist.to_dict()}")
    logger.info(f"Default rate: {data[target_col].mean():.3f}")
    
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
    Clean and prepare loan data with the new dataset structure.
    Features: LoanID, Age, Income, LoanAmount, CreditScore, MonthsEmployed, 
             NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, 
             EmploymentType, MaritalStatus, HasMortgage, HasDependents, 
             LoanPurpose, HasCoSigner, Default
    """
    logger.info(f"Starting data cleaning. Train: {train.shape}, Test: {test.shape}")
    
    # Work on copies
    train_clean = train.copy()
    test_clean = test.copy()
    
    # 1. Remove ID column (not predictive)
    for df in [train_clean, test_clean]:
        if 'LoanID' in df.columns:
            df.drop('LoanID', axis=1, inplace=True)
    
    # 2. Handle boolean columns - convert Yes/No to 1/0
    bool_columns = ['HasMortgage', 'HasDependents', 'HasCoSigner']
    for df in [train_clean, test_clean]:
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # 3. Handle categorical columns intelligently
    # Education - ordinal encoding (from lowest to highest education level)
    education_order = {
        'High School': 1,
        "Bachelor's": 2,
        "Master's": 3,
        'PhD': 4
    }
    
    for df in [train_clean, test_clean]:
        if 'Education' in df.columns:
            df['education_level'] = df['Education'].map(education_order).fillna(1)  # Default to High School
    
    # 4. Employment Type - risk-based encoding
    # Calculate employment risk from training data only
    if 'EmploymentType' in train_clean.columns:
        employment_risk = train_clean.groupby('EmploymentType')['Default'].mean()
        logger.info(f"Employment risk by type: {employment_risk.to_dict()}")
        
        for df in [train_clean, test_clean]:
            df['employment_risk_score'] = df['EmploymentType'].map(employment_risk).fillna(employment_risk.mean())
    
    # 5. Marital Status - simple binary encoding
    for df in [train_clean, test_clean]:
        if 'MaritalStatus' in df.columns:
            df['is_married'] = (df['MaritalStatus'] == 'Married').astype(int)
            df['is_divorced'] = (df['MaritalStatus'] == 'Divorced').astype(int)
            df['is_single'] = (df['MaritalStatus'] == 'Single').astype(int)
    
    # 6. Loan Purpose - risk-based encoding
    if 'LoanPurpose' in train_clean.columns:
        purpose_risk = train_clean.groupby('LoanPurpose')['Default'].mean()
        logger.info(f"Loan purpose risk: {purpose_risk.to_dict()}")
        
        for df in [train_clean, test_clean]:
            df['purpose_risk_score'] = df['LoanPurpose'].map(purpose_risk).fillna(purpose_risk.mean())
    
    # 7. Drop original categorical columns that we've encoded
    drop_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']
    for df in [train_clean, test_clean]:
        df.drop([col for col in drop_cols if col in df.columns], axis=1, inplace=True)
    
    # 8. Handle missing values
    for df in [train_clean, test_clean]:
        # Numerical columns - use median
        num_cols = df.select_dtypes(include=[np.number]).columns
        num_cols = num_cols.drop('Default', errors='ignore')  # Don't fill target
        
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {df[col].isnull().sum()} missing values in {col} with median {median_val}")
    
    # 9. Handle outliers for key numerical features
    numerical_features = ['Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
                         'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
    
    for feature in numerical_features:
        if feature in train_clean.columns:
            # Calculate outlier bounds from training data only
            Q1 = train_clean[feature].quantile(0.25)
            Q3 = train_clean[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            for df in [train_clean, test_clean]:
                # Cap outliers instead of removing them
                df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
    
    # 10. Final validation
    logger.info(f"Final shapes - Train: {train_clean.shape}, Test: {test_clean.shape}")
    logger.info(f"Missing values - Train: {train_clean.isnull().sum().sum()}, Test: {test_clean.isnull().sum().sum()}")
    
    # Check for any infinite values
    train_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill any remaining NaNs
    train_clean.fillna(0, inplace=True)
    test_clean.fillna(0, inplace=True)
    
    logger.info(f"Final column names: {train_clean.columns.tolist()}")
    
    return train_clean, test_clean


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