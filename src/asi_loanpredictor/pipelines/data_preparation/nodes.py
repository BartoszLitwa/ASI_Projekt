from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from imblearn.over_sampling import SMOTE
from kedro.io import DataCatalog
from kedro.pipeline import node
import logging

logger = logging.getLogger(__name__)

def split_train_test(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split the raw data into train and test sets.
    """
    logger.info(f"Splitting data with shape {data.shape}")
    
    target_col = 'Risk_Flag' if 'Risk_Flag' in data.columns else None
    if target_col is None:
        raise KeyError("'Risk_Flag' column not found in data")
    
    logger.info(f"Target distribution: {data[target_col].value_counts().to_dict()}")
    
    train, test = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=data[target_col] if target_col else None
    )
    
    logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test


def clean_loan_data(train: pd.DataFrame, test: pd.DataFrame):
    """
    Enhanced data cleaning that preserves more useful information.
    """
    logger.info(f"Starting data cleaning. Train: {train.shape}, Test: {test.shape}")
    
    # Work on copies
    train_clean = train.copy()
    test_clean = test.copy()
    
    # Handle high-cardinality categorical variables intelligently
    # Instead of dropping, we'll encode them properly
    
    # 1. Profession - group rare professions
    if 'Profession' in train_clean.columns:
        profession_counts = train_clean['Profession'].value_counts()
        rare_professions = profession_counts[profession_counts < 50].index
        
        for df in [train_clean, test_clean]:
            df['Profession'] = df['Profession'].replace(rare_professions, 'Other')
            # Create profession risk encoding based on default rates
        
        # Calculate profession risk scores on training data
        profession_risk = train_clean.groupby('Profession')['Risk_Flag'].mean()
        
        for df in [train_clean, test_clean]:
            df['profession_risk_score'] = df['Profession'].map(profession_risk).fillna(profession_risk.mean())
    
    # 2. Geographic features - create state-level features
    if 'STATE' in train_clean.columns:
        state_risk = train_clean.groupby('STATE')['Risk_Flag'].mean()
        state_income = train_clean.groupby('STATE')['Income'].mean()
        
        for df in [train_clean, test_clean]:
            df['state_risk_score'] = df['STATE'].map(state_risk).fillna(state_risk.mean())
            df['state_avg_income'] = df['STATE'].map(state_income).fillna(state_income.mean())
            df['income_vs_state_avg'] = df['Income'] / df['state_avg_income']
    
    # 3. City features - create city risk scores for top cities
    if 'CITY' in train_clean.columns:
        city_counts = train_clean['CITY'].value_counts()
        top_cities = city_counts.head(20).index
        
        city_risk = train_clean[train_clean['CITY'].isin(top_cities)].groupby('CITY')['Risk_Flag'].mean()
        
        for df in [train_clean, test_clean]:
            df['is_top_city'] = df['CITY'].isin(top_cities).astype(int)
            df['city_risk_score'] = df['CITY'].map(city_risk).fillna(city_risk.mean() if len(city_risk) > 0 else 0.5)
    
    # Now drop the original high-cardinality columns since we've extracted useful info
    drop_cols = [col for col in ['Id', 'Profession', 'CITY', 'STATE'] if col in train_clean.columns]
    train_clean = train_clean.drop(columns=drop_cols)
    test_clean = test_clean.drop(columns=drop_cols)
    
    # Handle missing values more intelligently
    for df in [train_clean, test_clean]:
        # For categorical columns, use mode
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                df[col] = df[col].fillna(fill_value)
        
        # For numerical columns, use median for better robustness
        for col in df.select_dtypes(include=['number']).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
    
    # Handle specific categorical columns with better encoding
    cat_cols = [col for col in ['Married/Single', 'House_Ownership', 'Car_Ownership'] if col in train_clean.columns]
    
    # One-hot encode key categorical variables
    train_encoded = pd.get_dummies(train_clean, columns=cat_cols, drop_first=True)
    test_encoded = pd.get_dummies(test_clean, columns=cat_cols, drop_first=True)
    
    # Align columns between train and test (important for consistency)
    train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)
    
    # Data quality checks
    logger.info(f"After cleaning - Train: {train_encoded.shape}, Test: {test_encoded.shape}")
    logger.info(f"Missing values in train: {train_encoded.isnull().sum().sum()}")
    logger.info(f"Missing values in test: {test_encoded.isnull().sum().sum()}")
    
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