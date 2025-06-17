from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import SMOTE
from kedro.io import DataCatalog
from kedro.pipeline import node


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


def split_train_test(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split the raw data into train and test sets.
    """
    target_col = 'Risk_Flag' if 'Risk_Flag' in data.columns else None
    train, test = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data[target_col] if target_col else None)
    return train, test


def clean_loan_data(train: pd.DataFrame, test: pd.DataFrame):
    # Drop high-cardinality and non-numeric columns for simplicity
    drop_cols = [col for col in ['Id', 'Profession', 'CITY', 'STATE'] if col in train.columns]
    train = train.drop(columns=drop_cols)
    test = test.drop(columns=drop_cols)
    # Fill missing values
    for df in [train, test]:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        for col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].fillna(df[col].median())
    # Only one-hot encode a few key categoricals
    cat_cols = [col for col in ['Married/Single', 'House_Ownership', 'Car_Ownership'] if col in train.columns]
    train = pd.get_dummies(train, columns=cat_cols, drop_first=True)
    test = pd.get_dummies(test, columns=cat_cols, drop_first=True)
    # Align columns
    train, test = train.align(test, join='left', axis=1, fill_value=0)
    return train, test

# Remove advanced/unused nodes for simplicity