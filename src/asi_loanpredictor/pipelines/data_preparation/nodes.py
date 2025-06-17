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
    bool_cols = ["HasMortgage", "HasDependents", "HasCoSigner"]
    df[bool_cols] = df[bool_cols].replace({"Yes": 1, "No": 0})
    cat_cols = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]
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
        if df[col].dtype.name == 'category':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["loan_to_income"] = df["loanamount"] / (df["income"] + 1)
    df["income_per_credit_line"] = df["income"] / (df["numcreditlines"] + 1)
    df["credit_score_band"] = pd.cut(
        df["creditscore"],
        bins=[300, 580, 670, 740, 800, 850],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"]
    ).astype("category")
    return df


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop("default", axis=1)
    y = df["default"].map({"No": 0, "Yes": 1})

    # Check for and clean missing values or infs
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.dropna()
    y = y.loc[X.index]  # Make sure y aligns with cleaned X

    X_encoded = pd.get_dummies(X, drop_first=True)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_encoded, y)

    df_balanced = pd.DataFrame(X_res, columns=X_encoded.columns)
    df_balanced["default"] = y_res
    return df_balanced


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=0.2, stratify=df["default"], random_state=42)


def split_train_test(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split the raw data into train and test sets.
    """
    train, test = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data['Loan_Status'] if 'Loan_Status' in data.columns else None)
    return train, test


def clean_loan_data(train: pd.DataFrame, test: pd.DataFrame):
    """
    Clean the train and test data: handle missing values, encode categoricals, etc.
    Returns cleaned train and test DataFrames.
    """
    # Example: fill missing values with mode for categoricals, median for numerics
    for df in [train, test]:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        for col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].fillna(df[col].median())
    # Example: encode categoricals using pandas get_dummies (except target)
    target_col = 'Loan_Status' if 'Loan_Status' in train.columns else None
    train_features = train.drop(columns=[target_col]) if target_col else train.copy()
    test_features = test.drop(columns=[target_col]) if target_col and target_col in test.columns else test.copy()
    all_data = pd.concat([train_features, test_features], axis=0)
    all_data_encoded = pd.get_dummies(all_data)
    train_encoded = all_data_encoded.iloc[:len(train_features), :]
    test_encoded = all_data_encoded.iloc[len(train_features):, :]
    if target_col:
        train_encoded[target_col] = train[target_col].values
        if target_col in test.columns:
            test_encoded[target_col] = test[target_col].values
    return train_encoded, test_encoded