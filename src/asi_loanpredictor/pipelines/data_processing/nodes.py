import pandas as pd
from sklearn.preprocessing import StandardScaler

def feature_engineering(train: pd.DataFrame, test: pd.DataFrame):
    """
    Perform feature engineering (e.g., scaling numeric features).
    Returns processed train and test DataFrames.
    """
    target_col = 'Loan_Status' if 'Loan_Status' in train.columns else None
    X_train = train.drop(columns=[target_col]) if target_col else train.copy()
    X_test = test.drop(columns=[target_col]) if target_col and target_col in test.columns else test.copy()
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    if target_col:
        X_train_scaled[target_col] = train[target_col].values
        if target_col in test.columns:
            X_test_scaled[target_col] = test[target_col].values
    return X_train_scaled, X_test_scaled
