import pandas as pd
from sklearn.preprocessing import StandardScaler

def feature_engineering(train: pd.DataFrame, test: pd.DataFrame):
    """
    Perform feature engineering (e.g., scaling numeric features).
    Returns processed train and test DataFrames.
    """
    target_col = 'Risk_Flag' if 'Risk_Flag' in train.columns else None
    X_train = train.drop(columns=[target_col]) if target_col else train.copy()
    X_test = test.drop(columns=[target_col]) if target_col and target_col in test.columns else test.copy()
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    if target_col:
        X_train_scaled[target_col] = train[target_col].values
        if target_col in test.columns:
            X_test_scaled[target_col] = test[target_col].values
    # Drop Id column if present
    if 'Id' in X_train_scaled.columns:
        X_train_scaled = X_train_scaled.drop(columns=['Id'])
    if 'Id' in X_test_scaled.columns:
        X_test_scaled = X_test_scaled.drop(columns=['Id'])
    X_train_scaled = X_train_scaled.reset_index(drop=True)
    X_test_scaled = X_test_scaled.reset_index(drop=True)
    return X_train_scaled, X_test_scaled
