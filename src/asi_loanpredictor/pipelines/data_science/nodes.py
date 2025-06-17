import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted
import pickle

def train_model(train: pd.DataFrame):
    """
    Train a RandomForestClassifier on the training data.
    Returns the trained model and the features used.
    """
    target_col = 'Risk_Flag'
    X = train.drop(columns=[target_col])
    if 'Id' in X.columns:
        X = X.drop(columns=['Id'])
    # Convert all boolean columns to int
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
    y = train[target_col]
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    
    # Verify the model is fitted before returning
    try:
        check_is_fitted(model)
    except Exception as e:
        raise RuntimeError("Model fitting failed.") from e

    return model, X.columns.tolist()

def evaluate_model(model, features, test: pd.DataFrame):
    """
    Evaluate the trained model on the test set.
    Returns accuracy.
    """
    target_col = 'Risk_Flag'
    # Make an explicit copy to avoid SettingWithCopyWarning
    X_test = test[features].copy()
    if 'Id' in X_test.columns:
        X_test = X_test.drop(columns=['Id'])
    # Convert all boolean columns to int
    for col in X_test.select_dtypes(include=['bool']).columns:
        X_test[col] = X_test[col].astype(int)
    y_test = test[target_col]
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
