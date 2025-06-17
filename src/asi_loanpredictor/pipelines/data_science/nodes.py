import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
    y = train[target_col]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

def evaluate_model(model, features, test: pd.DataFrame):
    """
    Evaluate the trained model on the test set.
    Returns accuracy.
    """
    target_col = 'Risk_Flag'
    X_test = test[features]
    if 'Id' in X_test.columns:
        X_test = X_test.drop(columns=['Id'])
    y_test = test[target_col]
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
