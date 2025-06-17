import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_model(train: pd.DataFrame):
    """
    Train a RandomForestClassifier on the training data.
    Returns the trained model and the features used.
    """
    target_col = 'Loan_Status'
    X = train.drop(columns=[target_col])
    y = train[target_col].map({'Y': 1, 'N': 0}) if train[target_col].dtype == object else train[target_col]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

def evaluate_model(model, features, test: pd.DataFrame):
    """
    Evaluate the trained model on the test set.
    Returns accuracy.
    """
    target_col = 'Loan_Status'
    X_test = test[features]
    y_test = test[target_col].map({'Y': 1, 'N': 0}) if test[target_col].dtype == object else test[target_col]
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
