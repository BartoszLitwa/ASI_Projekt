import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.validation import check_is_fitted
import pickle
import logging

logger = logging.getLogger(__name__)

def train_model(train: pd.DataFrame, parameters: dict):
    """
    Train a RandomForestClassifier on the training data with improved hyperparameters.
    Returns the trained model, features used, and training metrics.
    """
    target_col = 'Risk_Flag'
    X = train.drop(columns=[target_col])
    if 'Id' in X.columns:
        X = X.drop(columns=['Id'])
    
    # Convert all boolean columns to int
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
    
    y = train[target_col]
    
    # Log data info
    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"Target balance: {y.mean():.3f}")
    
    # Get model parameters
    rf_params = parameters.get('random_forest', {})
    
    # Create and train model with better parameters
    model = RandomForestClassifier(
        n_estimators=rf_params.get('n_estimators', 300),
        max_depth=rf_params.get('max_depth', 15),
        min_samples_split=rf_params.get('min_samples_split', 10),
        min_samples_leaf=rf_params.get('min_samples_leaf', 4),
        max_features=rf_params.get('max_features', 'sqrt'),
        class_weight=rf_params.get('class_weight', 'balanced'),
        random_state=rf_params.get('random_state', 42),
        n_jobs=-1  # Use all available cores
    )
    
    model.fit(X, y)
    
    # Verify the model is fitted
    try:
        check_is_fitted(model)
    except Exception as e:
        raise RuntimeError("Model fitting failed.") from e
    
    # Calculate feature importances
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 most important features:")
    logger.info(feature_importance.head(10).to_string())
    
    # Cross-validation scores
    eval_params = parameters.get('evaluation', {})
    cv_folds = eval_params.get('cv_folds', 5)
    
    cv_scores = {}
    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        scores = cross_val_score(model, X, y, cv=kfold, scoring=metric, n_jobs=-1)
        cv_scores[f'cv_{metric}'] = {
            'mean': scores.mean(),
            'std': scores.std()
        }
        logger.info(f"CV {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return model, X.columns.tolist(), feature_importance, cv_scores

def evaluate_model(model, features, test: pd.DataFrame, parameters: dict):
    """
    Evaluate the trained model on the test set with comprehensive metrics.
    Returns detailed evaluation metrics.
    """
    target_col = 'Risk_Flag'
    
    # Prepare test data
    X_test = test[features].copy()
    if 'Id' in X_test.columns:
        X_test = X_test.drop(columns=['Id'])
    
    # Convert all boolean columns to int
    for col in X_test.select_dtypes(include=['bool']).columns:
        X_test[col] = X_test[col].astype(int)
    
    y_test = test[target_col]
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate comprehensive metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Log results
    logger.info("Test Set Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric.upper()}: {value:.4f}")
    
    # Detailed classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # Additional analysis for imbalanced dataset
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics.update({
        'specificity': specificity,
        'sensitivity': sensitivity,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    })
    
    logger.info(f"Specificity (True Negative Rate): {specificity:.4f}")
    logger.info(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
    
    return metrics
