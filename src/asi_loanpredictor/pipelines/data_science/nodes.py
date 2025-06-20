import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_score
import logging
import json

logger = logging.getLogger(__name__)

def train_model(train_features: pd.DataFrame):
    """
    Train a Random Forest model optimized for loan risk prediction.
    Focus on balanced performance between precision and recall for risky loans.
    """
    logger.info(f"Training model with data shape: {train_features.shape}")
    
    # Separate features and target
    target_col = 'Risk_Flag'
    if target_col not in train_features.columns:
        raise KeyError(f"Target column '{target_col}' not found")
    
    X = train_features.drop(target_col, axis=1)
    y = train_features[target_col]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"Feature columns: {X.columns.tolist()}")
    
    # Handle class imbalance with balanced Random Forest
    # Slightly more conservative parameters for better precision
    model = RandomForestClassifier(
        n_estimators=250,           # More trees for stability
        max_depth=12,               # Reduced depth to prevent overfitting
        min_samples_split=30,       # Higher to prevent overfitting on small groups
        min_samples_leaf=15,        # Ensure leaves have meaningful samples
        max_features='sqrt',        # Use sqrt of features to reduce overfitting
        class_weight='balanced_subsample',  # More aggressive class balancing
        random_state=42,
        n_jobs=-1                   # Use all available cores
    )
    
    # Train the model
    logger.info("Training Random Forest model...")
    model.fit(X, y)
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 most important features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Cross-validation to check model stability
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    logger.info(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Find optimal threshold for business metrics
    y_proba = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_proba)
    
    # Business-focused threshold optimization
    # We want to maximize F1 score but prioritize precision slightly
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Find threshold that maximizes F1 but with minimum 60% precision for risky loans
    valid_indices = precision >= 0.60
    if np.any(valid_indices):
        valid_f1 = f1_scores[valid_indices]
        best_idx = np.argmax(valid_f1)
        optimal_threshold = thresholds[np.where(valid_indices)[0][best_idx]]
    else:
        # Fallback to best F1 score
        optimal_threshold = thresholds[np.argmax(f1_scores)]
    
    logger.info(f"Optimal threshold for business use: {optimal_threshold:.4f}")
    
    return model, X.columns.tolist(), optimal_threshold

def evaluate_model(model, test_features: pd.DataFrame, optimal_threshold=None):
    """
    Comprehensive model evaluation focusing on loan risk prediction metrics.
    """
    logger.info(f"Evaluating model with test data shape: {test_features.shape}")
    
    # Separate features and target
    target_col = 'Risk_Flag'
    X_test = test_features.drop(target_col, axis=1)
    y_test = test_features[target_col]
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of being risky
    
    # Use optimal threshold if provided, otherwise use default 0.5
    threshold = optimal_threshold if optimal_threshold is not None else 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    logger.info(f"Using decision threshold: {threshold:.4f}")
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Detailed classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Log detailed results
    logger.info("=== MODEL EVALUATION RESULTS ===")
    logger.info(f"Decision Threshold: {threshold:.4f}")
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
    logger.info(f"Overall Accuracy: {class_report['accuracy']:.4f}")
    
    logger.info("\nClass-wise Performance:")
    logger.info(f"Class 0 (Safe Loans):")
    logger.info(f"  Precision: {class_report['0']['precision']:.4f}")
    logger.info(f"  Recall: {class_report['0']['recall']:.4f}")
    logger.info(f"  F1-score: {class_report['0']['f1-score']:.4f}")
    
    logger.info(f"Class 1 (Risky Loans):")
    logger.info(f"  Precision: {class_report['1']['precision']:.4f}")
    logger.info(f"  Recall: {class_report['1']['recall']:.4f}")
    logger.info(f"  F1-score: {class_report['1']['f1-score']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
    logger.info(f"  False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
    
    # Calculate business metrics
    total_loans = len(y_test)
    risky_loans = (y_test == 1).sum()
    safe_loans = (y_test == 0).sum()
    
    # How many risky loans would we approve? (False Negatives)
    false_negatives = cm[1,0]
    # How many safe loans would we reject? (False Positives)  
    false_positives = cm[0,1]
    
    logger.info(f"\nBusiness Impact Analysis:")
    logger.info(f"  Total loan applications: {total_loans}")
    logger.info(f"  Actually risky loans: {risky_loans} ({risky_loans/total_loans:.1%})")
    logger.info(f"  Risky loans we'd approve (BAD): {false_negatives} ({false_negatives/risky_loans:.1%} of risky)")
    logger.info(f"  Safe loans we'd reject (LOST BUSINESS): {false_positives} ({false_positives/safe_loans:.1%} of safe)")
    
    # Calculate potential financial impact
    avg_loan_amount = 1000000  # Assume 10L average loan
    bad_loan_loss_rate = 0.7   # Assume 70% loss on bad loans
    
    potential_loss = false_negatives * avg_loan_amount * bad_loan_loss_rate
    lost_revenue = false_positives * avg_loan_amount * 0.1  # Assume 10% profit margin
    
    logger.info(f"\nEstimated Financial Impact (for {total_loans} loans):")
    logger.info(f"  Potential loss from bad loans: ₹{potential_loss/10000000:.1f} Cr")
    logger.info(f"  Lost revenue from rejected good loans: ₹{lost_revenue/10000000:.1f} Cr")
    
    # Full classification report for detailed analysis
    logger.info("\nDetailed Classification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Prepare evaluation results to save
    evaluation_data = {
        'threshold': float(threshold),
        'roc_auc': float(roc_auc),
        'accuracy': float(class_report['accuracy']),
        'precision_safe': float(class_report['0']['precision']),
        'recall_safe': float(class_report['0']['recall']),
        'f1_safe': float(class_report['0']['f1-score']),
        'precision_risky': float(class_report['1']['precision']),
        'recall_risky': float(class_report['1']['recall']),
        'f1_risky': float(class_report['1']['f1-score']),
        'confusion_matrix': cm.tolist(),
        'business_metrics': {
            'total_loans': int(total_loans),
            'risky_loans': int(risky_loans),
            'false_negatives': int(false_negatives),
            'false_positives': int(false_positives),
            'risky_approval_rate': float(false_negatives/risky_loans),
            'safe_rejection_rate': float(false_positives/safe_loans),
            'potential_loss_cr': float(potential_loss/10000000),
            'lost_revenue_cr': float(lost_revenue/10000000)
        }
    }
    
    return evaluation_data
