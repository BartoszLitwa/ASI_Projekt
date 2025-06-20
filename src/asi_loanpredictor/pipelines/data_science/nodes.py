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
    Train a Random Forest model optimized for loan default prediction.
    Focus on balanced performance between precision and recall for default prediction.
    """
    logger.info(f"Training model with data shape: {train_features.shape}")
    
    # Separate features and target - updated for new dataset
    target_col = 'Default'
    if target_col not in train_features.columns:
        raise KeyError(f"Target column '{target_col}' not found")
    
    X = train_features.drop(target_col, axis=1)
    y = train_features[target_col]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"Default rate: {y.mean():.3f}")
    logger.info(f"Feature columns count: {len(X.columns)}")
    
    # Handle class imbalance with balanced Random Forest
    # Optimized parameters for loan default prediction
    model = RandomForestClassifier(
        n_estimators=300,           # More trees for stability with larger feature set
        max_depth=15,               # Slightly deeper for complex feature interactions
        min_samples_split=20,       # Prevent overfitting
        min_samples_leaf=10,        # Ensure meaningful leaf nodes
        max_features='sqrt',        # Use sqrt of features to reduce overfitting
        class_weight='balanced',    # Handle class imbalance
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
    
    logger.info("Top 15 most important features:")
    for idx, row in feature_importance.head(15).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Cross-validation to check model stability
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    logger.info(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Find optimal threshold for business metrics
    y_proba = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_proba)
    
    # Business-focused threshold optimization for loan defaults
    # We want to catch defaults (high recall) while maintaining reasonable precision
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Find threshold that maximizes F1 but with minimum 50% precision for defaults
    valid_indices = precision >= 0.50
    if np.any(valid_indices):
        valid_f1 = f1_scores[valid_indices]
        best_idx = np.argmax(valid_f1)
        optimal_threshold = thresholds[np.where(valid_indices)[0][best_idx]]
    else:
        # Fallback to best F1 score
        optimal_threshold = thresholds[np.argmax(f1_scores)]
    
    logger.info(f"Optimal threshold for default prediction: {optimal_threshold:.4f}")
    
    return model, X.columns.tolist(), optimal_threshold

def evaluate_model(model, test_features: pd.DataFrame, optimal_threshold=None):
    """
    Comprehensive model evaluation focusing on loan default prediction metrics.
    """
    logger.info(f"Evaluating model with test data shape: {test_features.shape}")
    
    # Separate features and target - updated for new dataset
    target_col = 'Default'
    X_test = test_features.drop(target_col, axis=1)
    y_test = test_features[target_col]
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of default
    
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
    logger.info(f"Class 0 (No Default):")
    logger.info(f"  Precision: {class_report['0']['precision']:.4f}")
    logger.info(f"  Recall: {class_report['0']['recall']:.4f}")
    logger.info(f"  F1-score: {class_report['0']['f1-score']:.4f}")
    
    logger.info(f"Class 1 (Default):")
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
    actual_defaults = (y_test == 1).sum()
    actual_good_loans = (y_test == 0).sum()
    
    # How many defaults would we miss? (False Negatives)
    false_negatives = cm[1,0]
    # How many good loans would we reject? (False Positives)  
    false_positives = cm[0,1]
    
    logger.info(f"\nBusiness Impact Analysis:")
    logger.info(f"  Total loan applications: {total_loans}")
    logger.info(f"  Actually defaulting loans: {actual_defaults} ({actual_defaults/total_loans:.1%})")
    logger.info(f"  Defaults we'd miss (BAD): {false_negatives} ({false_negatives/actual_defaults*100:.1f}% of defaults)")
    logger.info(f"  Good loans we'd reject (LOST BUSINESS): {false_positives} ({false_positives/actual_good_loans*100:.1f}% of good loans)")
    
    # Calculate potential financial impact based on typical loan amounts from dataset
    avg_loan_amount = test_features['LoanAmount'].mean() if 'LoanAmount' in test_features.columns else 100000
    default_loss_rate = 0.8   # Assume 80% loss on defaulted loans
    
    potential_loss = false_negatives * avg_loan_amount * default_loss_rate
    lost_revenue = false_positives * avg_loan_amount * 0.12  # Assume 12% total profit over loan term
    
    logger.info(f"\nEstimated Financial Impact (for {total_loans} loans):")
    logger.info(f"  Average loan amount: ${avg_loan_amount:,.0f}")
    logger.info(f"  Potential loss from missed defaults: ${potential_loss:,.0f}")
    logger.info(f"  Lost revenue from rejected good loans: ${lost_revenue:,.0f}")
    logger.info(f"  Net impact: ${potential_loss - lost_revenue:,.0f}")
    
    # Model performance summary
    logger.info(f"\nModel Performance Summary:")
    logger.info(f"  Default Detection Rate (Recall): {class_report['1']['recall']:.1%}")
    logger.info(f"  False Alarm Rate: {false_positives/(false_positives + cm[0,0]):.1%}")
    logger.info(f"  Precision (of flagged defaults): {class_report['1']['precision']:.1%}")
    
    # Full classification report for detailed analysis
    logger.info("\nDetailed Classification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Prepare evaluation results to save
    evaluation_data = {
        'threshold': float(threshold),
        'roc_auc': float(roc_auc),
        'accuracy': float(class_report['accuracy']),
        'precision_no_default': float(class_report['0']['precision']),
        'recall_no_default': float(class_report['0']['recall']),
        'f1_no_default': float(class_report['0']['f1-score']),
        'precision_default': float(class_report['1']['precision']),
        'recall_default': float(class_report['1']['recall']),
        'f1_default': float(class_report['1']['f1-score']),
        'confusion_matrix': cm.tolist(),
        'business_metrics': {
            'total_loans': int(total_loans),
            'actual_defaults': int(actual_defaults),
            'false_negatives': int(false_negatives),
            'false_positives': int(false_positives),
            'default_detection_rate': float(class_report['1']['recall']),
            'false_alarm_rate': float(false_positives/(false_positives + cm[0,0])),
            'avg_loan_amount': float(avg_loan_amount),
            'potential_loss': float(potential_loss),
            'lost_revenue': float(lost_revenue),
            'net_impact': float(potential_loss - lost_revenue)
        }
    }
    
    return evaluation_data
