import pandas as pd
from sklearn.preprocessing import StandardScaler

def feature_engineering(train: pd.DataFrame, test: pd.DataFrame):
    # Simple feature: income per year of job
    for df in [train, test]:
        if 'Income' in df.columns and 'CURRENT_JOB_YRS' in df.columns:
            df['income_per_year_of_job'] = df['Income'] / (df['CURRENT_JOB_YRS'] + 1)
        # Age bucket
        if 'Age' in df.columns:
            df['age_bucket'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['young', 'middle', 'senior'])
    # One-hot encode age_bucket
    train = pd.get_dummies(train, columns=['age_bucket'], drop_first=True) if 'age_bucket' in train.columns else train
    test = pd.get_dummies(test, columns=['age_bucket'], drop_first=True) if 'age_bucket' in test.columns else test
    # Align columns
    train, test = train.align(test, join='left', axis=1, fill_value=0)
    return train, test
