import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    target_col = 'Target'
    feature_cols = [col for col in df.columns if col not in [target_col, 'UDI', 'Product ID']]

    # Handle missing values
    for col in feature_cols:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Save columns explicitly for later prediction
    joblib.dump(X.columns.tolist(), 'model_columns.joblib')

    return X, y

def train_and_save_model(X, y):
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    joblib.dump(clf, 'model.joblib')
    return clf

def predict_from_input(model, input_data):
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)
    return prediction, prediction_prob

# Train and save the model & columns (run once)
if __name__ == '__main__':
    X, y = load_and_prepare_data('predictive_maintenance.csv')
    train_and_save_model(X, y)
