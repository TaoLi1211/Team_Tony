import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from clearml import Task
import joblib

# Initialize ClearML task
task = Task.init(
    project_name='Predictive Maintenance',
    task_name='RandomForest GridSearch Training'
)

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    df = df.drop(columns=['UDI', 'Product ID'], errors='ignore')
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    return df

def train_model(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, scoring='f1', verbose=2, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    return grid_search

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    # Log metrics clearly to ClearML
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    
    task.get_logger().report_text("Classification Report:\n" + report)
    task.get_logger().report_confusion_matrix(
        title="Confusion Matrix",
        matrix=cm,
        labels=["Healthy", "Needs Maintenance"]
    )

def save_model(model, path='model.pkl'):
    joblib.dump(model, path)
    task.upload_artifact('trained_model', path)

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data('predictive_maintenance.csv')
    df = preprocess_data(df)

    X = df.drop('Target', axis=1)
    y = df['Target']

    # Split the data clearly into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model with GridSearch
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the trained model
    save_model(model.best_estimator_)

