from clearml import Task
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

task = Task.init(project_name="Predictive Maintenance", task_name="Model Training")

def train_model():
    df = pd.read_csv('./clean_data.csv')
    X = df.drop('Target', axis=1)
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    training_columns = X_train.columns.tolist()
    joblib.dump(training_columns, './training_columns.pkl')

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }

    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    joblib.dump(grid_search.best_estimator_, './best_model.pkl')
    pd.DataFrame(X_test).to_csv('./X_test.csv', index=False)
    pd.DataFrame(y_test).to_csv('./y_test.csv', index=False)

    task.upload_artifact('best_model', './best_model.pkl')

if __name__ == "__main__":
    train_model()
