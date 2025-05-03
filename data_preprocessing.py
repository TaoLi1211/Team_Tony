from clearml import Task
import pandas as pd

task = Task.init(project_name="Predictive Maintenance", task_name="Data Preprocessing")

def preprocess_data():
    df = pd.read_csv('./raw_data.csv')

    # Remove unnecessary columns
    df = df.drop(columns=['UDI', 'Product ID'], errors='ignore')

    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # One-hot encoding categorical variables
    df = pd.get_dummies(df, drop_first=True)

    df.to_csv('./clean_data.csv', index=False)

if __name__ == "__main__":
    preprocess_data()
