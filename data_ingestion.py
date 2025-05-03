from clearml import Task
import pandas as pd

task = Task.init(project_name="Predictive Maintenance", task_name="Data Ingestion")

def ingest_data():
    df = pd.read_csv('predictive_maintenance.csv')
    df.to_csv('./raw_data.csv', index=False)

if __name__ == "__main__":
    ingest_data()