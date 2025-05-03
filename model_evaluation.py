from clearml import Task
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib

task = Task.init(project_name="Predictive Maintenance", task_name="Model Evaluation")

def evaluate_model():
    model = joblib.load('./best_model.pkl')
    X_test = pd.read_csv('./X_test.csv')
    y_test = pd.read_csv('./y_test.csv')

    predictions = model.predict(X_test)

    report = classification_report(y_test, predictions, output_dict=False)
    cm = confusion_matrix(y_test, predictions)

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    # Log metrics to ClearML
    task.get_logger().report_text("Classification Report:\n" + report)
    
    # Corrected ClearML confusion matrix reporting
    task.get_logger().report_confusion_matrix(
    title="Confusion Matrix",
    series="Model Evaluation",
    matrix=cm,
    xaxis='Predicted Label',
    yaxis='True Label',
    xlabels=["Healthy", "Needs Maintenance"],
    ylabels=["Healthy", "Needs Maintenance"]
)


if __name__ == "__main__":
    evaluate_model()
