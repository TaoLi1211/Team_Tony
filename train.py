import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# (Optional) If you need to display non-English characters in plots, set a proper font:
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # or ['Microsoft YaHei']
# matplotlib.rcParams['axes.unicode_minus'] = False

# 1. Read data
df = pd.read_csv('https://github.com/TaoLi1211/Team_Tony/blob/4ec5c3835939d10714d759ae04aa9d7a310e09de/predictive_maintenance.csv')
print("Data preview:")
print(df.head())
print("\nData info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# 2. Data preprocessing
# Assume the target column is named 'Target'. Adjust if needed.
target_col = 'Target'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in the dataset.")

# Create a list of feature columns (excluding the target column)
feature_cols = [col for col in df.columns if col != target_col]

# To avoid chained assignment warnings, make a copy of the DataFrame
df = df.copy()

# Fill missing values:
# - For numeric columns, fill with mean
# - For categorical/object columns, fill with mode
for col in feature_cols:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# One-hot encode categorical columns
categorical_cols = [col for col in feature_cols if df[col].dtype == 'object']
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 3. Split data into training and test sets
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify=y helps maintain class distribution
)

# 4. Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("\nClassification report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# 6. Plot the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
