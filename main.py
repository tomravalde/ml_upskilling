import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Data Preprocessing
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to train and evaluate model
def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    print('Classification Report:')
    print(classification_report(y_test, y_pred))

# Logistic Regression
print("Logistic Regression")
lr_model = LogisticRegression()
train_evaluate_model(lr_model, X_train, y_train, X_test, y_test)

# Random Forest
print("\nRandom Forest")
rf_model = RandomForestClassifier()
train_evaluate_model(rf_model, X_train, y_train, X_test, y_test)

# Support Vector Machine
print("\nSupport Vector Machine")
svm_model = SVC()
train_evaluate_model(svm_model, X_train, y_train, X_test, y_test)

# Neural Network
print("\nNeural Network")
mlp_model = MLPClassifier(max_iter=1000)
train_evaluate_model(mlp_model, X_train, y_train, X_test, y_test)

# Hyperparameter Tuning (example for Random Forest)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\nBest Parameters for Random Forest:")
print(grid_search.best_params_)

# Plotting feature importance for Random Forest
best_rf_model = grid_search.best_estimator_
feature_importances = best_rf_model.feature_importances_
features = df.columns[:-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importances')
plt.show()
