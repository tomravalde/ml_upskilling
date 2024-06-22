import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_curve,
                             auc, precision_recall_curve, average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# Load Wine Quality dataset
df = pd.read_csv('winequality-red.csv', sep=';')

# Convert quality to binary classification (good: quality >= 7, bad: quality < 7)
df['target'] = (df['quality'] >= 7).astype(int)
df = df.drop('quality', axis=1)

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
def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # Plot ROC Curve and Precision-Recall Curve
    if hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = model.predict_proba(X_test)[:, 1]

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores, pos_label=1)
    average_precision = average_precision_score(y_test, y_scores)

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve: AP={average_precision:0.2f} - {model_name}')
    plt.show()


# Logistic Regression
print("Logistic Regression")
lr_model = LogisticRegression()
train_evaluate_model(lr_model, X_train, y_train, X_test, y_test, "Logistic Regression")

# Random Forest
print("\nRandom Forest")
rf_model = RandomForestClassifier()
train_evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")

# Support Vector Machine
print("\nSupport Vector Machine")
svm_model = SVC(probability=True)
train_evaluate_model(svm_model, X_train, y_train, X_test, y_test, "Support Vector Machine")

# Neural Network
print("\nNeural Network")
mlp_model = MLPClassifier(max_iter=1000)
train_evaluate_model(mlp_model, X_train, y_train, X_test, y_test, "Neural Network")

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