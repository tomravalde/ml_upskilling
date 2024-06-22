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
from imblearn.over_sampling import SMOTE


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
    """
    f1-score: harmonic mean of precision and recall
    f_beta-score: beta weights the relative importance of precision and recall
    support: number of samples in each class of the test set
    macro avg: of each class's performance metric, calculate arithmetic mean, i.e. equally 
        weighting each class
    weighted avg: of each class's performance metric, calculated mean weighted be each class size; 
        this will favour the majority class, so beware in imbalanced class
    """

    # Plot ROC Curve and Precision-Recall Curve
    if hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = model.predict_proba(X_test)[:, 1]

    # ROC Curve
    """
    Change the value between 0-1 of the boundary that separates a class and plot FP rate vs TP rate
    """
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
    """
    Change the value between 0-1 of the boundary that separates a class and plot Recall vs 
    Precision. More useful for imbalanced sets where positive class is rare (and therefore easy to 
    get a very recall but poor precision)
    """
    precision, recall, _ = precision_recall_curve(y_test, y_scores, pos_label=1)
    average_precision = average_precision_score(y_test, y_scores)

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve: AP={average_precision:0.2f} - {model_name}')
    plt.show()

    # Distribution of predicted probabilities
    # Combine actual and predicted into a DataFrame for easier plotting
    results = pd.DataFrame({'Actual': y_test, 'Predicted Probability': y_scores})

    # Plot the distribution of predicted probabilities for each class
    plt.figure(figsize=(12, 6))
    sns.histplot(data=results, x='Predicted Probability', hue='Actual', element='step',
                 stat='density', common_norm=False, bins=30, palette='viridis')
    plt.title('Distribution of Predicted Probabilities vs Actual Classes')
    plt.xlabel('Predicted Probability of Good Quality Wine')
    plt.ylabel('Density')
    plt.legend(title='Actual Class', labels=['Bad Quality', 'Good Quality'])
    plt.show()


# Loop over different input datasets
df_full = pd.read_csv('winequality-red.csv', sep=',')
df_missing = pd.read_csv('winequality-red-missing-data.csv', sep=',')
df_imputed = pd.read_csv('winequality-red-imputed.csv', sep=',')

# for df in [df_full, df_missing]:
for df in [df_missing]:
    # Data Preprocessing
    X = df.drop('target', axis=1)
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply SMOTE to ovversample the minorty class
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # # Logistic Regression
    # print("Logistic Regression")
    # lr_model = LogisticRegression()
    # train_evaluate_model(lr_model, X_train, y_train, X_test, y_test, "Logistic Regression")

    # Random Forest
    print("\nRandom Forest")
    rf_model = RandomForestClassifier()
    train_evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")

    # Random Forest with SMOTE
    print("\nRandom Forest with SMOTE")
    train_evaluate_model(rf_model, X_train_res, y_train_res, X_test, y_test,
                         "Random Forest with SMOTE")

    # Random Forest with class weights
    """
    Adjust the Gini to scale the contribution of each class to the impurity calculation. This will 
    modify the splitting criteria.
    
    Balanced: weights are inversely proportional to class frequencies, so minority classes are 
        up-weighted
        
    (Can also specify custom weights)
    """

    print("\nRandom Forest with class weights")
    rf_weighted = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                         random_state=42)
    train_evaluate_model(rf_weighted, X_train, y_train, X_test, y_test,
                         "Random Forest with class weights")

    # # Support Vector Machine
    # print("\nSupport Vector Machine")
    # svm_model = SVC(probability=True)
    # train_evaluate_model(svm_model, X_train, y_train, X_test, y_test, "Support Vector Machine")
    #
    # # Neural Network
    # print("\nNeural Network")
    # mlp_model = MLPClassifier(max_iter=1000)
    # train_evaluate_model(mlp_model, X_train, y_train, X_test, y_test, "Neural Network")


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
"""
Each split aims to make the data within each group as homogenous as possible according to some 
criteria. If a feature used for a split does this successfully, it is important
Quantify how much each split improves a model's predictions. We add these scores at feature-level 
(and normalise) to get a score
- classification: Gini impurity (how 'mixed' a group is)
- Regression: MSE
"""

best_rf_model = grid_search.best_estimator_
feature_importances = best_rf_model.feature_importances_
features = df.columns[:-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
# TODO: Order by importance
plt.title('Feature Importances')
plt.show()

