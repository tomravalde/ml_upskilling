import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

from retrieve_and_create_data import collection

# Load the dataset (comment out as necessary)
df = collection.get_dataframe("raw_with_target")
df = collection.get_dataframe("missing_data")
df = collection.get_dataframe("clean_rows_only")
df = collection.get_dataframe("imputed_mean")
df = collection.get_dataframe("imputed_median")
df = collection.get_dataframe("imputed_mode")
df = collection.get_dataframe("imputed_constant")
df = collection.get_dataframe("imputed_iterative")


###################################################################################################
# Visualize missing data
###################################################################################################
"""
https://github.com/ResidentMario/missingno?tab=readme-ov-file
"""

# 1. Missing Data Heatmap
"""
nullity correlation: how strongly the presence or absence of one variable affects the presence of 
another
Nullity correlation ranges from -1 (if one variable appears the other definitely does not) to 0 
(variables appearing or not appearing have no effect on one another) to 1 (if one variable appears 
the other definitely also does).
"""
plt.figure(figsize=(10, 6))
msno.heatmap(df)
plt.title('Missing Data Heatmap')
plt.show()

# 2. Missing Data Bar Plot
plt.figure(figsize=(10, 6))
msno.bar(df) # log=True may help
plt.title('Missing Data Bar Plot')
plt.show()

# 3. Matrix Plot
plt.figure(figsize=(10, 6))
msno.matrix(df)
plt.title('Missing Data Matrix Plot')
plt.show()


###################################################################################################
# Explore the data
###################################################################################################

# 1. Histogram of Wine Quality Ratings
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=df, palette='viridis')
plt.title('Histogram of Wine Quality Ratings')
plt.xlabel('Wine Quality')
plt.ylabel('Frequency')
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 3. Box Plots for Feature Distributions by Quality
plt.figure(figsize=(14, 10))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(4, 3, i)
    sns.boxplot(x='quality', y=column, data=df, palette='viridis')
    plt.title(f'Box Plot of {column} by Quality')
plt.tight_layout()
plt.show()

# 4. Pair Plot of Features
plt.figure(figsize=(14, 10))
sns.pairplot(df, hue='quality', palette='viridis', markers='o')
plt.title('Pair Plot of Features')
plt.show()

# 5. Distribution of Features
plt.figure(figsize=(14, 10))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(4, 3, i)
    sns.histplot(df[column], kde=True, color='skyblue')
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

# 6. Class imbalance
"""
Models will perform more poorly on the minority class. Address this through:
- Oversample the minority class by duplication
- Oversample the minority class using SMOTE: Synthetic Minority Over-sample Technique
- Undersample the majority class
- Choice ofr performance metric: Precision, Recall, F1, ROC and P-R AUC
- Use ensemble methods with class-weight adjustment, so each tree 
- Algorithmic modifications to upweidght the miniorty class, via sklearn's `class_weight` parameter
- Sampling techniques described below
"""
class_counts = df['target'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Class Balance')
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.xticks(ticks=[0, 1], labels=['Bad Quality', 'Good Quality'])
plt.show()