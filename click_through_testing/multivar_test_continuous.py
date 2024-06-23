import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Simulate a dataset for multivariate testing
np.random.seed(42)
data_size = 200

# Independent variables
X1 = np.random.normal(50, 10, data_size)  # Example feature 1
X2 = np.random.normal(30, 5, data_size)   # Example feature 2
X3 = np.random.normal(20, 2, data_size)   # Example feature 3

# Dependent variable with some noise
y = 5 + 1.5 * X1 + 0.5 * X2 - 2 * X3 + np.random.normal(0, 10, data_size)

# Create a DataFrame
df = pd.DataFrame({'Feature1': X1, 'Feature2': X2, 'Feature3': X3, 'Outcome': y})

# Display the first few rows of the dataset
print(df.head())

# Pairplot to visualize relationships between variables
sns.pairplot(df)
plt.show()

# Add a constant to the independent variables matrix
X = df[['Feature1', 'Feature2', 'Feature3']]
X = sm.add_constant(X)

# Fit the multivariate regression model
model = sm.OLS(df['Outcome'], X).fit()

# Print the model summary
print(model.summary())

# Visualize the results: true vs predicted values
df['Predicted'] = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(df['Outcome'], df['Predicted'])
plt.plot([df['Outcome'].min(), df['Outcome'].max()], [df['Outcome'].min(), df['Outcome'].max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.show()
