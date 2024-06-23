import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Sample data: CTR for different variations of the website and other predictors
data = {
    'CTR': [0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.07, 0.08],
    'size': ['small', 'large', 'small', 'large', 'small', 'large', 'small', 'large'],
    'colour': ['red', 'green', 'blue', 'red', 'green', 'blue', 'red', 'green']
}

# Create a DataFrame from the data
df_unencoded = pd.DataFrame(data)

# One-hot encode the categorical variables
df = pd.get_dummies(df_unencoded, columns=['size', 'colour'], drop_first=True)

# Define predictors and target variable
X = df[['size_small', 'colour_green', 'colour_red']]  # One-hot encoded features
y = df['CTR']

# Add constant term to the predictors
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(y, X.astype(float)).fit() # Requires all features to be zeros and ones

# Print summary of regression analysis
print(model.summary())

# Fit the ANOVA model (doesn't require one-hot encoding)
formula = 'CTR ~ C(size) * C(colour)'
"""
C(<feature>) indicates categorical variable
* includes main effects and interaction term
"""

anova_model = ols(formula, data=df_unencoded).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)

# Print ANOVA table
print("\nANOVA Table:")
print(anova_table)
"""
F-statistic compares between-group variance / within-group variance, thus larger value indicates 
that more variation in y is explained by the factor (or interaction) versus unexplained variation 
(residual)

p-value therefore gives the probibility of obtaining an F-statistic at least as large as the one 
calculated were the null hypothesis true, i.e. the variance is not explained by the factor (or 
interaction)
"""
