import requests
import numpy as np
import pandas as pd
from utils import DataFrameCollection

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Initialise collection of DataFrames
collection = DataFrameCollection()

###################################################################################################
# retrieve data
###################################################################################################

# URL of the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Save the content of the response to a file
    with open('data/winequality-red.csv', 'wb') as file:
        file.write(response.content)
    print("Dataset downloaded successfully.")
else:
    print("Failed to download the dataset.")

# Store
df = pd.read_csv('data/winequality-red.csv', sep=';')
collection.add_dataframe("raw", df)


###################################################################################################
# Convert quality to binary classification (good: quality >= 7, bad: quality < 7)
###################################################################################################

df['target'] = (df['quality'] >= 7).astype(int)

# Store
df.to_csv('data/winequality-red.csv', index=False)
collection.add_dataframe("raw_with_target", df)


###################################################################################################
# Introduce missing data
###################################################################################################

random_state = 42
missing_fraction = 0.05
np.random.seed(random_state)  # For reproducibility
df_missing = df.copy()

# Calculate the total number of elements in the DataFrame
total_elements = df_missing.size

# Calculate the number of elements to set as NaN
n_missing = int(np.floor(missing_fraction * total_elements))

# Randomly select indices to set as NaN
for _ in range(n_missing):
    ix = np.random.choice(df_missing.index)
    col = np.random.choice(df_missing.columns)
    df_missing.loc[ix, col] = np.nan

# Display the first few rows of the dataset with missing values
print(df_missing.head())

# Store
collection.add_dataframe("missing_data", df)

###################################################################################################
# Handle missing data
###################################################################################################

# 1. Remove rows with missing values
df_clean_rows = df_missing.dropna() # Rows reduce from 1599 to 890
collection.add_dataframe("clean_rows_only", df_clean_rows)

## VARIATION: Drop rows with missing values in specific columns
# df_clean_specific = df.dropna(subset=['column1', 'column2'])

# 2. Remove columns with missing values
df_clean_cols = df_missing.dropna(axis=1) # No cols left :(
collection.add_dataframe("clean_cols_only", df_clean_cols)

##: VARIATION: Fill missing values in a specific column with mean
# df['column_name'] = df['column_name'].fillna(df['column_name'].mean())

# 3. Impute using mean
df_filled_mean = df.fillna(df.mean())
collection.add_dataframe("imputed_mean", df_filled_mean)

# 4. Impute using median
df_filled_median = df.fillna(df.median())
collection.add_dataframe("imputed_median", df_filled_median)

# 5. Impute using mode (for categorical data)
df_filled_mode = df.fillna(df.mode().iloc[0])
collection.add_dataframe("imputed_mode", df_filled_mode)

# 6. Impute with a constant value
df_filled_constant = df.fillna(-999)  # Replace -999 with your chosen constant
collection.add_dataframe("imputed_constant", df_filled_constant)

# 7. Iterative imputer
"""
each feature's missing values = f(other features)
1. Initialise imputation of missing values using simple method, e.g. mean
2. Select a feature, f
    - Training set = rows where f is not missing
    - Regression model to predict missing f values from values of remaining features, continuous 
    or classification, according to the feature)
    - Replace missing values with predicted values
3. Repeat Step 2 for remaining features
4. Continue iterations until imputed values stabilise (delta falls below specified threshold) 
    or predefined number iterations
    
Advantages:
- *Accurate*
- *Flexible*, handling both contiuous and categorical features
- *Consistency* of resulting values with overall data distribution and inter-feature relations
"""

# Initialize IterativeImputer
imputer = IterativeImputer(random_state=0) # Initialise

df_imputed = pd.DataFrame(imputer.fit_transform(df)) # Fit and transform
df_imputed.columns = df.columns
df_imputed.index = df.index

collection.add_dataframe("imputed_iterative", df_imputed)

# 8. CATEGORICAL DATA: impute using the mode
# Fill missing values in categorical columns with the mode
# categorical_columns = ['cat_column1', 'cat_column2']
# for col in categorical_columns:
#     df[col] = df[col].fillna(df[col].mode().iloc[0])

# 9. Custom strategies
def custom_imputer(df):
    """
    e.g. logic based on domain knowledge

    Usage:
        > df_custom_imputed = custom_imputer(df)
    """
    # Your custom logic here
    return df