import requests
import numpy as np
import pandas as pd

# URL of the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Save the content of the response to a file
    with open('winequality-red.csv', 'wb') as file:
        file.write(response.content)
    print("Dataset downloaded successfully.")
else:
    print("Failed to download the dataset.")

df = pd.read_csv('winequality-red.csv', sep=';')


# Convert quality to binary classification (good: quality >= 7, bad: quality < 7)
df['target'] = (df['quality'] >= 7).astype(int)
df = df.drop('quality', axis=1)
df.to_csv('winequality-red.csv', index=False)


# Introduce missing data
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

# Save for use elsewhere
df_missing.to_csv('winequality-red-missing-data.csv', index=False)

# TODO: Create imbalanced dataset