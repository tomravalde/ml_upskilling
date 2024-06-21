import requests

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