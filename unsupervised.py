import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Mall Customer Segmentation Data
df = pd.read_csv('data/Mall_Customers.csv')
"""
Originally from https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python?phase=FinishSSORegistration&returnUrl=/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python/versions/1?resource=download&SSORegistrationToken=CfDJ8CHCUm6ypKVLpjizcZHPE72QCs41IsO_tML1-M6PQqRNM_loD7bOxBQDLFPB-geHPT53bYLjJzAixz1KVVy8hbxM2mI7DH8WxezbWWoSTGgfeQwz6JVj8FSjpVzGZ0vsakm4x9FnJoOve_f37_cEVlJ2qrEsqBEN3GQGuc1i2T2cDkfngVGgHMhnleTAqolxf9dqfrvoWnrLPBvAAu8IQ2jMTlVHzgl2fda3P_m0P20AC-WNnY0z0hhSFoPeVoWNUugBkczQ4X8qhUMldfuEY2IBo984DDExdK1iWEJaNLQAe-bjAwpjwtCqcehGL5CRMNvka8UEj-qED-RM59ROj9jwLA&DisplayName=Tom%20Ravalde
"""

# Display the first few rows of the dataset
print(df.head())

# Basic data information
print(df.info())

# Drop the CustomerID column as it's not useful for clustering
df = df.drop(columns=['CustomerID'])

# Encode the Gender column
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Standardize the data
"""
So that distances are meaningful
"""
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_df)

# Plot the clusters
sns.pairplot(df, hue='Cluster', palette='tab10')
plt.suptitle('Customer Segmentation Clusters', y=1.02)
plt.show()

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_df)

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = df['Cluster']

# Scatter plot of the PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='tab10', data=pca_df)
plt.title('PCA of Customer Segmentation')
plt.show()

# Cluster centroids
centroids = kmeans.cluster_centers_
centroids_pca = pca.transform(centroids)

# Plotting the centroids
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='tab10', data=pca_df)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='red', label='Centroids', marker='X')
plt.title('PCA of Customer Segmentation with Centroids')
plt.legend()
plt.show()

# Summary statistics for each cluster
print(df.groupby('Cluster').mean())

###################################################################################################
# Determine the optimal number of clusters using the Elbow Method
wss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    wss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WSS)')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Apply K-Means Clustering with the optimal number of clusters
optimal_clusters = 6  # This should be chosen based on the Elbow Method plot
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_df)

# Plot the clusters
sns.pairplot(df, hue='Cluster', palette='tab10')
plt.suptitle('Customer Segmentation Clusters', y=1.02)
plt.show()

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_df)

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = df['Cluster']

# Scatter plot of the PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='tab10', data=pca_df)
plt.title('PCA of Customer Segmentation')
plt.show()

# Cluster centroids
centroids = kmeans.cluster_centers_
centroids_pca = pca.transform(centroids)

# Plotting the centroids
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='tab10', data=pca_df)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='red', label='Centroids', marker='X')
plt.title('PCA of Customer Segmentation with Centroids')
plt.legend()
plt.show()

# Summary statistics for each cluster
print(df.groupby('Cluster').mean())


