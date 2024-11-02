import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset (replace 'transactions.csv' with the actual file path)
df = pd.read_csv('creditcard.csv')

# Separate features and labels
X = df.drop(columns=['Class'])
y = df['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=3, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# Reduce dimensions for visualization using PCA (2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', marker='o', s=10)
plt.title('DBSCAN Clustering of Transactions')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Plot the frauds (Class = 1) and non-frauds (Class = 0) for comparison
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label='Non-Fraud', c='blue', s=10, alpha=0.5)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label='Fraud', c='red', s=30, marker='x')
plt.title('Fraud vs Non-Fraud Transactions in PCA Space')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()
