import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Perform PCA to reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Function to plot clusters
def plot_clusters(X, y, cluster_labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='x', s=100, label='True Labels')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.legend()
    plt.show()

# Step 1: Perform clustering without standardization
kmeans_no_std = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans_no_std.fit(X)
plot_clusters(X_pca, y, kmeans_no_std.labels_, 'Clustering without Z-score Standardization')

# Step 2: Perform Z-score standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Perform clustering after standardization
kmeans_std = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans_std.fit(X_scaled)
plot_clusters(X_pca, y, kmeans_std.labels_, 'Clustering after Z-score Standardization')
