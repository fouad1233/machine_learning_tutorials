import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the dataset
num_clusters = 4
points_per_cluster = 100
cluster_std_deviation = 1.0  # Spread of each cluster

# Define centers of clusters
cluster_centers = [
    (5, 5),
    (15, 5),
    (5, 15),
    (15, 15)
]

# Generate random data points for each cluster
data = []
for cluster_id, center in enumerate(cluster_centers):
    # Generate points around each center
    cluster_data = np.random.randn(points_per_cluster, 2) * cluster_std_deviation + center
    # Add cluster ID and store in data
    cluster_data = np.hstack((cluster_data, np.full((points_per_cluster, 1), cluster_id)))
    data.append(cluster_data)

# Combine all clusters into one dataset
data = np.vstack(data)

# Convert to a DataFrame
df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2', 'Cluster ID'])

# Write the data to an Excel file
df.to_excel("data/test_data.xlsx", index=False)

# Plot the dataset
plt.figure(figsize=(8, 8))
plt.scatter(data[:, 0], data[:, 1], s=30, c='b', marker='o', edgecolor='k')
plt.title("Synthetic Dataset for K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
