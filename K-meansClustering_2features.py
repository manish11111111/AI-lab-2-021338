import numpy as np
import matplotlib.pyplot as plt

# Dataset with 12 samples and 4 features
data = np.array([
    [1.0, 1.5, 0.8, 1.2],
    [1.2, 1.7, 0.9, 1.1],
    [1.1, 1.4, 0.7, 1.3],
    [1.3, 1.6, 0.85, 1.15],
    [4.0, 4.5, 3.8, 4.2],
    [4.2, 4.7, 3.9, 4.1],
    [4.1, 4.4, 3.7, 4.3],
    [4.3, 4.6, 3.85, 4.15],
    [2.0, 2.2, 1.8, 2.1],
    [2.1, 2.3, 1.9, 2.0],
    [4.5, 4.8, 4.0, 4.4],
    [4.4, 4.9, 3.95, 4.35]
])

k = 3
max_iters = 100
np.random.seed(42)

# Initialize centroids randomly from data points
centroids = data[np.random.choice(len(data), k, replace=False)]

for iteration in range(max_iters):
    # Compute distances from points to centroids
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)  # shape (n_samples, k)
    cluster_assignments = np.argmin(distances, axis=1)  # Assign clusters based on closest centroid

    # Compute new centroids
    new_centroids = np.array([
        data[cluster_assignments == j].mean(axis=0) if np.any(cluster_assignments == j) else centroids[j]
        for j in range(k)
    ])

    # Check convergence
    if np.allclose(centroids, new_centroids):
        print(f"Converged at iteration {iteration + 1}")
        break

    centroids = new_centroids

# Final results
print("Final cluster assignments:")
for idx, cluster in enumerate(cluster_assignments):
    print(f"Sample {idx}: Cluster {cluster}")

print("\nFinal centroid positions:")
print(centroids)

# Compute WCSS
wcss = 0
for j in range(k):
    points = data[cluster_assignments == j]
    wcss += np.sum((points - centroids[j]) ** 2)
print(f"\nWithin-Cluster Sum of Squares (WCSS): {wcss:.4f}")

# Visualization (project onto first two features)
colors = ['red', 'green', 'blue']
for j in range(k):
    cluster_points = data[cluster_assignments == j]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[j], label=f'Cluster {j}')

plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('k-Means Clustering (k=3) - Projected on first two features')
plt.legend()
plt.grid(True)
plt.show()
