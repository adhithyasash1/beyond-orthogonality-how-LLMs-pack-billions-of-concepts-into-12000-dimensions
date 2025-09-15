import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

# ==============================================
# STEP 1: Generate synthetic high-dimensional data
# ==============================================
# We create 300 points in 50 dimensions, grouped into 3 clusters.
# This mimics a typical high-dimensional dataset where JL projections are useful.

n_samples = 300        # number of data points
n_features = 50        # original dimension (high-dimensional space)
n_clusters = 3         # number of clusters to generate

# Create synthetic data (X = points, y = cluster labels)
X, y = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)

# Compute pairwise Euclidean distances in the original space
original_distances = pairwise_distances(X)

# ==============================================
# STEP 2: Define Achlioptas Random Projection
# ==============================================
# This function creates a projection matrix with entries from either:
# - Binary distribution (±1 with probability 0.5 each)
# - Sparse distribution (+sqrt(3), 0, -sqrt(3) with probabilities 1/6, 2/3, 1/6)

def achlioptas_random_projection(X, k, method="binary"):
    """
    Perform Achlioptas random projection.
    X: input matrix of shape (n, d)
    k: reduced dimension
    method: 'binary' or 'sparse' projection matrix
    Returns: projected data of shape (n, k)
    """
    n, d = X.shape
    if method == "binary":
        # Rademacher distribution: ±1 with equal probability
        R = np.random.choice([-1, 1], size=(d, k))
    elif method == "sparse":
        # Sparse Achlioptas distribution: saves computation
        R = np.random.choice([np.sqrt(3), 0, -np.sqrt(3)], size=(d, k), p=[1/6, 2/3, 1/6])
    else:
        raise ValueError("Unknown projection method")

    # Project data and normalize by sqrt(k) as per JL lemma
    return (1 / np.sqrt(k)) * X.dot(R)

# ==============================================
# STEP 3: Apply JL projection for various target dimensions
# ==============================================
reduced_dims = [5, 10, 20, 30]  # target dimensions we want to test
distortion_means = []           # to store mean distortion values
distortion_stds = []            # to store standard deviations

for k in reduced_dims:
    # Project data into k dimensions
    X_proj = achlioptas_random_projection(X, k, method="binary")

    # Compute distances in the reduced space
    projected_distances = pairwise_distances(X_proj)

    # Compute ratio of projected distances to original distances
    # We only consider upper triangular matrix (i < j) to avoid duplicates
    ratio = (projected_distances / original_distances)[np.triu_indices_from(original_distances, k=1)]
    ratio = ratio[np.isfinite(ratio)]  # remove infinities (caused by 0-distance pairs)

    # Store mean and standard deviation of distortion
    distortion_means.append(np.mean(ratio))
    distortion_stds.append(np.std(ratio))

# ==============================================
# STEP 4: Visualization
# ==============================================
# We plot how the mean distance ratio changes with reduced dimension.
# Ideally, the mean ratio should be close to 1 (perfect distance preservation).

plt.figure(figsize=(10, 5))
plt.errorbar(reduced_dims, distortion_means, yerr=distortion_stds, fmt='-o', capsize=5)
plt.axhline(1, color='red', linestyle='--', label="Perfect distance preservation")
plt.title("Achlioptas JL Projection: Distance Preservation vs Reduced Dimension")
plt.xlabel("Reduced Dimension (k)")
plt.ylabel("Mean Distance Ratio (Projected / Original)")
plt.legend()
plt.grid()
plt.show()
