"""
embedding geometry demo
-----------------------

this script illustrates three core ideas from nicholas yoder's article
"beyond orthogonality: how language models pack billions of concepts into 12,000 dimensions":

1. quasi-orthogonality in high dimensions
2. the johnson–lindenstrauss lemma
3. embedding capacity estimates for language model embeddings

requirements:
    numpy
    matplotlib
    scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection


# -----------------------------------------------------------
# helper functions
# -----------------------------------------------------------

def random_unit_vectors(n_vectors: int, dim: int) -> np.ndarray:
    """
    generate n_vectors random unit vectors in 'dim'-dimensional space
    """
    x = np.random.randn(n_vectors, dim)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def pairwise_angles(vectors: np.ndarray, sample_size: int = 5000) -> np.ndarray:
    """
    compute pairwise angles (in degrees) between random pairs of vectors
    """
    n = vectors.shape[0]
    idx = np.random.choice(n, size=(sample_size, 2), replace=False)
    v1 = vectors[idx[:, 0]]
    v2 = vectors[idx[:, 1]]
    dots = np.sum(v1 * v2, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    return angles


def jl_projection_distortion(n_points: int, orig_dim: int, proj_dim: int) -> np.ndarray:
    """
    demonstrate johnson–lindenstrauss lemma:
    project random high-dimensional points to lower dimensions
    and compute distance distortions
    """
    x = np.random.randn(n_points, orig_dim)

    # original pairwise distances
    idx = np.random.choice(n_points, size=(2000, 2), replace=False)
    dists_orig = np.linalg.norm(x[idx[:, 0]] - x[idx[:, 1]], axis=1)

    # project down
    projector = GaussianRandomProjection(n_components=proj_dim)
    x_proj = projector.fit_transform(x)
    dists_proj = np.linalg.norm(x_proj[idx[:, 0]] - x_proj[idx[:, 1]], axis=1)

    ratio = dists_proj / dists_orig
    return ratio


def embedding_capacity(k: int, f: int) -> float:
    """
    compute capacity estimate from article:
    vectors ≈ 10^(k * f^2 / 1500)
    """
    return 10 ** (k * (f ** 2) / 1500)


# -----------------------------------------------------------
# main demo
# -----------------------------------------------------------

def main():
    print("\n--- quasi-orthogonality demo ---")
    dim = 100
    n_vectors = 1000
    vectors = random_unit_vectors(n_vectors, dim)
    angles = pairwise_angles(vectors)

    print(f"in {dim} dimensions, {n_vectors} random vectors have:")
    print(f"  mean angle: {np.mean(angles):.2f}°")
    print(f"  std dev of angles: {np.std(angles):.2f}°")

    plt.figure(figsize=(6, 4))
    plt.hist(angles, bins=40, color="steelblue", alpha=0.8)
    plt.axvline(90, color="red", linestyle="--", label="perfect orthogonality")
    plt.title("distribution of angles between random vectors (100d)")
    plt.xlabel("angle (degrees)")
    plt.ylabel("frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- johnson–lindenstrauss lemma demo ---")
    n_points = 500
    orig_dim = 1000
    proj_dim = 50
    ratios = jl_projection_distortion(n_points, orig_dim, proj_dim)

    print(f"projected {n_points} points from {orig_dim}d → {proj_dim}d")
    print(f"  mean distortion ratio: {np.mean(ratios):.3f}")
    print(f"  min ratio: {np.min(ratios):.3f}, max ratio: {np.max(ratios):.3f}")

    plt.figure(figsize=(6, 4))
    plt.hist(ratios, bins=40, color="darkgreen", alpha=0.8)
    plt.axvline(1.0, color="red", linestyle="--", label="perfect preservation")
    plt.title("distance preservation under jl projection")
    plt.xlabel("projected/original distance ratio")
    plt.ylabel("frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- embedding capacity estimates ---")
    k = 12288  # gpt-3 embedding dimension
    for angle in [89, 88, 87, 85]:
        f = 90 - angle
        cap = embedding_capacity(k, f)
        print(f"at {angle}° (f={f}): capacity ≈ 10^{np.log10(cap):.1f}")

    print("\ndone.")


if __name__ == "__main__":
    main()
