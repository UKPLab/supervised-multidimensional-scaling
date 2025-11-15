import numpy as np
import pytest
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist

from smds import SupervisedMDS
from smds.shapes.discrete_shapes.cluster import ClusterShape


@pytest.fixture
def random_data() -> tuple[np.ndarray, np.ndarray]:
    """Provides random X and y data for basic testing."""
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 3, size=50).astype(float)
    return X, y


def test_cluster_smoke_test(random_data: tuple[np.ndarray, np.ndarray]) -> None:
    """
    A simple "smoke test" to ensure ClusterShape can be fit and transformed
    without errors and produces an output of the correct shape.
    """
    X, y = random_data
    n_samples = X.shape[0]
    n_components = 2

    smds_engine = SupervisedMDS(n_components=n_components, manifold=ClusterShape())
    X_proj = smds_engine.fit_transform(X, y)

    assert X_proj.shape == (n_samples, n_components)


@pytest.fixture
def structured_cluster_data_2d() -> tuple[np.ndarray, np.ndarray]:
    """
    Provides a dataset where points are already clearly separated into two groups.
    """
    # Create 25 points centered around [-10, -10]
    cluster0 = np.random.randn(25, 2) - 10

    # Create 25 points centered around [10, 10]
    cluster1 = np.random.randn(25, 2) + 10

    X = np.vstack([cluster0, cluster1])
    y = np.array([0.0] * 25 + [1.0] * 25)

    # Shuffle the data
    indices = np.arange(50)
    np.random.shuffle(indices)

    return X[indices], y[indices]


def test_cluster_preserves_structure_in_2d(structured_data: tuple[np.ndarray, np.ndarray]) -> None:
    """
    Sanity Check (2D -> 2D): Tests the core behavior of ClusterShape: ensuring that points with the same
    label are closer to each other than to points with different labels.
    """
    X, y = structured_data
    smds_engine = SupervisedMDS(n_components=2, manifold=ClusterShape())

    X_proj = smds_engine.fit_transform(X, y)

    # Identify the indices for each cluster
    indices_c0 = np.where(y == 0.0)[0]
    indices_c1 = np.where(y == 1.0)[0]

    # Calculate the average within-cluster distance
    dist_within_c0 = pdist(X_proj[indices_c0]).mean()
    dist_within_c1 = pdist(X_proj[indices_c1]).mean()
    avg_within_cluster_dist = (dist_within_c0 + dist_within_c1) / 2

    # Calculate the distance between the centroids of the two clusters
    centroid_c0 = X_proj[indices_c0].mean(axis=0)
    centroid_c1 = X_proj[indices_c1].mean(axis=0)
    dist_between_clusters = np.linalg.norm(centroid_c0 - centroid_c1)

    #print(f"Avg within-cluster distance: {avg_within_cluster_dist:.4f}")
    #print(f"Distance between cluster centroids: {dist_between_clusters:.4f}")
    assert avg_within_cluster_dist < dist_between_clusters


#ToDo: test for int and string lables IF team decides to include those


@pytest.fixture
def structured_cluster_data_high_dim() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Provides a high-dimensional dataset containing a hidden 2D cluster structure.
    This tests the ability of SMDS to recover a latent manifold.
    """
    # Create the simple, 2D ground-truth data
    latent_dim = 2
    high_dim = 10
    n_samples_per_cluster = 25

    cluster1_latent = np.random.randn(n_samples_per_cluster, latent_dim) - 10
    cluster2_latent = np.random.randn(n_samples_per_cluster, latent_dim) + 10
    X_latent = np.vstack([cluster1_latent, cluster2_latent])
    y = np.array([0.0] * n_samples_per_cluster + [1.0] * n_samples_per_cluster)

    # Create a random projection matrix to map into higher dim
    projection_matrix = np.random.randn(latent_dim, high_dim)
    X_high_dim = X_latent @ projection_matrix

    # Shuffle to ensure the model isn't relying on data order
    indices = np.arange(X_high_dim.shape[0])
    np.random.shuffle(indices)

    return X_high_dim[indices], y[indices], X_latent[indices]


def test_cluster_recovers_structure_from_high_dim(
        structured_data_high_dim: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """
    Tests if SMDS can find and recover a 2D cluster structure
    hidden in a noisy, high-dimensional space.
    """
    X, y, X_original = structured_data_high_dim

    smds_engine = SupervisedMDS(n_components=2, manifold=ClusterShape())
    X_proj = smds_engine.fit_transform(X, y)

    indices_c0 = np.where(y == 0.0)[0]
    indices_c1 = np.where(y == 1.0)[0]

    # --- Sanity check ---
    # Checks if the recovered clusters are separated at all.
    avg_within_cluster_dist = (pdist(X_proj[indices_c0]).mean() + pdist(X_proj[indices_c1]).mean()) / 2
    dist_between_clusters = np.linalg.norm(X_proj[indices_c0].mean(axis=0) - X_proj[indices_c1].mean(axis=0))
    #print(f"Avg within-cluster distance: {avg_within_cluster_dist:.4f}")
    #print(f"Distance between cluster centroids: {dist_between_clusters:.4f}")
    assert avg_within_cluster_dist < dist_between_clusters, "Sanity check failed: Clusters are not separated."

    # --- Procrustes Analysis ---
    # Compares the recovered shape to the original latent shape, ignoring rotation/scale.
    mtx1, mtx2, disparity = procrustes(X_original, X_proj)
    #print(f"Procrustes disparity between original and recovered shape: {disparity:.4f}")
    assert disparity < 0.1, "Procrustes analysis shows the recovered shape is too different from the original."

    # --- Built-in Score ---
    # The score measures how well the projection satisfies the ClusterShape distance rules.
    score = smds_engine.score(X, y)
    #print(f"SMDS score for ClusterShape: {score:.4f}")
    assert score > 0.95, "SMDS score is too low, indicating a poor clustering result."
