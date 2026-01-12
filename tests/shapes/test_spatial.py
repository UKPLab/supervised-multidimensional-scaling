import numpy as np
import pytest

from smds import ComputedStage1, SupervisedMDS
from smds.shapes.spatial_shapes import CylindricalShape, GeodesicShape, SphericalShape


@pytest.fixture
def engine_cylindrical() -> SupervisedMDS:
    return SupervisedMDS(ComputedStage1(n_components=2, manifold=CylindricalShape(radius=1.0)), alpha=0.1)


@pytest.fixture
def engine_geodesic() -> SupervisedMDS:
    return SupervisedMDS(ComputedStage1(n_components=2, manifold=GeodesicShape(radius=1.0)), alpha=0.1)


@pytest.fixture
def engine_spherical() -> SupervisedMDS:
    return SupervisedMDS(ComputedStage1(n_components=2, manifold=SphericalShape(radius=1.0)), alpha=0.1)


@pytest.fixture
def X() -> np.typing.NDArray[np.float64]:
    return np.random.randn(10, 10)


@pytest.fixture
def city_coordinates() -> np.typing.NDArray[np.float64]:
    # List of (Latitude, Longitude) covering 6 continents
    # Tokyo, Mumbai, Cairo, Lagos, London, Helsinki, NYC, Mexico City, São Paulo, Sydney
    return np.array(
        [
            (35.6895, 139.6917),  # Tokyo
            (19.0760, 72.8777),  # Mumbai
            (30.0444, 31.2357),  # Cairo
            (6.5244, 3.3792),  # Lagos
            (51.5074, -0.1278),  # London
            (60.1699, 24.9384),  # Helsinki
            (40.7128, -74.0060),  # New York City
            (19.4326, -99.1332),  # Mexico City
            (-23.5505, -46.6333),  # São Paulo
            (-33.8688, 151.2093),  # Sydney
        ],
        dtype=np.float64,
    )


def test_cylindrical(
    engine_cylindrical: SupervisedMDS, X: np.typing.NDArray[np.float64], city_coordinates: np.typing.NDArray[np.float64]
) -> None:
    engine_cylindrical.fit(X, city_coordinates)
    X_proj = engine_cylindrical.transform(X)
    assert X_proj.shape == (10, 2)


def test_geodesic(
    engine_geodesic: SupervisedMDS, X: np.typing.NDArray[np.float64], city_coordinates: np.typing.NDArray[np.float64]
) -> None:
    engine_geodesic.fit(X, city_coordinates)
    X_proj = engine_geodesic.transform(X)
    assert X_proj.shape == (10, 2)


def test_spherical(
    engine_spherical: SupervisedMDS, X: np.typing.NDArray[np.float64], city_coordinates: np.typing.NDArray[np.float64]
) -> None:
    engine_spherical.fit(X, city_coordinates)
    X_proj = engine_spherical.transform(X)
    assert X_proj.shape == (10, 2)
