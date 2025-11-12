from enum import Enum

class ManifoldType(str, Enum):
    """
    Enumeration of all supported manifold types.
    
    Inherits from 'str' so that members can be used 
    directly as string keys in the registry.
    """
    # Aliases for 'cluster'
    TRIVIAL = "trivial"
    CLUSTER = "cluster"
    
    # Aliases for 'linear'
    EUCLIDEAN = "euclidean"
    LINEAR = "linear"
    
    LOG_LINEAR = "log_linear"
    HELIX = "helix"
    DISCRETE_CIRCULAR = "discrete_circular"
    CHAIN = "chain"
    SEMICIRCULAR = "semicircular"
    LOG_SEMICIRCULAR = "log_semicircular"
    SPHERE_CHORD = "sphere_chord"
    GEODESIC = "geodesic"
    CYLINDER_CHORD = "cylinder_chord"