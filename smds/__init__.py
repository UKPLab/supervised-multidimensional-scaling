import warnings

warnings.warn(
    "The 'supervised-multidimensional-scaling' package on PyPI has been renamed to smds. "
    "Please update your dependencies.",
    DeprecationWarning,
    stacklevel=2
)

from .smds import SupervisedMDS

__all__ = ["SupervisedMDS"]