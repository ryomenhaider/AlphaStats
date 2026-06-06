class StatitudeError(Exception):
    """Base exception."""


class InsufficientDataError(StatitudeError):
    """Fewer observations than the method requires."""


class InvalidInputError(StatitudeError):
    """Wrong shape, type, or value."""


class ConvergenceError(StatitudeError):
    """Iterative method failed to converge."""


class MissingDependencyError(StatitudeError):
    """Optional dependency not installed."""
