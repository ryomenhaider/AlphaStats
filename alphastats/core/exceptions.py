class StatslibError(Exception):
    """Base exception."""


class InsufficientDataError(StatslibError):
    """Fewer observations than the method requires."""


class InvalidInputError(StatslibError):
    """Wrong shape, type, or value."""


class ConvergenceError(StatslibError):
    """Iterative method failed to converge."""


class MissingDependencyError(StatslibError):
    """Optional dependency not installed."""
