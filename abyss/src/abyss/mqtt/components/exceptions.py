"""
MQTT Components Exception Hierarchy

Simplified exception classes for the MQTT components system.
Only includes exceptions that are actually used in the codebase.
"""


class AbyssError(Exception):
    """
    Base exception for all Abyss MQTT component errors.
    
    All Abyss-specific exceptions should inherit from this base class
    to enable catching all Abyss errors with a single except clause.
    """
    pass


class AbyssCommunicationError(AbyssError):
    """
    MQTT communication errors.
    
    Raised when:
    - MQTT connection fails
    - Message publishing fails
    - Network communication issues occur
    """
    pass


class AbyssProcessingError(AbyssError):
    """
    Data processing errors.
    
    Raised when:
    - Message processing fails
    - Data validation fails  
    - Algorithm execution fails
    """
    pass


class MQTTPublishError(AbyssCommunicationError):
    """
    MQTT message publishing failures.
    
    Raised when message publishing fails due to network issues,
    broker rejection, or client errors.
    """
    pass


# Utility functions for exception handling
def wrap_exception(exc: Exception, new_exc_type: type, message: str | None = None) -> Exception:
    """
    Wrap an existing exception in a new exception type while preserving the original.
    
    Args:
        exc: Original exception to wrap
        new_exc_type: New exception type to wrap with
        message: Optional custom message (uses original message if None)
        
    Returns:
        New exception instance with original exception chained
    """
    if message is None:
        message = str(exc)
    
    new_exc = new_exc_type(f"{message}: {type(exc).__name__}: {exc}")
    new_exc.__cause__ = exc
    return new_exc