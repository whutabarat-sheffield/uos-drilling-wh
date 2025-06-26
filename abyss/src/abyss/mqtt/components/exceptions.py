"""
MQTT Components Exception Hierarchy

Standardized exception classes for consistent error handling throughout
the MQTT components system.
"""


class AbyssError(Exception):
    """
    Base exception for all Abyss MQTT component errors.
    
    All Abyss-specific exceptions should inherit from this base class
    to enable catching all Abyss errors with a single except clause.
    """
    pass


class AbyssConfigurationError(AbyssError):
    """
    Configuration-related errors.
    
    Raised when:
    - Configuration files are missing or malformed
    - Required configuration values are missing
    - Configuration validation fails
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


class AbyssSystemError(AbyssError):
    """
    System-level errors.
    
    Raised when:
    - Resource exhaustion occurs
    - Critical system failures happen
    - Unrecoverable errors occur
    """
    pass


# Specific communication exceptions
class MQTTConnectionError(AbyssCommunicationError):
    """
    MQTT connection failures.
    
    Raised when MQTT client cannot connect to broker or
    loses connection unexpectedly.
    """
    pass


class MQTTPublishError(AbyssCommunicationError):
    """
    MQTT message publishing failures.
    
    Raised when message publishing fails due to network issues,
    broker rejection, or client errors.
    """
    pass


class MQTTSubscriptionError(AbyssCommunicationError):
    """
    MQTT subscription failures.
    
    Raised when client cannot subscribe to required topics.
    """
    pass


# Specific processing exceptions
class MessageValidationError(AbyssProcessingError):
    """
    Message validation failures.
    
    Raised when incoming messages fail validation checks
    (format, schema, required fields, etc.).
    """
    pass


class MessageCorrelationError(AbyssProcessingError):
    """
    Message correlation failures.
    
    Raised when messages cannot be correlated properly
    or correlation timeouts occur.
    """
    pass


class DepthEstimationError(AbyssProcessingError):
    """
    Depth estimation algorithm failures.
    
    Raised when depth estimation algorithms fail to process
    data or produce invalid results.
    """
    pass


class DataConversionError(AbyssProcessingError):
    """
    Data conversion failures.
    
    Raised when data cannot be converted between formats
    or required data fields are missing.
    """
    pass


# Specific system exceptions
class BufferOverflowError(AbyssSystemError):
    """
    Message buffer overflow.
    
    Raised when message buffers exceed maximum capacity
    and cannot accept new messages.
    """
    pass


class ResourceExhaustionError(AbyssSystemError):
    """
    System resource exhaustion.
    
    Raised when system runs out of memory, disk space,
    or other critical resources.
    """
    pass


class ComponentInitializationError(AbyssSystemError):
    """
    Component initialization failures.
    
    Raised when components cannot be properly initialized
    due to missing dependencies or configuration issues.
    """
    pass


# Utility functions for exception handling
def wrap_exception(exc: Exception, new_exc_type: type, message: str = None) -> Exception:
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


def is_recoverable_error(exc: Exception) -> bool:
    """
    Determine if an error is potentially recoverable.
    
    Args:
        exc: Exception to check
        
    Returns:
        True if error might be recoverable with retry/fallback
    """
    # Communication errors are often recoverable
    if isinstance(exc, AbyssCommunicationError):
        return True
    
    # Some processing errors might be recoverable
    if isinstance(exc, (MessageValidationError, MessageCorrelationError)):
        return True
    
    # Configuration and system errors are usually not recoverable
    if isinstance(exc, (AbyssConfigurationError, AbyssSystemError)):
        return False
    
    # For unknown errors, assume not recoverable
    return False


def get_error_severity(exc: Exception) -> str:
    """
    Get the severity level of an error for logging purposes.
    
    Args:
        exc: Exception to categorize
        
    Returns:
        Severity level: 'critical', 'error', 'warning', or 'info'
    """
    if isinstance(exc, AbyssSystemError):
        return 'critical'
    
    if isinstance(exc, AbyssConfigurationError):
        return 'error'
    
    if isinstance(exc, AbyssCommunicationError):
        return 'warning'
    
    if isinstance(exc, AbyssProcessingError):
        return 'warning'
    
    return 'error'  # Default for unknown exceptions