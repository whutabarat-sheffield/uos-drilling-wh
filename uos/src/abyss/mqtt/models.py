from dataclasses import dataclass, field
from typing import Any, Dict, Set #, Callable, Optional
# import functools
import time
import uuid

"""TODO: Figure out how to use composite key for matching. This will combine ResultID and SourceTimestamp"""

@dataclass
class Message:
    """Container for a single message with its metadata."""
    type_id: str             # Identifier for the type of message
    key: str                 # Key used for matching
    data: Any                # Actual message content
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    timestamp: float = field(default_factory=time.time)     # Creation timestamp
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique ID

@dataclass
class MatchedMessageSet:
    """Container for matched messages."""
    key: str                           # The common correlation key
    messages: Dict[str, Message]       # Messages organized by their type_id
    timestamp: float = field(default_factory=time.time)  # When match was created
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique ID
    
    def is_complete(self, required_types: Set[str]) -> bool:
        """Check if all required message types are present."""
        return set(self.messages.keys()) == required_types
