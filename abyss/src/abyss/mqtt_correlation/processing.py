import json
from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Optional

# class MessageProcessor(ABC):
#     """Base class for processing raw MQTT messages."""
    
#     @abstractmethod
#     async def process(self, topic: str, payload: bytes) -> Tuple[str, str, Any]:
#         """Process a raw MQTT message.
        
#         Args:
#             topic: MQTT topic
#             payload: Raw message payload
            
#         Returns:
#             Tuple of (message_type, correlation_key, processed_data)
#         """
#         pass
        

class MessageProcessor:
    """Base class for processing MQTT messages."""
    
    def __init__(self, type_id: str):
        """Initialize message processor.
        
        Args:
            type_id: Unique identifier for the message type this processor handles
        """
        self.type_id = type_id
        
    async def process(self, topic: str, payload: bytes) -> tuple[str, str, dict]:
        """Process an MQTT message.
        
        Args:
            topic: MQTT topic
            payload: Raw message payload
            
        Returns:
            Tuple of (type_id, key, data)
        """
        raise NotImplementedError("Subclasses must implement process method")













class JSONMessageProcessor(MessageProcessor):
    """Process JSON-formatted MQTT messages."""
    
    def __init__(
        self,
        type_id: str,
        key_field: str,
        topic_prefix: Optional[str] = None
    ):
        """Initialize JSON message processor.
        
        Args:
            type_id: Type identifier for messages processed by this processor
            key_field: Field in the JSON to use as correlation key
            topic_prefix: Optional topic prefix to strip from topics
        """
        self.type_id = type_id
        self.key_field = key_field
        self.topic_prefix = topic_prefix
        
    async def process(self, topic: str, payload: bytes) -> Tuple[str, str, Any]:
        """Process a JSON message.
        
        Args:
            topic: MQTT topic
            payload: JSON message payload
            
        Returns:
            Tuple of (message_type, correlation_key, processed_data)
        """
        # Parse JSON payload
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            # Handle non-JSON payloads
            data = {"raw": payload.decode("utf-8", errors="replace")}
            
        # Extract correlation key
        key = data.get(self.key_field, "unknown")
        
        # Add topic information (stripped of prefix if needed)
        if self.topic_prefix and topic.startswith(self.topic_prefix):
            topic = topic[len(self.topic_prefix):]
            
        data["_topic"] = topic
        
        return self.type_id, key, data