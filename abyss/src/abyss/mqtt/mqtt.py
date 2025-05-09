import asyncio
from typing import Dict, Callable, Coroutine, Any, Set, Optional
import aiomqtt
import ssl

class MQTTSubscriber:
    """Handles MQTT connection and message subscription."""
    
    def __init__(
        self,
        broker: str,
        port: int = 1883,
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = False
    ):
        # Keep existing initialization
        self.broker = broker
        self.port = port
        self.client_id = client_id
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.subscribed_topics: Set[str] = set()
        self.message_callbacks: Dict[str, Callable] = {}
        self.running = False
        
    def add_topic_handler(self, topic: str, callback: Callable[[str, bytes], Coroutine[Any, Any, None]]):
        """Add a handler for a specific topic."""
        self.subscribed_topics.add(topic)
        self.message_callbacks[topic] = callback
        
    async def start(self):
        """Start the MQTT subscriber and process messages."""
        self.running = True
        
        try:
            async with aiomqtt.Client(
                hostname=self.broker,
                port=self.port,
                identifier=self.client_id,
                username=self.username,
                password=self.password,
                tls_context=ssl.create_default_context() if self.use_tls else None
            ) as client:
                # Subscribe to all registered topics
                for topic in self.subscribed_topics:
                    await client.subscribe(topic)
                    
                # Process messages
                async for message in client.messages:
                    if not self.running:
                        break
                            
                        topic = message.topic.value
                        payload = message.payload
                        
                        # Find matching callbacks
                        for pattern, callback in self.message_callbacks.items():
                            if self._topic_matches(topic, pattern):
                                try:
                                    # Call the callback with await since it's an async function
                                    await callback(topic, payload)
                                except Exception as e:
                                    print(f"MQTT error: {e}")
                                    self.running = False
                                    break
        except Exception as e:
            print(f"MQTT connection error: {e}")
            self.running = False
        finally:
            # Ensure proper cleanup when the method exits
            self.running = False
    
    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if a topic matches a pattern with wildcards."""
        # Split the topic and pattern into parts
        topic_parts = topic.split('/')
        pattern_parts = pattern.split('/')
        
        # If lengths don't match and it's not a multi-level wildcard
        if len(topic_parts) != len(pattern_parts) and '#' not in pattern_parts:
            return False
            
        # Check each part
        for i, pattern_part in enumerate(pattern_parts):
            # Multi-level wildcard matches everything after
            if pattern_part == '#':
                return True
            # Single-level wildcard matches only one level
            elif pattern_part == '+':
                if i >= len(topic_parts):
                    return False
                continue
            # Exact match required
            elif i >= len(topic_parts) or pattern_part != topic_parts[i]:
                return False
                
        # If we've checked all parts and they match
        return len(topic_parts) == len(pattern_parts)
        
    def stop(self):
        """Stop receiving messages."""
        self.running = False