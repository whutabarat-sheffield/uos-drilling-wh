from typing import Dict, List, Tuple, Any
from .processing import MessageProcessor

class MessageRouter:
    """Routes MQTT messages to appropriate processors."""
    
    def __init__(self):
        """Initialize message router."""
        self.routes: Dict[str, List[MessageProcessor]] = {}
        
    def add_route(self, topic_pattern: str, processor: MessageProcessor):
        """Add a route for a specific topic pattern.
        
        Args:
            topic_pattern: MQTT topic pattern
            processor: Message processor for this topic
        """
        if topic_pattern not in self.routes:
            self.routes[topic_pattern] = []
            
        self.routes[topic_pattern].append(processor)
        
    async def route_message(self, topic: str, payload: bytes) -> List[Tuple[str, str, Any]]:
        """Route a message to appropriate processors.
        
        Args:
            topic: MQTT topic
            payload: Raw message payload
            
        Returns:
            List of (type_id, key, data) tuples from all matching processors
        """
        results = []
        
        for pattern, processors in self.routes.items():
            if self._topic_matches(topic, pattern):
                for processor in processors:
                    result = await processor.process(topic, payload)
                    results.append(result)
                    
        return results
        
    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if a topic matches a pattern with wildcards."""
        # Same implementation as in MQTTSubscriber
        if pattern == topic:
            return True
            
        if '#' in pattern:
            prefix = pattern.split('#')[0].rstrip('/')
            return topic.startswith(prefix)
            
        if '+' in pattern:
            parts = pattern.split('/')
            topic_parts = topic.split('/')
            
            if len(parts) != len(topic_parts):
                return False
                
            for i, part in enumerate(parts):
                if part != '+' and part != topic_parts[i]:
                    return False
                    
            return True
            
        return False