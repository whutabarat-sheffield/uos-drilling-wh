from typing import Set, Callable, Awaitable
from .mqtt import MQTTSubscriber
from .routing import MessageRouter
from .correlation import CorrelationEngine
from .processing import MessageProcessor
from .output import MultiOutputHandler, OutputHandler, CallbackOutputHandler
from .models import Message, MatchedMessageSet

class MQTTCorrelationService:
    """Service that correlates MQTT messages."""
    
    def __init__(
        self,
        broker: str,
        port: int = 1883,
        required_types: Set[str] = set(),
        match_timeout: float = 60.0
    ):
        """Initialize MQTT correlation service.
        
        Args:
            broker: MQTT broker address
            port: MQTT broker port
            required_types: Set of message types required for a complete match
            match_timeout: Time in seconds before partial matches expire
        """
        self.subscriber = MQTTSubscriber(broker, port)
        self.router = MessageRouter()
        self.correlation = CorrelationEngine(
            required_types or set(),
            match_timeout
        )
        self.output_handler = MultiOutputHandler()
        
    def add_processor(self, topic_pattern: str, processor: MessageProcessor):
        """Add a message processor for a specific topic pattern.
        
        Args:
            topic_pattern: MQTT topic pattern
            processor: Message processor for this topic
        """
        self.router.add_route(topic_pattern, processor)
        self.subscriber.add_topic_handler(topic_pattern, self._handle_message)
        
        # If this processor produces a new message type, add it to required types
        self.correlation.required_types.add(processor.type_id)
        
    def add_output_handler(self, handler: OutputHandler):
        """Add an output handler.
        
        Args:
            handler: Output handler to add
        """
        self.output_handler.add_handler(handler)
        
    def add_callback(self, callback: Callable[[MatchedMessageSet], Awaitable[None]]):
        """Add a callback function for matched message sets.
        
        Args:
            callback: Async function to call with matched message sets
        """
        handler = CallbackOutputHandler(callback)
        self.output_handler.add_handler(handler)
        
    async def _handle_message(self, topic: str, payload: bytes):
        """Handle an MQTT message.
        
        Args:
            topic: MQTT topic
            payload: Raw message payload
        """
        # Route the message to appropriate processors
        processed_messages = await self.router.route_message(topic, payload)
        
        # Add each processed message to the correlation engine
        for type_id, key, data in processed_messages:
            message = Message(type_id=type_id, key=key, data=data)
            matched_set = self.correlation.add_message(message)
            
            # If a complete match was formed, handle it
            if matched_set:
                await self.output_handler.handle(matched_set)
                
    async def start(self):
        """Start the correlation service."""
        await self.subscriber.start()
        
    async def stop(self):
        """Stop the correlation service."""
        self.subscriber.stop()