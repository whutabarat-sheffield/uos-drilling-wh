import asyncio
import json
import logging
import yaml
from typing import Dict, List, Optional, Callable, Any, Coroutine
from dataclasses import dataclass
from functools import reduce
import pandas as pd
from aiomqtt import Client, Message, MqttError
from abyss.mqtt.confighandler import ConfigHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------ Data Converter Component ------

class MQTTDataConverter:
    """Converts MQTT messages to pandas DataFrames."""
    
    def __init__(self, config_handler: ConfigHandler, source_id: Optional[str] = None):
        """Initialize with configuration and optional source identifier."""
        self.config = config_handler
        self.source_id = source_id
        logger.info(f"Initialized converter for source: {source_id}")
    
    def convert_to_df(self, result_msg: Optional[str] = None, trace_msg: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Convert MQTT messages to a DataFrame."""
        result_df = self._parse_result(result_msg) if result_msg else None
        trace_df = self._parse_trace(trace_msg) if trace_msg else None
        
        return self._merge_dataframes(result_df, trace_df)
    
    # The parsing methods remain largely the same
    def _reduce_dict(self, data_dict: Dict, search_key: str) -> Any:
        """Helper method to extract values from nested dictionaries."""
        logger.debug(f"Reducing dictionary for key: {search_key}")
        
        values = reduce(
            lambda acc, key: acc + [data_dict[key]] if search_key in key else acc,
            data_dict.keys(),
            []
        )
        
        if not values:
            raise KeyError(f"Key '{search_key}' not found in dictionary")
            
        return values[0]['Value']
    
    def _parse_result(self, result_msg: str) -> Optional[pd.DataFrame]:
        """Parse result message into DataFrame."""
        # Implementation remains the same
        # ...
    
    def _parse_trace(self, trace_msg: str) -> Optional[pd.DataFrame]:
        """Parse trace message into DataFrame."""
        # Implementation remains the same
        # ...
    
    def _merge_dataframes(self, result_df: Optional[pd.DataFrame], trace_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Merge result and trace DataFrames if both exist."""
        # Implementation remains the same
        # ...


class DataConverterFactory:
    """Factory for creating data converters."""
    
    def __init__(self, config_handler: ConfigHandler):
        """Initialize with configuration."""
        self.config = config_handler
        self._converters = {}
        logger.info("DataConverterFactory initialized")
    
    def get_converter(self, source_id: Optional[str] = None) -> MQTTDataConverter:
        """Get a converter for the specified source."""
        if source_id not in self._converters:
            logger.info(f"Creating new converter for source: {source_id}")
            self._converters[source_id] = MQTTDataConverter(self.config, source_id)
        return self._converters[source_id]


# ------ Message Handling Component ------

class TopicHandler:
    """Handles messages for a specific topic."""
    
    def __init__(self, base_topic: str, converter: MQTTDataConverter, 
                 processor: Callable[[pd.DataFrame], Coroutine[Any, Any, None]]):
        """Initialize with topic, converter, and processor."""
        self.base_topic = base_topic
        self.converter = converter
        self.processor = processor
        self.result_msg = None
        self.trace_msg = None
        logger.info(f"TopicHandler initialized for topic: {base_topic}")
    
    async def handle_message(self, message: Message) -> None:
        """Handle an incoming message."""
        topic = message.topic.value
        payload = message.payload.decode("utf-8") if isinstance(message.payload, bytes) else str(message.payload)
        
        if topic.endswith('/result'):
            logger.debug(f"Received result message for topic: {self.base_topic}")
            self.result_msg = payload
        elif topic.endswith('/trace'):
            logger.debug(f"Received trace message for topic: {self.base_topic}")
            self.trace_msg = payload
        
        # Try to process if we have both messages
        if self.result_msg and self.trace_msg:
            await self._process_messages()
    
    async def _process_messages(self) -> None:
        """Process the current messages and reset."""
        try:
            df = self.converter.convert_to_df(
                result_msg=self.result_msg,
                trace_msg=self.trace_msg
            )
            
            if df is not None:
                logger.info(f"Processing DataFrame with {len(df)} rows for topic: {self.base_topic}")
                await self.processor(df)
            
            # Reset after processing
            self.result_msg = None
            self.trace_msg = None
            
        except Exception as e:
            logger.error(f"Error processing messages: {str(e)}", exc_info=True)


# ------ MQTT Client Component ------

class AsyncMQTTClient:
    """Handles MQTT connection and message routing using asyncio."""
    
    def __init__(self, config_handler: ConfigHandler):
        """Initialize with configuration."""
        self.config = config_handler
        self.topic_handlers = {}
        self.client = None
        logger.info("Async MQTT client initialized")
    
    def register_handler(self, base_topic: str, handler: TopicHandler) -> None:
        """Register a handler for a topic pair."""
        self.topic_handlers[base_topic] = handler
        logger.info(f"Registered handler for topic: {base_topic}")
    
    async def connect(self) -> None:
        """Connect to the MQTT broker."""
        broker_host = self.config.mqtt.broker.host
        broker_port = self.config.mqtt.broker.port
        username = self.config.mqtt.broker.username
        password = self.config.mqtt.broker.password
        
        try:
            logger.info(f"Connecting to MQTT broker at {broker_host}:{broker_port}")
            self.client = Client(
                hostname=broker_host,
                port=broker_port,
                username=username if username else None,
                password=password if password else None
            )
            await self.client.connect()
            logger.info("Connected to MQTT broker")
            
        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {str(e)}", exc_info=True)
            raise
    
    async def subscribe_to_topics(self) -> None:
        """Subscribe to all registered topics."""
        if self.client is None or not self.client.is_connected:
            raise RuntimeError("MQTT client not connected")
        
        for base_topic in self.topic_handlers.keys():
            result_topic = f"{base_topic}/result"
            trace_topic = f"{base_topic}/trace"
            
            await self.client.subscribe(result_topic)
            await self.client.subscribe(trace_topic)
            logger.info(f"Subscribed to topics: {result_topic}, {trace_topic}")
    
    async def process_messages(self) -> None:
        """Process incoming messages."""
        if self.client is None or not self.client.is_connected:
            raise RuntimeError("MQTT client not connected")
        
        async with self.client.messages() as messages:
            async for message in messages:
                topic = message.topic.value
                
                # Find the appropriate handler
                for base_topic, handler in self.topic_handlers.items():
                    if topic.startswith(base_topic):
                        logger.debug(f"Routing message to handler for topic: {base_topic}")
                        await handler.handle_message(message)
                        break
    
    async def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            logger.info("Disconnected from MQTT broker")


# ------ System Coordinator ------

class AsyncMQTTProcessingSystem:
    """Coordinates the entire MQTT processing system using asyncio."""
    
    def __init__(self, config_path: str, 
                 data_processor: Callable[[pd.DataFrame], Coroutine[Any, Any, None]]):
        """Initialize the processing system."""
        self.config_handler = ConfigHandler.from_yaml(config_path)
        self.converter_factory = DataConverterFactory(self.config_handler)
        self.mqtt_client = AsyncMQTTClient(self.config_handler)
        self.data_processor = data_processor
        logger.info("Async MQTT processing system initialized")
    
    def register_topic_pair(self, base_topic: str, source_id: str = None) -> None:
        """Register a topic pair (result and trace) for processing."""
        # Create a converter for this source
        converter = self.converter_factory.get_converter(source_id)
        
        # Create a topic handler that will process both result and trace
        handler = TopicHandler(base_topic, converter, self.data_processor)
        
        # Register with MQTT client
        self.mqtt_client.register_handler(base_topic, handler)
        
        logger.info(f"Registered topic pair for processing: {base_topic} (source: {source_id})")
    
    async def start(self) -> None:
        """Start the processing system."""
        await self.mqtt_client.connect()
        await self.mqtt_client.subscribe_to_topics()
        logger.info("Async MQTT processing system started")
        
        # Process messages indefinitely
        await self.mqtt_client.process_messages()
    
    async def stop(self) -> None:
        """Stop the processing system."""
        await self.mqtt_client.disconnect()
        logger.info("Async MQTT processing system stopped")


# ------ Example Usage ------

async def process_drilling_data(df: pd.DataFrame) -> None:
    """Process the drilling data (async version)."""
    logger.info(f"Processing drilling data: {len(df)} rows")
    # Example of async processing
    await asyncio.sleep(0.1)  # Simulate some async processing work
    print(df.head())


async def main():
    # Initialize the system
    system = AsyncMQTTProcessingSystem("config.yaml", process_drilling_data)
    
    # Register topics for different drilling tools
    system.register_topic_pair("drilling/toolbox1/tool1", "tool1")
    system.register_topic_pair("drilling/toolbox1/tool2", "tool2")
    system.register_topic_pair("drilling/toolbox2/tool1", "tool3")
    
    try:
        print("System running. Press Ctrl+C to stop.")
        await system.start()
    except asyncio.CancelledError:
        # Stop the system when cancelled
        await system.stop()
        print("System stopped.")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        await system.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down.")