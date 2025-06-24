import json
import logging
import threading
import queue
import paho.mqtt.client as mqtt
import pandas as pd
from functools import reduce
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
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
        """
        Initialize with configuration and optional source identifier.
        
        Args:
            config_handler: Configuration for MQTT message structure
            source_id: Optional identifier for the message source
        """
        self.config = config_handler
        self.source_id = source_id
        logger.info(f"Initialized converter for source: {source_id}")
    
    def convert_to_df(self, result_msg: Optional[str] = None, trace_msg: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Convert MQTT messages to a DataFrame.
        
        Args:
            result_msg: MQTT result message as JSON string
            trace_msg: MQTT trace message as JSON string
            
        Returns:
            DataFrame containing the processed data, or None if processing fails
        """
        result_df = self._parse_result(result_msg) if result_msg else None
        trace_df = self._parse_trace(trace_msg) if trace_msg else None
        
        return self._merge_dataframes(result_df, trace_df)
    
    def _reduce_dict(self, data_dict: Dict, search_key: str) -> Any:
        """Helper method to extract values from nested dictionaries."""
        logger.debug(f"Reducing dictionary for key: {search_key}")
        logger.debug(f"Data dictionary keys: {data_dict}")
        
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
        logger.info(f"Processing RESULT message for source: {self.source_id}")
        
        try:
            result_data = json.loads(result_msg)
            data = result_data["Messages"]["Payload"]
            
            # Use config to get correct data IDs
            torque_empty_vals = self._reduce_dict(data, self.config.mqtt.data_ids.torque_empty_vals)
            thrust_empty_vals = self._reduce_dict(data, self.config.mqtt.data_ids.thrust_empty_vals)
            step_vals = self._reduce_dict(data, self.config.mqtt.data_ids.step_vals)
            hole_id = self._reduce_dict(data, self.config.mqtt.data_ids.machine_id)
            local = self._reduce_dict(data, self.config.mqtt.data_ids.result_id)
            
            # Create DataFrame
            df = pd.DataFrame({
                'Step (nb)': step_vals,
                'I Torque Empty (A)': torque_empty_vals,
                'I Thrust Empty (A)': thrust_empty_vals,
                'HOLE_ID': [str(hole_id)] * len(step_vals),
                'local': [local] * len(step_vals),
                'PREDRILLED': [1] * len(step_vals),
                'source_id': [self.source_id] * len(step_vals) if self.source_id else [None] * len(step_vals)
            })
            
            df['local'] = df['local'].astype('int32')
            logger.info(f"Result DataFrame created: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing RESULT data: {str(e)}", exc_info=True)
            return None
    
    def _parse_trace(self, trace_msg: str) -> Optional[pd.DataFrame]:
        """Parse trace message into DataFrame."""
        logger.info(f"Processing TRACE message for source: {self.source_id}")
        
        try:
            trace_data = json.loads(trace_msg)
            data = trace_data["Messages"]["Payload"]
            
            position = self._reduce_dict(data, self.config.mqtt.data_ids.position)
            torque = self._reduce_dict(data, self.config.mqtt.data_ids.torque)
            thrust = self._reduce_dict(data, self.config.mqtt.data_ids.thrust)
            step = self._reduce_dict(data, self.config.mqtt.data_ids.step)
            
            df = pd.DataFrame({
                'Step (nb)': step,
                'Position (mm)': position,
                'I Torque (A)': torque,
                'I Thrust (A)': thrust,
                'source_id': [self.source_id] * len(step) if self.source_id else [None] * len(step)
            })
            
            df['Step (nb)'] = df['Step (nb)'].astype('int32')
            df['Position (mm)'] = -df['Position (mm)']  # Invert position
            
            logger.info(f"Trace DataFrame created: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing TRACE data: {str(e)}", exc_info=True)
            return None
    
    def _merge_dataframes(self, result_df: Optional[pd.DataFrame], trace_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Merge result and trace DataFrames if both exist."""
        if result_df is not None and trace_df is not None:
            try:
                combined_df = pd.merge(result_df, trace_df, on=['Step (nb)', 'source_id'], how='outer')
                logger.info(f"Combined DataFrame created: {len(combined_df)} rows")
                return combined_df
            except Exception as e:
                logger.error(f"Error merging DataFrames: {str(e)}", exc_info=True)
        
        return result_df if result_df is not None else trace_df


class DataConverterFactory:
    """Factory for creating data converters."""
    
    def __init__(self, config_handler: ConfigHandler):
        """Initialize with configuration."""
        self.config = config_handler
        self._converters = {}
        logger.info("DataConverterFactory initialized")
    
    def get_converter(self, source_id: Optional[str] = None) -> MQTTDataConverter:
        """
        Get a converter for the specified source.
        
        Args:
            source_id: Identifier for the data source
            
        Returns:
            A converter instance for the source
        """
        if source_id not in self._converters:
            logger.info(f"Creating new converter for source: {source_id}")
            self._converters[source_id] = MQTTDataConverter(self.config, source_id)
        return self._converters[source_id]


# ------ Message Handling Component ------

@dataclass
class MQTTMessage:
    """Represents an MQTT message with topic and payload."""
    topic: str
    payload: str
    qos: int = 0
    retain: bool = False


class TopicHandler:
    """Handles messages for a specific topic."""
    
    def __init__(self, topic: str, converter: MQTTDataConverter, processor: Callable[[pd.DataFrame], None]):
        """
        Initialize with topic, converter, and processor.
        
        Args:
            topic: The MQTT topic
            converter: Converter for the message data
            processor: Function to process the converted data
        """
        self.topic = topic
        self.converter = converter
        self.processor = processor
        self.result_msg = None
        self.trace_msg = None
        self._lock = threading.Lock()
        logger.info(f"TopicHandler initialized for topic: {topic}")
    
    def handle_message(self, message: MQTTMessage) -> None:
        """
        Handle an incoming message.
        
        Args:
            message: The MQTT message
        """
        with self._lock:
            if message.topic.endswith('/result'):
                logger.debug(f"Received result message for topic: {self.topic}")
                self.result_msg = message.payload
            elif message.topic.endswith('/trace'):
                logger.debug(f"Received trace message for topic: {self.topic}")
                self.trace_msg = message.payload
            
            # Try to process if we have both messages
            if self.result_msg and self.trace_msg:
                self._process_messages()
    
    def _process_messages(self) -> None:
        """Process the current messages and reset."""
        try:
            df = self.converter.convert_to_df(
                result_msg=self.result_msg,
                trace_msg=self.trace_msg
            )
            
            if df is not None:
                logger.info(f"Processing DataFrame with {len(df)} rows for topic: {self.topic}")
                self.processor(df)
            
            # Reset after processing
            self.result_msg = None
            self.trace_msg = None
            
        except Exception as e:
            logger.error(f"Error processing messages: {str(e)}", exc_info=True)


# ------ MQTT Client Component ------

class MQTTClient:
    """Handles MQTT connection and message routing."""
    
    def __init__(self, config_handler: ConfigHandler):
        """
        Initialize with configuration.
        
        Args:
            config_handler: Configuration for MQTT
        """
        self.config = config_handler
        self.client = mqtt.Client()
        self.topic_handlers = {}
        self.message_queue = queue.Queue()
        self.running = False
        self._setup_client()
        logger.info("MQTT client initialized")
    
    def _setup_client(self) -> None:
        """Set up the MQTT client callbacks."""
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # Set up authentication if provided
        if self.config.mqtt.broker.username and self.config.mqtt.broker.password:
            self.client.username_pw_set(
                self.config.mqtt.broker.username, 
                self.config.mqtt.broker.password
            )
    
    def _on_connect(self, client, userdata, flags, rc) -> None:
        """Callback for when the client connects to the broker."""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            # Subscribe to all configured topics
            for topic in self.topic_handlers.keys():
                self.client.subscribe(topic)
                logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.error(f"Failed to connect to MQTT broker with code: {rc}")
    
    def _on_message(self, client, userdata, msg) -> None:
        """Callback for when a message is received."""
        logger.debug(f"Message received on topic: {msg.topic}")
        # Add to queue for processing
        message = MQTTMessage(
            topic=msg.topic,
            payload=msg.payload.decode('utf-8'),
            qos=msg.qos,
            retain=msg.retain
        )
        self.message_queue.put(message)
    
    def _on_disconnect(self, client, userdata, rc) -> None:
        """Callback for when the client disconnects from the broker."""
        if rc != 0:
            logger.warning(f"Unexpected disconnection from MQTT broker with code: {rc}")
        else:
            logger.info("Disconnected from MQTT broker")
    
    def register_handler(self, topic: str, handler: TopicHandler) -> None:
        """
        Register a handler for a topic.
        
        Args:
            topic: The MQTT topic
            handler: The handler for the topic
        """
        self.topic_handlers[topic] = handler
        if self.client.is_connected():
            self.client.subscribe(topic)
            logger.info(f"Subscribed to topic: {topic}")
    
    def start(self) -> None:
        """Start the MQTT client and processing thread."""
        # Connect to broker
        broker_host = self.config.mqtt.broker.host
        broker_port = self.config.mqtt.broker.port
        
        try:
            logger.info(f"Connecting to MQTT broker at {broker_host}:{broker_port}")
            self.client.connect(broker_host, broker_port, 60)
            self.client.loop_start()
            
            # Start message processing thread
            self.running = True
            self.process_thread = threading.Thread(target=self._process_messages)
            self.process_thread.daemon = True
            self.process_thread.start()
            logger.info("MQTT client and processing thread started")
            
        except Exception as e:
            logger.error(f"Error starting MQTT client: {str(e)}", exc_info=True)
            raise
    
    def stop(self) -> None:
        """Stop the MQTT client and processing thread."""
        self.running = False
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("MQTT client stopped")
    
    def _process_messages(self) -> None:
        """Process messages from the queue."""
        while self.running:
            try:
                # Get message from queue with timeout
                message = self.message_queue.get(timeout=1.0)
                
                # Find the appropriate handler
                for topic_pattern, handler in self.topic_handlers.items():
                    if mqtt.topic_matches_sub(topic_pattern, message.topic):
                        logger.debug(f"Routing message to handler for topic: {topic_pattern}")
                        handler.handle_message(message)
                        break
                
                self.message_queue.task_done()
                
            except queue.Empty:
                # No message available, just continue
                pass
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}", exc_info=True)


# ------ System Coordinator ------

class MQTTProcessingSystem:
    """Coordinates the entire MQTT processing system."""
    
    def __init__(self, config_path: str, data_processor: Callable[[pd.DataFrame], None]):
        """
        Initialize the processing system.
        
        Args:
            config_path: Path to the configuration file
            data_processor: Function to process the converted data
        """
        self.config_handler = ConfigHandler.from_yaml(config_path)
        self.converter_factory = DataConverterFactory(self.config_handler)
        self.mqtt_client = MQTTClient(self.config_handler)
        self.data_processor = data_processor
        logger.info("MQTT processing system initialized")
    
    def register_topic_pair(self, base_topic: str, source_id: Optional[str] = None) -> None:
        """
        Register a topic pair (result and trace) for processing.
        
        Args:
            base_topic: The base topic without result/trace suffix
            source_id: Optional identifier for the data source
        """
        # Create a converter for this source
        converter = self.converter_factory.get_converter(source_id)
        
        # Create handlers for result and trace topics
        result_topic = f"{base_topic}/result"
        trace_topic = f"{base_topic}/trace"
        
        # Create a topic handler that will process both result and trace
        handler = TopicHandler(base_topic, converter, self.data_processor)
        
        # Register with MQTT client
        self.mqtt_client.register_handler(result_topic, handler)
        self.mqtt_client.register_handler(trace_topic, handler)
        
        logger.info(f"Registered topic pair for processing: {base_topic} (source: {source_id})")
    
    def start(self) -> None:
        """Start the processing system."""
        self.mqtt_client.start()
        logger.info("MQTT processing system started")
    
    def stop(self) -> None:
        """Stop the processing system."""
        self.mqtt_client.stop()
        logger.info("MQTT processing system stopped")


# ------ Example Usage ------

def process_drilling_data(df: pd.DataFrame) -> None:
    """
    Process the drilling data.
    
    Args:
        df: DataFrame containing the drilling data
    """
    # This is where you would implement your data processing logic
    logger.info(f"Processing drilling data: {len(df)} rows")
    # Example: Store data, analyze it, trigger alerts, etc.
    print(df.head())


if __name__ == "__main__":
    # Initialize and start the system
    system = MQTTProcessingSystem("config.yaml", process_drilling_data)
    
    # Register topics for different drilling tools
    system.register_topic_pair("drilling/toolbox1/tool1", "tool1")
    system.register_topic_pair("drilling/toolbox1/tool2", "tool2")
    system.register_topic_pair("drilling/toolbox2/tool1", "tool3")
    
    try:
        # Start the system
        system.start()
        
        # Keep the main thread running
        print("System running. Press Ctrl+C to stop.")
        while True:
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        # Stop the system on Ctrl+C
        system.stop()
        print("System stopped.")