"""
Refactored implementation of the MQTT Drilling Data Analyser with clear separation of concerns.
This design implements a modular approach with specialized components.
"""
ALGO_VERSION = "0.2.1.b1"


import logging
import paho.mqtt.client as mqtt
import yaml
import json
import time
from typing import Dict, List, Optional, Callable, Any, TypeVar, Generic, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict

# Import your specific modules
from abyss.uos_depth_est_core import DepthInference
from abyss.uos_depth_est_utils import convert_mqtt_to_df

# Type definitions
T = TypeVar('T')  # Generic type for the data
Config = Dict[str, Any]  # Type for configuration

@dataclass(frozen=True)
class TimestampedData(Generic[T]):
    """Immutable container for timestamped data with source information."""
    timestamp: float  # Unix timestamp
    data: T
    source: str
    
    def __hash__(self):
        return hash((self.timestamp, self.source))
    
    def __eq__(self, other):
        if not isinstance(other, TimestampedData):
            return False
        return self.timestamp == other.timestamp and self.source == other.source

class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_path: str):
        """
        Initialize with a configuration file path.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Config:
        """Load configuration from the specified YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                logging.info(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            logging.error(f"Failed to load configuration: {str(e)}")
            raise ValueError(f"Failed to load configuration from {self.config_path}: {str(e)}")
    
    def get_broker_config(self) -> Dict[str, Any]:
        """Get the MQTT broker configuration."""
        return self.config.get('mqtt', {}).get('broker', {})
    
    def get_topic_root(self) -> str:
        """Get the root for MQTT topics."""
        return self.config.get('mqtt', {}).get('listener', {}).get('root', '')
    
    def get_result_topic_suffix(self) -> str:
        """Get the suffix for result topics."""
        return self.config.get('mqtt', {}).get('listener', {}).get('result', '')
    
    def get_trace_topic_suffix(self) -> str:
        """Get the suffix for trace topics."""
        return self.config.get('mqtt', {}).get('listener', {}).get('trace', '')
    
    def get_keypoints_topic_suffix(self) -> str:
        """Get the suffix for keypoints result topics."""
        return self.config.get('mqtt', {}).get('estimation', {}).get('keypoints', '')
    
    def get_depth_est_topic_suffix(self) -> str:
        """Get the suffix for depth estimation result topics."""
        return self.config.get('mqtt', {}).get('estimation', {}).get('depth_estimation', '')
    
    def get_full_config(self) -> Config:
        """Get the full configuration dictionary."""
        return self.config

class MessageParser:
    """Handles parsing and preparation of MQTT messages."""
    
    @staticmethod
    def parse_message(topic: str, payload: bytes) -> Optional[TimestampedData]:
        """
        Parse an MQTT message into a TimestampedData object.
        
        Args:
            topic: MQTT topic
            payload: Message payload
            
        Returns:
            TimestampedData object or None if parsing fails
        """
        try:
            data = json.loads(payload)
            source_timestamps = MessageParser.find_in_dict(data, 'SourceTimestamp')
            
            if not source_timestamps:
                logging.error(f"No SourceTimestamp found in message on topic {topic}")
                return None
                
            dt = datetime.strptime(source_timestamps[0], "%Y-%m-%dT%H:%M:%SZ")
            unix_timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
            
            return TimestampedData(
                timestamp=unix_timestamp,
                data=data,
                source=topic
            )
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {topic}")
            return None
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            return None
    
    @staticmethod
    def find_in_dict(data: dict, target_key: str) -> list:
        """
        Recursively search for a key in a nested dictionary.
        Returns list of values found for that key.
        """
        results = []
        
        def _search(current_dict):
            for key, value in current_dict.items():
                if key == target_key:
                    results.append(value)
                if isinstance(value, dict):
                    _search(value)
        
        _search(data)
        return results
    
    @staticmethod
    def extract_tool_info(topic: str) -> Optional[Tuple[str, str]]:
        """
        Extract toolbox ID and tool ID from a topic.
        
        Args:
            topic: MQTT topic string
            
        Returns:
            Tuple of (toolbox_id, tool_id) or None if extraction fails
        """
        parts = topic.split('/')
        if len(parts) >= 4:
            return parts[1], parts[2]
        return None
    
    @staticmethod
    def format_timestamp(timestamp: float) -> str:
        """
        Format a Unix timestamp as an ISO-8601 string in UTC.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            ISO-8601 formatted string
        """
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    @staticmethod
    def create_result_payload(value: Any, timestamp: float, machine_id: str, result_id: str, algo_version: str) -> Dict[str, Any]:
        """
        Create a standardized result payload.
        
        Args:
            value: Result value
            timestamp: Unix timestamp
            machine_id: Machine identifier
            result_id: Result identifier
            algo_version: Algorithm version
            
        Returns:
            Dictionary with result payload
        """
        return {
            "Value": value,
            "SourceTimestamp": MessageParser.format_timestamp(timestamp),
            "MachineId": machine_id,
            "ResultId": result_id,
            "AlgoVersion": algo_version
        }

class MessageBuffer:
    """Manages the buffering and matching of timestamped messages."""
    
    def __init__(self, time_window: float = 1.0, cleanup_interval: float = 10.0):
        """
        Initialize the message buffer.
        
        Args:
            time_window: Time window in seconds for matching messages
            cleanup_interval: Interval in seconds for cleanup operations
        """
        self.buffers: Dict[str, List[TimestampedData]] = defaultdict(list)
        self.time_window = time_window
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = datetime.now(timezone.utc).timestamp()
    
    def add_message(self, topic_pattern: str, message: TimestampedData) -> None:
        """
        Add a message to the appropriate buffer.
        
        Args:
            topic_pattern: Pattern for the topic
            message: TimestampedData to add
        """
        self.buffers[topic_pattern].append(message)
        logging.debug(f"Added message to buffer {topic_pattern}, buffer size: {len(self.buffers[topic_pattern])}")
    
    def find_matches(self, result_topic_pattern: str, trace_topic_pattern: str) -> List[List[TimestampedData]]:
        """
        Find matching pairs of messages from result and trace buffers.
        
        Args:
            result_topic_pattern: Pattern for result topics
            trace_topic_pattern: Pattern for trace topics
            
        Returns:
            List of matched message pairs
        """
        matches = []
        to_remove_result = []
        to_remove_trace = []
        
        result_messages = self.buffers.get(result_topic_pattern, [])
        trace_messages = self.buffers.get(trace_topic_pattern, [])
        
        for result_msg in result_messages:
            r_tool_info = MessageParser.extract_tool_info(result_msg.source)
            if not r_tool_info:
                continue
                
            r_tool_key = f"{r_tool_info[0]}/{r_tool_info[1]}"
            
            for trace_msg in trace_messages:
                t_tool_info = MessageParser.extract_tool_info(trace_msg.source)
                if not t_tool_info:
                    continue
                    
                t_tool_key = f"{t_tool_info[0]}/{t_tool_info[1]}"
                
                if (r_tool_key == t_tool_key and 
                    abs(result_msg.timestamp - trace_msg.timestamp) <= self.time_window and
                    result_msg not in to_remove_result and 
                    trace_msg not in to_remove_trace):
                    
                    matches.append([result_msg, trace_msg])
                    to_remove_result.append(result_msg)
                    to_remove_trace.append(trace_msg)
        
        # Remove matched messages from buffers
        self.buffers[result_topic_pattern] = [msg for msg in result_messages if msg not in to_remove_result]
        self.buffers[trace_topic_pattern] = [msg for msg in trace_messages if msg not in to_remove_trace]
        
        return matches
    
    def cleanup_old_messages(self, result_topic_pattern: str, trace_topic_pattern: str) -> None:
        """
        Remove messages older than a threshold while preserving potential matches.
        
        Args:
            result_topic_pattern: Pattern for result topics
            trace_topic_pattern: Pattern for trace topics
        """
        try:
            current_time = datetime.now(timezone.utc).timestamp()
            
            # Only clean up at appropriate intervals
            if current_time - self.last_cleanup < self.cleanup_interval:
                return
                
            self.last_cleanup = current_time
            
            # Get message buffers
            result_messages = self.buffers.get(result_topic_pattern, [])
            trace_messages = self.buffers.get(trace_topic_pattern, [])
            
            # Create a set of potentially matchable toolbox/tool combinations
            potential_matches = set()
            recent_time_threshold = current_time - (2 * self.time_window)
            
            # Collect potentially matchable tool ids from both result and trace messages
            for msg_list in [result_messages, trace_messages]:
                for msg in msg_list:
                    tool_info = MessageParser.extract_tool_info(msg.source)
                    if tool_info and current_time - msg.timestamp <= recent_time_threshold:
                        potential_matches.add(f"{tool_info[0]}/{tool_info[1]}")
            
            # More conservative time threshold for old message removal
            old_time_threshold = current_time - (5 * self.cleanup_interval)
            
            # Helper function to filter messages
            def should_keep_message(msg):
                if current_time - msg.timestamp <= recent_time_threshold:
                    return True
                    
                tool_info = MessageParser.extract_tool_info(msg.source)
                if not tool_info:
                    return False
                    
                tool_key = f"{tool_info[0]}/{tool_info[1]}"
                
                # Keep if part of a potential match or not too old
                return tool_key in potential_matches or current_time - msg.timestamp <= old_time_threshold
            
            # Apply filtering
            new_result_messages = [msg for msg in result_messages if should_keep_message(msg)]
            new_trace_messages = [msg for msg in trace_messages if should_keep_message(msg)]
            
            # Update buffers
            removed_count = len(result_messages) - len(new_result_messages) + len(trace_messages) - len(new_trace_messages)
            self.buffers[result_topic_pattern] = new_result_messages
            self.buffers[trace_topic_pattern] = new_trace_messages
            
            if removed_count > 0:
                logging.info(f"Removed {removed_count} old messages during cleanup")
                
        except Exception as e:
            logging.error(f"Error in cleanup_old_messages: {str(e)}", exc_info=True)

class MQTTClientManager:
    """Manages MQTT client connections and message handling."""
    
    def __init__(self, config_manager: ConfigManager, on_message_callback: Callable[[str, bytes], None]):
        """
        Initialize the MQTT client manager.
        
        Args:
            config_manager: Configuration manager
            on_message_callback: Callback for handling received messages
        """
        self.config_manager = config_manager
        self.on_message_callback = on_message_callback
        self.clients = {}
    
    def create_client(self, client_id: str, topics: List[str]) -> mqtt.Client:
        """
        Create and configure an MQTT client.
        
        Args:
            client_id: Client identifier
            topics: List of topics to subscribe to
            
        Returns:
            Configured MQTT client
        """
        # Create client with proper parameter order
        client = mqtt.Client(client_id=client_id)
        
        broker_config = self.config_manager.get_broker_config()
        if broker_config.get('username') and broker_config.get('password'):
            client.username_pw_set(
                broker_config['username'], 
                broker_config['password']
            )
        
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logging.info(f"Client {client_id} connected successfully")
                for topic in topics:
                    client.subscribe(topic)
                    logging.info(f"Subscribed to {topic}")
            else:
                logging.error(f"Failed to connect client {client_id} with result code {rc}")
        
        def on_message(client, userdata, msg):
            try:
                self.on_message_callback(msg.topic, msg.payload)
            except Exception as e:
                logging.error(f"Error in message callback: {str(e)}", exc_info=True)
        
        client.on_connect = on_connect
        client.on_message = on_message
        
        self.clients[client_id] = client
        return client
    
    def connect_all(self) -> None:
        """Connect all registered clients to the broker."""
        broker_config = self.config_manager.get_broker_config()
        host = broker_config.get('host', 'localhost')
        port = broker_config.get('port', 1883)
        
        for client_id, client in self.clients.items():
            try:
                client.connect(host, port)
                client.loop_start()
                logging.info(f"Started client {client_id}")
            except Exception as e:
                logging.error(f"Error connecting client {client_id}: {str(e)}", exc_info=True)
    
    def disconnect_all(self) -> None:
        """Disconnect all clients from the broker."""
        for client_id, client in self.clients.items():
            try:
                client.loop_stop()
                client.disconnect()
                logging.info(f"Disconnected client {client_id}")
            except Exception as e:
                logging.error(f"Error disconnecting client {client_id}: {str(e)}", exc_info=True)
    
    def publish(self, client_id: str, topic: str, payload: Dict[str, Any]) -> bool:
        """
        Publish a message using the specified client.
        
        Args:
            client_id: Client identifier
            topic: Topic to publish to
            payload: Message payload
            
        Returns:
            True if successful, False otherwise
        """
        client = self.clients.get(client_id)
        if not client:
            logging.error(f"Client {client_id} not found")
            return False
            
        try:
            message_json = json.dumps(payload)
            client.publish(topic, message_json)
            logging.debug(f"Published to {topic}: {message_json}")
            return True
        except Exception as e:
            logging.error(f"Error publishing to {topic}: {str(e)}", exc_info=True)
            return False

class DepthEstimationProcessor:
    """Processes matched messages to perform depth estimation."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the depth estimation processor.
        
        Args:
            config_manager: Configuration manager
        """
        self.config_manager = config_manager
        self.depth_inference = DepthInference()
        self.machine_id = "MACHINE_ID"
        self.result_id = "RESULT_ID"
        self.algo_version = "0.1.1"
    
    def process_matched_pair(self, result_msg: TimestampedData, trace_msg: TimestampedData) -> Dict[str, Any]:
        """
        Process a matched pair of result and trace messages.
        
        Args:
            result_msg: Result message
            trace_msg: Trace message
            
        Returns:
            Dictionary with processing results or None if processing fails
        """
        tool_info = None
        try:
            tool_info = MessageParser.extract_tool_info(result_msg.source)
            if not tool_info:
                logging.error("Invalid topic format, cannot extract tool info")
                return self._create_error_result("unknown", "unknown", result_msg.timestamp, 
                                               "Invalid topic format, cannot extract tool info")
                
            toolbox_id, tool_id = tool_info
            
            # Convert MQTT data to dataframe
            try:
                df = convert_mqtt_to_df(
                    json.dumps(result_msg.data), 
                    json.dumps(trace_msg.data), 
                    conf=self.config_manager.get_full_config()
                )
            except Exception as e:
                logging.error(f"Failed to convert MQTT data to dataframe: {str(e)}")
                return self._create_error_result(toolbox_id, tool_id, result_msg.timestamp, 
                                               "Failed to convert data")
            
            # Check if dataframe is valid
            if df is None or df.empty:
                logging.error("Converted dataframe is empty or None")
                return self._create_error_result(toolbox_id, tool_id, result_msg.timestamp, 
                                               "Empty dataframe")
            
            # Extract identifiers
            try:
                self.machine_id = str(df.iloc[0]['HOLE_ID'])
                self.result_id = str(df.iloc[0]['local'])
            except (KeyError, IndexError) as e:
                logging.warning(f"Failed to extract identifiers from dataframe: {str(e)}")
                # Continue with default values
            
            # Debug logging - save raw data and dataframe to files if debug enabled
            self._save_debug_data(result_msg, trace_msg, df, toolbox_id, tool_id)
            
            # Check if we have enough data for depth estimation
            if 'Step (nb)' not in df.columns or len(df['Step (nb)'].unique()) < 2:
                logging.warning("Insufficient step data for depth estimation")
                return self._create_error_result(toolbox_id, tool_id, result_msg.timestamp, 
                                               "Not enough steps to estimate")
            
            # Perform depth estimation
            return self._perform_depth_estimation(df, toolbox_id, tool_id, result_msg.timestamp)
            
        except Exception as e:
            logging.critical(f"Critical error in process_matched_pair: {str(e)}", exc_info=True)
            if tool_info:
                return self._create_error_result(tool_info[0], tool_info[1], result_msg.timestamp, 
                                               f"Processing error: {str(e)}")
            return self._create_error_result("unknown", "unknown", result_msg.timestamp,
                                           f"Processing error: {str(e)}")
    
    def _save_debug_data(self, result_msg: TimestampedData, trace_msg: TimestampedData,
                        df, toolbox_id: str, tool_id: str) -> None:
        """Save debug data to files if debug logging is enabled."""
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            timestamp_str = str(result_msg.timestamp).replace('.', '_')
            debug_file = f"match_{timestamp_str}.txt"
            csv_file = f"data_{timestamp_str}.csv"
            
            try:
                with open(debug_file, 'w') as file:
                    file.write(f"Toolbox ID: {toolbox_id}\n")
                    file.write(f"Tool ID: {tool_id}\n")
                    file.write(f"Timestamp: {datetime.fromtimestamp(result_msg.timestamp)}\n\n")
                    file.write(f"RESULT:\n\n")
                    file.write(json.dumps(result_msg.data, indent=2))
                    file.write(f"\n\nTRACE:\n\n")
                    file.write(json.dumps(trace_msg.data, indent=2))
                
                df.to_csv(csv_file, index=False)
                logging.debug(f"Debug data saved to {debug_file} and {csv_file}")
            except Exception as e:
                logging.error(f"Failed to save debug data: {str(e)}")
    
    def _create_error_result(self, toolbox_id: str, tool_id: str, timestamp: float, 
                           error_message: str) -> Dict[str, Any]:
        """Create a result dictionary for error conditions."""
        return {
            "toolbox_id": toolbox_id,
            "tool_id": tool_id,
            "timestamp": timestamp,
            "keypoints_value": f"Error: {error_message}",
            "depth_est_value": f"Error: {error_message}",
            "machine_id": self.machine_id,
            "result_id": self.result_id,
            "algo_version": self.algo_version,
            "success": False
        }
    
    def _perform_depth_estimation(self, df, toolbox_id: str, tool_id: str, 
                                timestamp: float) -> Dict[str, Any]:
        """Perform depth estimation using the inference model."""
        try:
            # Run inference model to get keypoints
            l_result = self.depth_inference.infer3_common(df)
            
            # Calculate depth estimation from keypoints
            depth_estimation = [l_result[i+1] - l_result[i] for i in range(len(l_result)-1)]
            
            logging.info(f"Keypoints: {l_result}")
            logging.info(f"Depth estimation: {depth_estimation}")
            
            return {
                "toolbox_id": toolbox_id,
                "tool_id": tool_id,
                "timestamp": timestamp,
                "keypoints_value": l_result,
                "depth_est_value": depth_estimation,
                "machine_id": self.machine_id,
                "result_id": self.result_id,
                "algo_version": self.algo_version,
                "success": True
            }
            
        except Exception as e:
            logging.error(f"Error in depth estimation: {str(e)}", exc_info=True)
            return self._create_error_result(toolbox_id, tool_id, timestamp, 
                                           f"Depth estimation error: {str(e)}")

class MQTTDrillingDataAnalyser:
    """Main class that orchestrates the drilling data analysis process."""
    
    def __init__(self, config_path: str = 'mqtt_conf.yaml'):
        """
        Initialize the MQTT drilling data analyser.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        # Initialize components
        self.config_manager = ConfigManager(config_path)
        self.message_buffer = MessageBuffer(time_window=1.0, cleanup_interval=30.0)
        self.processor = DepthEstimationProcessor(self.config_manager)
        
        # Create topics
        root = self.config_manager.get_topic_root()
        result_suffix = self.config_manager.get_result_topic_suffix()
        trace_suffix = self.config_manager.get_trace_topic_suffix()
        
        self.result_topic_pattern = f"{root}/+/+/{result_suffix}"
        self.trace_topic_pattern = f"{root}/+/+/{trace_suffix}"
        
        # Initialize MQTT client manager
        self.mqtt_manager = MQTTClientManager(
            self.config_manager,
            self._on_message_received
        )
        
        # Create clients
        self.mqtt_manager.create_client(
            "result_client", 
            [self.result_topic_pattern]
        )
        
        self.mqtt_manager.create_client(
            "trace_client", 
            [self.trace_topic_pattern]
        )
    
    def _on_message_received(self, topic: str, payload: bytes) -> None:
        """
        Handle received MQTT messages.
        
        Args:
            topic: Message topic
            payload: Message payload
        """
        try:
            # Parse the message
            message = MessageParser.parse_message(topic, payload)
            if not message:
                return
            
            # Add to appropriate buffer
            if self.config_manager.get_result_topic_suffix() in topic:
                self.message_buffer.add_message(self.result_topic_pattern, message)
            elif self.config_manager.get_trace_topic_suffix() in topic:
                self.message_buffer.add_message(self.trace_topic_pattern, message)
            else:
                logging.warning(f"Received message on unexpected topic: {topic}")
                return
            
            # Find matches
            matches = self.message_buffer.find_matches(
                self.result_topic_pattern, 
                self.trace_topic_pattern
            )
            
            # Process matches
            for match in matches:
                result_msg = next(m for m in match if self.config_manager.get_result_topic_suffix() in m.source)
                trace_msg = next(m for m in match if self.config_manager.get_trace_topic_suffix() in m.source)
                
                result = self.processor.process_matched_pair(result_msg, trace_msg)
                if result:
                    self._publish_results(result)
            
            # Clean up old messages periodically
            self.message_buffer.cleanup_old_messages(
                self.result_topic_pattern, 
                self.trace_topic_pattern
            )
            
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}", exc_info=True)
    
    def _publish_results(self, result: Dict[str, Any]) -> None:
        """
        Publish processing results to appropriate topics.
        
        Args:
            result: Result dictionary from processor
        """
        if not result:
            return
            
        toolbox_id = result.get("toolbox_id")
        tool_id = result.get("tool_id")
        timestamp = result.get("timestamp")
        success = result.get("success", False)
        
        if not all([toolbox_id, tool_id, timestamp is not None]):
            logging.error("Missing required fields in result for publishing")
            return
        
        # Create payloads
        keypoints_payload = MessageParser.create_result_payload(
            result.get("keypoints_value"),
            float(timestamp) if timestamp is not None else 0.0,
            result.get("machine_id", "MACHINE_ID"),
            result.get("result_id", "RESULT_ID"),
            result.get("algo_version", ALGO_VERSION)
        )
        
        depth_est_payload = MessageParser.create_result_payload(
            result.get("depth_est_value"),
            float(timestamp) if timestamp is not None else 0.0,
            result.get("machine_id", "MACHINE_ID"),
            result.get("result_id", "RESULT_ID"),
            result.get("algo_version", ALGO_VERSION)
        )
        
        # Create topics
        root = self.config_manager.get_topic_root()
        keypoints_topic = f"{root}/{toolbox_id}/{tool_id}/{self.config_manager.get_keypoints_topic_suffix()}"
        depth_est_topic = f"{root}/{toolbox_id}/{tool_id}/{self.config_manager.get_depth_est_topic_suffix()}"
        
        # Publish results
        self.mqtt_manager.publish("result_client", keypoints_topic, keypoints_payload)
        self.mqtt_manager.publish("result_client", depth_est_topic, depth_est_payload)
        
        if success:
            logging.info(f"Published successful results for {toolbox_id}/{tool_id}")
        else:
            logging.warning(f"Published error results for {toolbox_id}/{tool_id}")
    
    def run(self) -> None:
        """Main method to start the analysis process."""
        try:
            logging.info("Starting MQTT Drilling Data Analyser")
            self.mqtt_manager.connect_all()
            
            # Keep the main thread running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received, shutting down")
        except Exception as e:
            logging.critical(f"Critical error in run method: {str(e)}", exc_info=True)
        finally:
            self.mqtt_manager.disconnect_all()
            logging.info("MQTT Drilling Data Analyser stopped")

# Example usage
if __name__ == "__main__":
    # Configure logging
    from abyss.uos_depth_est_utils import setup_logging
    setup_logging(logging.INFO)
    
    # Create and run the analyser
    analyser = MQTTDrillingDataAnalyser()
    analyser.run()