import logging
import paho.mqtt.client as mqtt
import yaml
import json
import time
import threading
from typing import Dict, List
from functools import reduce

from dataclasses import dataclass
from typing import Dict, List, TypeVar, Generic
from datetime import datetime, timezone
from collections import defaultdict


# Custom exception classes
class MessageProcessingError(Exception):
    """Raised when message processing fails"""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass

class TimestampError(Exception):
    """Raised when timestamp parsing fails"""
    pass


from abyss.uos_depth_est_core import (
    DepthInference
)

from abyss.uos_depth_est_utils import (
    convert_mqtt_to_df,
)


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

T = TypeVar('T')  # Generic type for the data

# @dataclass(frozen=True)  # Make the class immutable and hashable
@dataclass(frozen=False)  
class TimestampedData(Generic[T]):
    _timestamp: float  # Unix timestamp
    _data: T
    _source: str
    
    @property
    def timestamp(self) -> float:
        return self._timestamp
    
    @property
    def data(self) -> T:
        return self._data
    
    @property
    def source(self) -> str:
        return self._source
    processed: bool = False  # Flag to indicate if the message has been processed
    
    def __hash__(self):
        return hash((self.timestamp, self.source))
    
    def __eq__(self, other):
        if not isinstance(other, TimestampedData):
            return False
        return self.timestamp == other.timestamp and self.source == other.source

class MQTTDrillingDataAnalyser:
    def __init__(self, config_path='mqtt_conf.yaml'):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            logging.info(f"Loaded configuration from {config_path}")
        
        # MQTT connection parameters
        self.broker = self.config['mqtt']['broker']

        # Buffer to store messages from each source
        self.buffers: Dict[str, List[TimestampedData]] = defaultdict(list)

        # Stores topics for each source
        self.topics = []
        
        # Time window for matching messages (in seconds)
        self.time_window = 0.1
        
        # Last cleanup timestamp
        self.last_cleanup = datetime.now().timestamp()
        
        # Cleanup interval (10 times the time window)
        self.cleanup_interval = 1000 * self.time_window

        # Load the depth inference model
        self.depth_inference = DepthInference()

        # Add thread for continuous processing
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self.continuous_processing)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.ALGO_VERSION = "0.2.2"
        self.MACHINE_ID = "MACHINE_ID"
        self.RESULT_ID = "RESULT_ID"

        logging.debug("DrillingDataAnalyser initialized")

    def create_mqtt_client(self, client_id):
        """Create MQTT client with basic configuration"""
        client = mqtt.Client(client_id)
        
        if self.broker.get('username') and self.broker.get('password'):
            client.username_pw_set(
                self.broker['username'], 
                self.broker['password']
            )
            logging.debug(f"Configured MQTT client with username authentication")
        
        return client

    def create_listener(self, listener_type: str, client_id: str):
        """Create a generic MQTT client for listening to specified data type"""
        
        # Validate listener type
        valid_types = ['result', 'trace', 'heads']
        if listener_type not in valid_types:
            raise ValueError(f"Invalid listener type: {listener_type}. Must be one of {valid_types}")
        
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logging.info(f"{listener_type.title()} Listener Connected with result code %s", rc)
                topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener'][listener_type]}"
                client.subscribe(topic)
                self.topics.append(topic)
                logging.info(f"Subscribed to {topic}")
            else:
                logging.error(f"Failed to connect with result code {rc}")

        def on_message(client, userdata, msg):
            try:
                data = json.loads(msg.payload)
                logging.debug(f"{listener_type.title()} Message on %s", msg.topic)
                st = find_in_dict(data, 'SourceTimestamp')
                logging.debug("Source Timestamp: %s", st[0])
                dt = datetime.strptime(st[0], "%Y-%m-%dT%H:%M:%SZ")
                unix_timestamp = dt.timestamp()
                
                tsdata = TimestampedData(
                    _timestamp=unix_timestamp,
                    _data=data,
                    _source=msg.topic
                )
                self.add_message(tsdata)
                logging.debug("Unix Timestamp: %s", unix_timestamp)
                logging.debug("ToolboxId: %s", msg.topic.split('/')[1])
                logging.debug("ToolId: %s", msg.topic.split('/')[2])
            except json.JSONDecodeError as e:
                logging.error("JSON decode error", extra={
                    'topic': msg.topic,
                    'payload_size': len(msg.payload),
                    'error_position': getattr(e, 'pos', None),
                    'error_message': str(e)
                })
            except Exception as e:
                logging.error("Message processing error", extra={
                    'topic': msg.topic,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }, exc_info=True)

        client = mqtt.Client(client_id=client_id)
        client.on_connect = on_connect
        client.on_message = on_message
        
        # Store client reference based on type
        if listener_type == 'result':
            self.result_client = client
        elif listener_type == 'trace':
            self.trace_client = client
        
        return client

    def create_result_listener(self):
        """Create an MQTT client for listening to Result data"""
        return self.create_listener('result', 'result_listener')

    def create_trace_listener(self):
        """Create an MQTT client for listening to Trace data"""
        return self.create_listener('trace', 'trace_listener')

    def add_message(self, data: TimestampedData):
        """Add a message to the buffer with improved error handling"""
        try:
            # Validation
            if not data.source:
                raise ValueError("Message source cannot be empty")
            
            matching_topic = None
            if data.source and self.config['mqtt']['listener']['trace'] in data.source:
                matching_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['trace']}"
                logging.debug("Trace message received")
            elif data.source and self.config['mqtt']['listener']['result'] in data.source:
                matching_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['result']}"
                logging.debug("Result message received")
            elif data.source and self.config['mqtt']['listener']['heads'] in data.source:
                matching_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['heads']}"
                logging.debug("Heads message received")
            else:
                logging.debug("Non-essential topic received: %s", data.source)
                return

            logging.info("Adding %s message to buffer - Source: %s", matching_topic, data.source)
            logging.info("Buffer size before: %s", len(self.buffers[matching_topic]))
            
            self.buffers[matching_topic].append(data)
            
            logging.info("Buffer size after: %s", len(self.buffers[matching_topic]))
            logging.info("All buffers: %s", {k: len(v) for k,v in self.buffers.items()})
            
            # Check if any buffer is exceeding the maximum size
            for topic, buffer in self.buffers.items():
                if len(buffer) >= 10000:
                    self.cleanup_old_messages()
                    break
            
            current_time = datetime.now().timestamp()
            if current_time - self.last_cleanup >= self.cleanup_interval:
                self.cleanup_old_messages()
                self.last_cleanup = current_time

        except ValueError as e:
            logging.warning("Invalid message data: %s", str(e))
            return
        except ConfigurationError as e:
            logging.error("Configuration error in add_message: %s", str(e))
            return
        except Exception as e:
            logging.error("Unexpected error in add_message: %s", str(e))
            return

    def find_and_process_matches(self):
        """
        Find and process messages with matching timestamps and tool IDs.
        Optimized to group by timestamp first, then match by tool key.
        """
        try:
            logging.debug("Checking for matches")
            
            result_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['result']}"
            trace_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['trace']}"
            heads_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['heads']}"
            
            result_messages = [msg for msg in self.buffers.get(result_topic, []) if not msg.processed]
            trace_messages = [msg for msg in self.buffers.get(trace_topic, []) if not msg.processed]
            heads_messages = [msg for msg in self.buffers.get(heads_topic, []) if not msg.processed]
            
            logging.debug("Result messages: %s", len(result_messages))
            logging.debug("Trace messages: %s", len(trace_messages))
            logging.debug("Heads messages: %s", len(heads_messages))
            
            # Group messages by timestamp buckets (rounded to time_window precision)
            def get_time_bucket(timestamp):
                return round(timestamp / self.time_window) * self.time_window
            
            # Group result and trace messages by timestamp buckets
            result_by_time = defaultdict(list)
            trace_by_time = defaultdict(list)
            heads_by_time = defaultdict(list)
            
            for msg in result_messages:
                bucket = get_time_bucket(msg.timestamp)
                result_by_time[bucket].append(msg)
            
            for msg in trace_messages:
                bucket = get_time_bucket(msg.timestamp)
                trace_by_time[bucket].append(msg)
                
            for msg in heads_messages:
                bucket = get_time_bucket(msg.timestamp)
                heads_by_time[bucket].append(msg)
            
            matches_found = False
            processed_messages = set()
            
            # Find matches within each time bucket and adjacent buckets
            for time_bucket in result_by_time.keys():
                # Check current bucket and adjacent buckets (±1)
                for bucket_offset in [-1, 0, 1]:
                    check_bucket = time_bucket + (bucket_offset * self.time_window)
                    
                    if check_bucket in trace_by_time:
                        matches_found |= self._process_time_bucket_matches(
                            result_by_time[time_bucket],
                            trace_by_time[check_bucket],
                            heads_by_time.get(check_bucket, []),
                            processed_messages
                        )
            
            # Remove processed messages from buffers
            self._remove_processed_messages(processed_messages)
            
            return matches_found
            
        except ValueError as e:
            logging.error("Value error in message matching", extra={
                'error_message': str(e),
                'buffer_sizes': {k: len(v) for k, v in self.buffers.items()},
                'time_window': getattr(self, 'time_window', 'unknown')
            })
            return False
        except KeyError as e:
            logging.error("Configuration key missing", extra={
                'missing_key': str(e),
                'config_section': 'mqtt.listener',
                'available_keys': list(self.config.get('mqtt', {}).get('listener', {}).keys()) if hasattr(self, 'config') else []
            })
            return False
        except Exception as e:
            logging.error("Unexpected error in message matching", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'buffer_sizes': {k: len(v) for k, v in self.buffers.items()}
            }, exc_info=True)
            return False

    def _process_time_bucket_matches(self, result_msgs, trace_msgs, heads_msgs, processed_messages):
        """Process matches within a specific time bucket"""
        matches_found = False
        
        # Group by tool key within the time bucket
        result_by_tool = defaultdict(list)
        trace_by_tool = defaultdict(list)
        
        for msg in result_msgs:
            if msg not in processed_messages:
                parts = msg.source.split('/')
                tool_key = f"{parts[1]}/{parts[2]}"
                result_by_tool[tool_key].append(msg)
        
        for msg in trace_msgs:
            if msg not in processed_messages:
                parts = msg.source.split('/')
                tool_key = f"{parts[1]}/{parts[2]}"
                trace_by_tool[tool_key].append(msg)
        
        # Match result and trace messages by tool key
        for tool_key in result_by_tool.keys():
            if tool_key in trace_by_tool:
                for result_msg in result_by_tool[tool_key]:
                    for trace_msg in trace_by_tool[tool_key]:
                        if (abs(result_msg.timestamp - trace_msg.timestamp) <= self.time_window and
                            result_msg not in processed_messages and 
                            trace_msg not in processed_messages):
                            
                            # Find matching heads message by timestamp only
                            matching_heads = None
                            for heads_msg in heads_msgs:
                                if (abs(result_msg.timestamp - heads_msg.timestamp) <= self.time_window and
                                    heads_msg not in processed_messages):
                                    matching_heads = heads_msg
                                    break
                            
                            logging.info("Found matching pair for tool: %s", tool_key)
                            matches_found = True
                            
                            # Process the match
                            match_group = [result_msg, trace_msg]
                            if matching_heads:
                                match_group.append(matching_heads)
                                processed_messages.add(matching_heads)
                            
                            self.process_matching_messages(match_group)
                            
                            processed_messages.add(result_msg)
                            processed_messages.add(trace_msg)
                            
                            break  # Only match each result message once
        
        return matches_found

    def _remove_processed_messages(self, processed_messages):
        """Immediately remove processed messages from buffers"""
        try:
            result_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['result']}"
            trace_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['trace']}"
            heads_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['heads']}"
            
            # Log details about messages being removed as processed
            if processed_messages:
                logging.info("Marking messages as processed and removing from buffers", extra={
                    'processed_count': len(processed_messages),
                    'removal_method': 'immediate_removal'
                })
                
                for msg in processed_messages:
                    # Extract tool key for identification
                    tool_key = "unknown"
                    try:
                        parts = msg.source.split('/')
                        if len(parts) >= 3:
                            tool_key = f"{parts[1]}/{parts[2]}"
                    except Exception:
                        pass
                    
                    # Determine message type
                    message_type = "unknown"
                    if "ResultManagement" in msg.source:
                        message_type = "result"
                    elif "Trace" in msg.source:
                        message_type = "trace"
                    elif "AssetManagement" in msg.source:
                        message_type = "heads"
                    
                    logging.debug("Processing and removing message", extra={
                        'message_type': message_type,
                        'source_topic': msg.source,
                        'tool_key': tool_key,
                        'timestamp': msg.timestamp,
                        'processing_status': 'successfully_processed'
                    })
            
            # Remove processed messages immediately
            self.buffers[result_topic] = [msg for msg in self.buffers[result_topic] if msg not in processed_messages]
            self.buffers[trace_topic] = [msg for msg in self.buffers[trace_topic] if msg not in processed_messages]
            self.buffers[heads_topic] = [msg for msg in self.buffers[heads_topic] if msg not in processed_messages]
            
            logging.debug("Removed %s processed messages from buffers", len(processed_messages))
            
        except Exception as e:
            logging.error("Error removing processed messages: %s", str(e))

    def _convert_messages_to_df(self, result_msg, trace_msg, heads_msg=None):
        """Convert MQTT messages to DataFrame, including heads data if available"""
        try:
            # Base conversion
            df = convert_mqtt_to_df(
                json.dumps(result_msg.data), 
                json.dumps(trace_msg.data), 
                conf=self.config
            )
            
            # Enhance with heads data if available
            if heads_msg and heads_msg.data and df is not None:
                heads_data = self._extract_heads_data(heads_msg.data)
                if heads_data:
                    # Add heads data as additional columns
                    for key, value in heads_data.items():
                        df[f'heads_{key}'] = value
                    logging.debug("Enhanced DataFrame with heads data: %s", list(heads_data.keys()))
            
            return df
            
        except Exception as e:
            logging.error("Error converting messages to DataFrame: %s", str(e))
            raise MessageProcessingError(f"DataFrame conversion failed: {e}")

    def _extract_heads_data(self, heads_data):
        """Extract relevant data from heads message"""
        try:
            # Extract specific fields from heads message based on your requirements
            extracted = {}
            
            # Example extractions (adjust based on your heads message structure)
            if 'Payload' in heads_data:
                payload = heads_data['Payload']
                # Extract specific fields you need from heads payload
                for key, value_obj in payload.items():
                    if isinstance(value_obj, dict) and 'Value' in value_obj:
                        # Clean up the key name for column naming
                        clean_key = key.split('.')[-1] if '.' in key else key
                        extracted[clean_key] = value_obj['Value']
            
            return extracted
            
        except Exception as e:
            logging.warning("Failed to extract heads data: %s", str(e))
            return {} 

    def cleanup_old_messages(self):
        """
        Improved cleanup with time-based sliding window approach
        """
        try:
            current_time = datetime.now().timestamp()
            MAX_BUFFER_SIZE = 10000
            MAX_AGE_SECONDS = 300  # 5 minutes
            
            removed_count = 0
            
            for topic in list(self.buffers.keys()):  # Use list() to avoid dict changing during iteration
                if topic not in self.buffers:
                    continue
                    
                original_length = len(self.buffers[topic])
                
                # Remove old messages beyond time window
                self.buffers[topic] = [
                    msg for msg in self.buffers[topic] 
                    if current_time - msg.timestamp <= MAX_AGE_SECONDS
                ]
                age_removed = original_length - len(self.buffers[topic])
                
                # If still over size limit, keep only newest messages
                if len(self.buffers[topic]) > MAX_BUFFER_SIZE:
                    self.buffers[topic].sort(key=lambda msg: msg.timestamp, reverse=True)
                    excess = len(self.buffers[topic]) - MAX_BUFFER_SIZE
                    self.buffers[topic] = self.buffers[topic][:MAX_BUFFER_SIZE]
                    removed_count += age_removed + excess
                    logging.info("Buffer %s: removed %s old messages, %s excess messages", 
                               topic, age_removed, excess)
                else:
                    removed_count += age_removed
                    logging.debug("Buffer %s: removed %s old messages", topic, age_removed)
            
            logging.debug("Total cleanup removed: %s messages", removed_count)
            
        except Exception as e:
            logging.error("Error in cleanup_old_messages: %s", str(e))

    def process_matching_messages(self, matches: List[TimestampedData]):
        """Process messages that have matching timestamps"""
        df = None
        try:
            # Separate message types
            result_msg = next((m for m in matches if self.config['mqtt']['listener']['result'] in m.source), None)
            trace_msg = next((m for m in matches if self.config['mqtt']['listener']['trace'] in m.source), None)
            heads_msg = next((m for m in matches if self.config['mqtt']['listener']['heads'] in m.source), None)
            
            if not result_msg or not trace_msg:
                logging.warning("Missing required result or trace message")
                return
            
            # Extract tool information
            parts = result_msg.source.split('/')
            toolbox_id = parts[1]
            tool_id = parts[2]
            
            logging.info("Processing matched messages:")
            logging.info("Toolbox ID: %s, Tool ID: %s", toolbox_id, tool_id)
            logging.info("Timestamp: %s", datetime.fromtimestamp(result_msg.timestamp))
            logging.info("Time difference: %.3f seconds", abs(result_msg.timestamp - trace_msg.timestamp))
            if heads_msg:
                logging.info("Heads message included in processing")

            # Enhanced data conversion with heads data
            df = self._convert_messages_to_df(result_msg, trace_msg, heads_msg)
            
            assert df is not None, "Dataframe is None"
            assert not df.empty, "Dataframe is empty"
            logging.info("Dataframe:\n%s", df.head())

            self.MACHINE_ID = str(df.iloc[0]['HOLE_ID'])
            self.RESULT_ID = str(df.iloc[0]['local'])
            logging.info(f"Machine ID: {self.MACHINE_ID}")
            logging.info(f"Result ID: {self.RESULT_ID}")

            # Debug file output
            if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                fn = f"{result_msg.timestamp}.txt"
                with open(fn, 'w') as file:
                    file.write(f"Toolbox ID: {toolbox_id}\n")
                    file.write(f"Tool ID: {tool_id}\n")
                    file.write(f"Timestamp: {datetime.fromtimestamp(result_msg.timestamp)}\n\n")
                    file.write(f"RESULT:\n\n")
                    file.write(str(result_msg.data))
                    file.write(f"\n\nTRACE:\n\n")
                    file.write(str(trace_msg.data))
                    if heads_msg:
                        file.write(f"\n\nHEADS:\n\n")
                        file.write(str(heads_msg.data))
                logging.info(f"Stored matched data into {fn}")
                csvfn = f"{result_msg.timestamp}.csv"
                df.to_csv(csvfn, index=False)
                logging.info(f"Stored matched dataframe into {csvfn}")
            
        except MessageProcessingError as e:
            logging.error("Message processing error: %s", str(e))
            return
        except Exception as e:
            logging.critical("Unexpected error in process_matching_messages: %s", str(e))
            return

        # prepare data for publishing
        # dt = datetime.strptime(result_msg.timestamp, "%Y-%m-%dT%H:%M:%SZ")
        dt_utc = datetime.fromtimestamp(result_msg.timestamp)
        dt = datetime.strftime(dt_utc, "%Y-%m-%dT%H:%M:%SZ")

        # OUTPUT: Depth estimation
        # insufficient steps to estimate keypoints
        # TODO: replace magic numbers with a better system
        if df is None or 'Step (nb)' not in df.columns or len(df['Step (nb)'].unique()) < 2:
            keyp_topic = f"{self.config['mqtt']['listener']['root']}/{toolbox_id}/{tool_id}/{self.config['mqtt']['estimation']['keypoints']}"
            keyp_data = dict(Value = 'Not enough steps to estimate keypoints', SourceTimestamp = dt)
            self.result_client.publish(keyp_topic, json.dumps(keyp_data))
            dest_topic = f"{self.config['mqtt']['listener']['root']}/{toolbox_id}/{tool_id}/{self.config['mqtt']['estimation']['depth_estimation']}"
            dest_data = dict(Value = 'Not enough steps to estimate depth', SourceTimestamp = dt)
            self.result_client.publish(dest_topic, json.dumps(dest_data))
            logging.error("Not enough steps to estimate depth")
            return

        # more than one step, perform keypoint identification
        else:
            # Placeholders
            l_result = [0]
            depth_estimation = 0
            try:
                # Perform keypoint identification -- this calls the deep learning model
                # l_result = self.depth_inference.infer_common(df) # two-point result
                l_result = self.depth_inference.infer3_common(df) # two-point result                 
                # Perform depth estimation
                # depth_estimation = l_result[1]-l_result[0]
                depth_estimation = [l_result[i+1] - l_result[i] for i in range(len(l_result)-1)]
            except Exception as e:
                logging.error("Error in depth estimation: %s", str(e))
            
            logging.info(f"Keypoints: {l_result}")
            logging.info(f"Depth estimation: {depth_estimation}")
            # Prepare the topics for publishing
            keyp_topic = f"{self.config['mqtt']['listener']['root']}/{toolbox_id}/{tool_id}/{self.config['mqtt']['estimation']['keypoints']}"
            dest_topic = f"{self.config['mqtt']['listener']['root']}/{toolbox_id}/{tool_id}/{self.config['mqtt']['estimation']['depth_estimation']}"
            # Publish the results as in the configuration file              
            keyp_data = dict(Value = l_result, SourceTimestamp = dt, MachineId = self.MACHINE_ID, ResultId = self.RESULT_ID, AlgoVersion = self.ALGO_VERSION)
            self.result_client.publish(keyp_topic, json.dumps(keyp_data))
            dest_data = dict(Value = depth_estimation, SourceTimestamp = dt, MachineId = self.MACHINE_ID, ResultId = self.RESULT_ID, AlgoVersion = self.ALGO_VERSION)
            self.result_client.publish(dest_topic, json.dumps(dest_data))            
            
    def continuous_processing(self):
        """Continuously check for and process matches in a separate thread"""
        while self.processing_active:
            matches_found = self.find_and_process_matches()
            if not matches_found:
                # Sleep if no matches found to avoid CPU spinning
                time.sleep(0.1)    

    def run(self):
        """Main method to set up MQTT clients and start listening"""
        result_client = self.create_listener('result', 'result_listener')
        trace_client = self.create_listener('trace', 'trace_listener')
        heads_client = self.create_listener('heads', 'heads_listener')

        result_client.connect(self.config['mqtt']['broker']['host'], 
                            self.config['mqtt']['broker']['port'])
        trace_client.connect(self.config['mqtt']['broker']['host'], 
                           self.config['mqtt']['broker']['port'])
        heads_client.connect(self.config['mqtt']['broker']['host'],
                           self.config['mqtt']['broker']['port'])
        
        result_client.loop_start()
        trace_client.loop_start()
        heads_client.loop_start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Stopping MQTT clients...")
            result_client.loop_stop()
            trace_client.loop_stop()
            heads_client.loop_stop()
            logging.info("Disconnecting MQTT clients...")
            result_client.disconnect()
            trace_client.disconnect()
            heads_client.disconnect()
            self.processing_active = False
            self.processing_thread.join(timeout=1.0)