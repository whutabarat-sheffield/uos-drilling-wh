import logging
import paho.mqtt.client as mqtt
import yaml
import json
import time
from typing import Dict, List
from functools import reduce

from dataclasses import dataclass
from typing import Dict, List, TypeVar, Generic
from datetime import datetime, timezone
from collections import defaultdict


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

@dataclass(frozen=True)  # Make the class immutable and hashable
class TimestampedData(Generic[T]):
    timestamp: float  # Unix timestamp
    data: T
    source: str
    
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
        self.time_window = 1.0
        
        # Last cleanup timestamp
        self.last_cleanup = datetime.now().timestamp()
        
        # Cleanup interval (10 times the time window)
        self.cleanup_interval = 10 * self.time_window

        # Load the depth inference model
        self.depth_inference = DepthInference() 

        logging.debug("DrillingDataAnalyser initialized")

    def create_mqtt_client(self, client_id):
        """Create MQTT client with basic configuration"""
        client = mqtt.Client(client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
        
        if self.broker.get('username') and self.broker.get('password'):
            client.username_pw_set(
                self.broker['username'], 
                self.broker['password']
            )
            logging.debug(f"Configured MQTT client with username authentication")
        
        return client

    def create_result_listener(self):
        """Create an MQTT client for listening to Result data"""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logging.info("Result Listener Connected with result code %s", rc)
                topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['result']}"
                client.subscribe(topic)
                self.topics.append(topic)
                logging.info(f"Subscribed to {topic}")
            else:
                logging.error(f"Failed to connect with result code {rc}")

        def on_message(client, userdata, msg):
            try:
                data = json.loads(msg.payload)
                logging.debug("Result Message on %s", msg.topic)
                st = find_in_dict(data, 'SourceTimestamp')
                logging.debug("Source Timestamp: %s", st[0])
                dt = datetime.strptime(st[0], "%Y-%m-%dT%H:%M:%SZ")
                unix_timestamp = dt.timestamp()
                
                tsdata = TimestampedData(
                    timestamp=unix_timestamp,
                    data=data,
                    source=msg.topic
                )
                self.add_message(tsdata)
                logging.debug("Unix Timestamp: %s", unix_timestamp)
                logging.debug("ToolboxId: %s", msg.topic.split('/')[1])
                logging.debug("ToolId: %s", msg.topic.split('/')[2])
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {msg.topic}")
            except Exception as e:
                logging.error(f"Error processing message: {str(e)}")

        client = mqtt.Client("result_listener")
        client.on_connect = on_connect
        client.on_message = on_message
        self.result_client = client
        return client
    
    def create_trace_listener(self):
        """Create an MQTT client for listening to Trace data"""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logging.info("Trace Listener Connected with result code %s", rc)
                topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['trace']}"
                client.subscribe(topic)
                self.topics.append(topic)
                logging.info(f"Subscribed to {topic}")
            else:
                logging.error(f"Failed to connect with result code {rc}")

        def on_message(client, userdata, msg):
            try:
                data = json.loads(msg.payload)
                logging.debug("Trace Message on %s", msg.topic)
                st = find_in_dict(data, 'SourceTimestamp')
                logging.debug("Source Timestamp: %s", st[0])
                dt = datetime.strptime(st[0], "%Y-%m-%dT%H:%M:%SZ")
                unix_timestamp = dt.timestamp()
                
                tsdata = TimestampedData(
                    timestamp=unix_timestamp,
                    data=data,
                    source=msg.topic
                )
                self.add_message(tsdata)
                logging.debug("Unix Timestamp: %s", unix_timestamp)
                logging.debug("ToolboxId: %s", msg.topic.split('/')[1])
                logging.debug("ToolId: %s", msg.topic.split('/')[2])
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {msg.topic}")
            except Exception as e:
                logging.error(f"Error processing message: {str(e)}")

        client = mqtt.Client("trace_listener")
        client.on_connect = on_connect
        client.on_message = on_message

        self.trace_client = client
        return client

    def add_message(self, data: TimestampedData):
        """Add a message to the buffer and check for matches"""
        try:
            matching_topic = None
            if self.config['mqtt']['listener']['trace'] in data.source:
                matching_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['trace']}"
            else:
                matching_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['result']}"
            
            logging.debug("Adding message to buffer")
            logging.debug("Source: %s", data.source)
            logging.debug("Matching topic: %s", matching_topic)
            logging.debug("Buffer size before: %s", len(self.buffers[matching_topic]))
            
            self.buffers[matching_topic].append(data)
            
            logging.debug("Buffer size after: %s", len(self.buffers[matching_topic]))
            logging.debug("All buffers: %s", {k: len(v) for k,v in self.buffers.items()})
            
            self.find_and_process_matches()
            
            current_time = datetime.now().timestamp()
            if current_time - self.last_cleanup >= self.cleanup_interval:
                self.cleanup_old_messages()
                self.last_cleanup = current_time

        except Exception as e:
            logging.error("Error in add_message: %s", str(e))

    def find_and_process_matches(self):
        """Find and process messages with matching timestamps and tool IDs"""
        try:
            logging.info("Checking for matches")
            
            result_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['result']}"
            trace_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['trace']}"
            
            result_messages = self.buffers.get(result_topic, [])
            trace_messages = self.buffers.get(trace_topic, [])
            
            logging.info("Result messages: %s", len(result_messages))
            logging.info("Trace messages: %s", len(trace_messages))
            
            to_remove_result = []
            to_remove_trace = []
            
            for result_msg in result_messages:
                r_parts = result_msg.source.split('/')
                logging.info("r_parts: %s", r_parts)
                r_tool_key = f"{r_parts[1]}/{r_parts[2]}"
                
                for trace_msg in trace_messages:
                    t_parts = trace_msg.source.split('/')
                    t_tool_key = f"{t_parts[1]}/{t_parts[2]}"
                    
                    if (r_tool_key == t_tool_key and 
                        abs(result_msg.timestamp - trace_msg.timestamp) <= self.time_window and
                        result_msg not in to_remove_result and 
                        trace_msg not in to_remove_trace):
                        
                        logging.info("Found matching pair!")
                        logging.info("Tool: %s", r_tool_key)
                        logging.info("Timestamp: %s", datetime.fromtimestamp(result_msg.timestamp))
                        
                        self.process_matching_messages([result_msg, trace_msg])
                        
                        to_remove_result.append(result_msg)
                        to_remove_trace.append(trace_msg)
            
            self.buffers[result_topic] = [msg for msg in result_messages if msg not in to_remove_result]
            self.buffers[trace_topic] = [msg for msg in trace_messages if msg not in to_remove_trace]
            
        except Exception as e:
            logging.error("Error in find_and_process_matches: %s", str(e))

    def cleanup_old_messages(self):
        """Remove messages older than the cleanup interval"""
        try:
            logging.debug("Cleaning up old messages")
            current_time = datetime.now().timestamp()
            removed_count = 0
            
            for topic in self.topics:
                original_length = len(self.buffers[topic])
                self.buffers[topic] = [
                    msg for msg in self.buffers[topic]
                    if current_time - msg.timestamp <= self.cleanup_interval
                ]
                removed_count += original_length - len(self.buffers[topic])
            
            logging.info("Removed %s old messages", removed_count)
            
        except Exception as e:
            logging.error("Error in cleanup_old_messages: %s", str(e))

    def process_matching_messages(self, matches: List[TimestampedData]):
        """Process messages that have matching timestamps"""
        try:
            result_msg = next(m for m in matches if self.config['mqtt']['listener']['result'] in m.source)
            trace_msg = next(m for m in matches if self.config['mqtt']['listener']['trace'] in m.source)
            
            parts = result_msg.source.split('/')
            toolbox_id = parts[1]
            tool_id = parts[2]
            
            logging.info("Processing matched pair:")
            logging.info("Toolbox ID: %s", toolbox_id)
            logging.info("Tool ID: %s", tool_id)
            logging.info("Timestamp: %s", datetime.fromtimestamp(result_msg.timestamp))
            logging.info("Time difference: %.3f seconds", abs(result_msg.timestamp - trace_msg.timestamp))

            # Here we can add our specific processing logic for the matched messages
            # Example: depth_est_ml_mqtt(result_msg.data, trace_msg.data)


            # df = convert_mqtt_to_df(json.dumps(result_msg.data), json.dumps(trace_msg.data), conf=self.config['mqtt']['data_ids'])
            df = convert_mqtt_to_df(json.dumps(result_msg.data), json.dumps(trace_msg.data), conf=self.config)
            logging.info("Dataframe:\n%s", df.head())
            assert df is not None, "Dataframe is None"
            assert not df.empty, "Dataframe is empty"
            print(f"debug level: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")
            if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
                fn = f"{result_msg.timestamp}.txt"
                with open(fn, 'w') as file:
                    file.write(f"Toolbox ID: {toolbox_id}\n")
                    file.write(f"Tool ID: {tool_id}\n")
                    file.write(f"Timestamp: {datetime.fromtimestamp(result_msg.timestamp)}\n\n")
                    file.write(f"RESULT:\n\n")
                    file.write(str(result_msg.data))
                    file.write(f"\n\nTRACE:\n\n")
                    file.write(str(trace_msg.data))
                logging.info(f"Stored matched data into {fn}")
                csvfn = f"{result_msg.timestamp}.csv"
                df.to_csv(csvfn, index=False)
                logging.info(f"Stored matched dataframe into {csvfn}")
            
        except Exception as e:
            logging.critical("Error in process_matching_messages: %s", str(e))

        # prepare data for publishing
        # dt = datetime.strptime(result_msg.timestamp, "%Y-%m-%dT%H:%M:%SZ")
        dt_utc = datetime.fromtimestamp(result_msg.timestamp)
        dt = datetime.strftime(dt_utc, "%Y-%m-%dT%H:%M:%SZ")

        # OUTPUT: Depth estimation
        # insufficient steps to estimate keypoints
        if len(df['Step (nb)'].unique()) < 2:
            
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
            try:
                # Perform keypoint identification
                l_result = self.depth_inference.infer_common(df)
                logging.info(f"Keypoints: {l_result}")
                # Perform depth estimation
                depth_estimation = l_result[1]-l_result[0]
                logging.info(f"Depth estimation: {depth_estimation}")
                # Publish the results somewhere as in the configuration file
                keyp_topic = f"{self.config['mqtt']['listener']['root']}/{toolbox_id}/{tool_id}/{self.config['mqtt']['estimation']['keypoints']}"
                keyp_data = dict(Value = l_result, SourceTimestamp = dt)
                self.result_client.publish(keyp_topic, json.dumps(keyp_data))
                dest_topic = f"{self.config['mqtt']['listener']['root']}/{toolbox_id}/{tool_id}/{self.config['mqtt']['estimation']['depth_estimation']}"
                dest_data = dict(Value = depth_estimation, SourceTimestamp = dt)
                self.result_client.publish(dest_topic, json.dumps(dest_data))
            except Exception as e:
                logging.error("Error in depth estimation: %s", str(e))
            
            
    def run(self):
        """Main method to set up MQTT clients and start listening"""
        result_client = self.create_result_listener()
        trace_client = self.create_trace_listener()

        result_client.connect(self.config['mqtt']['broker']['host'], 
                            self.config['mqtt']['broker']['port'])
        trace_client.connect(self.config['mqtt']['broker']['host'], 
                           self.config['mqtt']['broker']['port'])
        
        result_client.loop_start()
        trace_client.loop_start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Stopping MQTT clients...")
            result_client.loop_stop()
            trace_client.loop_stop()
            result_client.disconnect()
            trace_client.disconnect()