import logging
import sys
import os
import paho.mqtt.client as mqtt
import yaml
import json
import time
import numpy as np
from typing import Dict, List
import pandas as pd
import argparse
from functools import reduce

from dataclasses import dataclass
from typing import Dict, List, TypeVar, Generic
from datetime import datetime, timezone
from collections import defaultdict

from abyss.uos_depth_est_core import (
    depth_est_ml,
    convert_mqtt_to_df,
    depth_est_ml_mqtt,
    DepthInference
)

def convert_mqtt_to_df(result_msg=None, trace_msg=None, conf=None):
    """
    Convert `RESULT` and/or `TRACE` MQTT messages to Pandas DataFrames and combine them.

    Parameters:
    result_msg (str, optional): The MQTT message for the `RESULT` structure as a JSON string.
    trace_msg (str, optional): The MQTT message for the `TRACE` structure as a JSON string.
    conf (dict, optional): Configuration dictionary with key paths to extract the required data. If not provided, default paths will be used.

    Returns:
    pd.DataFrame: A combined DataFrame with data from `RESULT` and `TRACE` messages, merged on step values.

    Notes:
    - If both `result_msg` and `trace_msg` are provided, the data is merged using step values as a key.
    - The paths in the configuration should be provided as tuples representing the hierarchical keys.
    """
    
    def reduce_dict(data_dict, search_key):
        # Using reduce function to filter dictionary values based on search_key
        # from https://www.geeksforgeeks.org/python-substring-key-match-in-dictionary/
        values = reduce(
            # lambda function that takes in two arguments, an accumulator list and a key
            lambda acc, key: acc + [data_dict[key]] if search_key in key else acc,
            # list of keys from the test_dict
            data_dict.keys(),
            # initial value for the accumulator (an empty list)
            []
        )
        return values[0]['Value']

    def parse_result(data, conf):
        # Default key paths for RESULT structure
        fore = ['Messages', 'Payload']
        
        data = data[fore[0]][fore[1]]   

        torque_empty_vals = reduce_dict(data, conf['torque_empty_vals'])
        logging.debug(f"Torque empty values: {torque_empty_vals}")
        thrust_empty_vals = reduce_dict(data, conf['thrust_empty_vals'])
        logging.debug(f"Thrust empty values: {thrust_empty_vals}")
        step_vals = reduce_dict(data, conf['step_vals'])
        logging.debug(f"Step values: {step_vals}")
        hole_id = reduce_dict(data, conf['machine_id'])
        logging.debug(f"Hole ID: {hole_id}")

        # Create a DataFrame
        try:
            df = pd.DataFrame({
                'Step (nb)': step_vals,
                'I Torque Empty (A)': torque_empty_vals,
                'I Thrust Empty (A)': thrust_empty_vals,
                'HOLE_ID': [str(hole_id)] * len(step_vals),
                'PREDRILLED': [1] * len(step_vals)
            })
        except:
            df = pd.DataFrame({
                'Step (nb)': step_vals,
                'I Torque Empty (A)': torque_empty_vals,
                'I Thrust Empty (A)': thrust_empty_vals,
                'HOLE_ID': [str(hole_id)],
                'PREDRILLED': [1]
            })

        return df

    def parse_trace(data, conf):
        # Default key paths for RESULT structure
        fore = ['Messages', 'Payload']

        data = data[fore[0]][fore[1]]   
        
        # def get_nested_value(data, key):
        #     key_path = fore + [key]
        #     return reduce(dict.get, key_path, data)

        position = reduce_dict(data, conf['position'])
        logging.debug(f"Position: {position}")
        torque = reduce_dict(data, conf['torque'])
        logging.debug(f"Torque: {torque}")
        thrust = reduce_dict(data, conf['thrust'])
        logging.debug(f"Thrust: {thrust}")
        step = reduce_dict(data, conf['step'])
        logging.debug(f"Step: {step}")

        # Create a DataFrame
        df = pd.DataFrame({
            'Step (nb)': step,
            'Position (mm)': position,
            'I Torque (A)': torque,
            'I Thrust (A)': thrust,
        })

        # Process columns
        df['Step (nb)'] = df['Step (nb)'].astype('int32')
        df['Position (mm)'] = -df['Position (mm)']  # Invert position

        return df

    result_df = None
    trace_df = None

    if result_msg:
        result_data = json.loads(result_msg)
        try:
            result_df = parse_result(result_data, conf)
        except Exception as e:
            logging.critical("Error parsing RESULT data: %s", str(e))

    if trace_msg:
        trace_data = json.loads(trace_msg)
        try:
            trace_df = parse_trace(trace_data, conf)
        except Exception as e:
            logging.critical("Error parsing TRACE data: %s", str(e))

    # Merge the DataFrames on the step values if both are present
    if result_df is not None and trace_df is not None:
        try:
            combined_df = pd.merge(result_df, trace_df, on='Step (nb)', how='outer')
            logging.debug(str(combined_df.dtypes))
            return combined_df
        except Exception as e:
            logging.critical("Error merging RESULT and TRACE DataFrames: %s", str(e))

    return result_df or trace_df


def setup_logging(level):
    """
    Configure logging with the specified level.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
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

class DrillingDataAnalyser:
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
            logging.debug("Checking for matches")
            
            result_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['result']}"
            trace_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['trace']}"
            
            result_messages = self.buffers.get(result_topic, [])
            trace_messages = self.buffers.get(trace_topic, [])
            
            logging.debug("Result messages: %s", len(result_messages))
            logging.debug("Trace messages: %s", len(trace_messages))
            
            to_remove_result = []
            to_remove_trace = []
            
            for result_msg in result_messages:
                r_parts = result_msg.source.split('/')
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


            df = convert_mqtt_to_df(json.dumps(result_msg.data), json.dumps(trace_msg.data), conf=self.config['mqtt']['data_ids'])
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

def main():
    """
    Main entry point for the application.
    
    Command line arguments:
        --config: Path to YAML configuration file (default: mqtt_conf.yaml)
        --log-level: Logging level (default: INFO)
    
    Example usage:
        python -m drilling_analyzer
        python -m drilling_analyzer --config=custom_config.yaml --log-level=DEBUG
    """
    parser = argparse.ArgumentParser(description='MQTT Drilling Data Analyzer')
    parser.add_argument(
        '--config', 
        type=str,
        default='mqtt_conf.yaml',
        help='Path to YAML configuration file (default: mqtt_conf.yaml)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    setup_logging(getattr(logging, args.log_level))
    
    try:
        analyzer = DrillingDataAnalyser(config_path=args.config)
        analyzer.run()
    except FileNotFoundError:
        logging.critical("Configuration file '%s' not found.", args.config, os.getcwd())
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.critical("Invalid YAML configuration in '%s': %s", args.config, str(e))
        sys.exit(1)
    except Exception as e:
        logging.critical("Error: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
