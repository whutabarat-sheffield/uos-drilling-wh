import paho.mqtt.client as mqtt
import yaml
import json
import time
import numpy as np
from typing import Dict, List
import pandas as pd
import argparse

from dataclasses import dataclass
from typing import Dict, List, TypeVar, Generic
from datetime import datetime
from collections import defaultdict

from abyss.uos_depth_est_core import (
    depth_est_ml,
    convert_mqtt_to_df,
    depth_est_ml_mqtt,
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

    def create_mqtt_client(self, client_id):
        """Create MQTT client with basic configuration"""
        client = mqtt.Client(client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
        
        if self.broker.get('username') and self.broker.get('password'):
            client.username_pw_set(
                self.broker['username'], 
                self.broker['password']
            )
        
        return client

    def create_result_listener(self):
        """Create an MQTT client for listening to Result data"""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Result Listener Connected with result code " + str(rc))
                topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['result']}"
                client.subscribe(topic)
                self.topics.append(topic)
                print(f"Subscribed to {topic}")
            else:
                print(f"Failed to connect with result code {rc}")

        def on_message(client, userdata, msg):
            try:
                data = json.loads(msg.payload)
                print(f"\nResult Message on {msg.topic}:")
                st = find_in_dict(data, 'SourceTimestamp')
                print(f"Source Timestamp: {st[0]}")
                dt = datetime.strptime(st[0], "%Y-%m-%dT%H:%M:%SZ")
                unix_timestamp = dt.timestamp()
                
                tsdata = TimestampedData(
                    timestamp=unix_timestamp,
                    data=data,
                    source=msg.topic
                )
                self.add_message(tsdata)
                print(f"Unix Timestamp: {unix_timestamp}")
                print(f"ToolboxId: {msg.topic.split('/')[1]}")
                print(f"ToolId: {msg.topic.split('/')[2]}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {msg.topic}")
            except Exception as e:
                print(f"Error processing message: {str(e)}")

        client = mqtt.Client("result_listener")
        client.on_connect = on_connect
        client.on_message = on_message
        return client
    
    def create_trace_listener(self):
        """Create an MQTT client for listening to Trace data"""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Trace Listener Connected with result code " + str(rc))
                topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['trace']}"
                client.subscribe(topic)
                self.topics.append(topic)
                print(f"Subscribed to {topic}")
            else:
                print(f"Failed to connect with result code {rc}")

        def on_message(client, userdata, msg):
            try:
                data = json.loads(msg.payload)
                print(f"\nTrace Message on {msg.topic}:")
                st = find_in_dict(data, 'SourceTimestamp')
                print(f"Source Timestamp: {st[0]}")
                dt = datetime.strptime(st[0], "%Y-%m-%dT%H:%M:%SZ")
                unix_timestamp = dt.timestamp()
                
                tsdata = TimestampedData(
                    timestamp=unix_timestamp,
                    data=data,
                    source=msg.topic
                )
                self.add_message(tsdata)
                print(f"Unix Timestamp: {unix_timestamp}")
                print(f"ToolboxId: {msg.topic.split('/')[1]}")
                print(f"ToolId: {msg.topic.split('/')[2]}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {msg.topic}")
            except Exception as e:
                print(f"Error processing message: {str(e)}")

        client = mqtt.Client("trace_listener")
        client.on_connect = on_connect
        client.on_message = on_message
        return client

    def add_message(self, data: TimestampedData):
        """Add a message to the buffer and check for matches"""
        try:
            # Determine correct buffer based on message type
            matching_topic = None
            if self.config['mqtt']['listener']['trace'] in data.source:
                matching_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['trace']}"
            else:
                matching_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['result']}"
            
            print(f"\nDEBUG: Adding message to buffer:")
            print(f"Source: {data.source}")
            print(f"Matching topic: {matching_topic}")
            print(f"Buffer size before: {len(self.buffers[matching_topic])}")
            
            # Store message in the correct buffer
            self.buffers[matching_topic].append(data)
            
            print(f"Buffer size after: {len(self.buffers[matching_topic])}")
            print(f"All buffers: {[f'{k}: {len(v)}' for k,v in self.buffers.items()]}")
            
            # Look for matches immediately after adding
            self.find_and_process_matches()
            
            # Check if it's time to cleanup
            current_time = datetime.now().timestamp()
            if current_time - self.last_cleanup >= self.cleanup_interval:
                self.cleanup_old_messages()
                self.last_cleanup = current_time

        except Exception as e:
            print(f"Error in add_message: {str(e)}")

    def find_and_process_matches(self):
        """Find and process messages with matching timestamps and tool IDs"""
        try:
            print("\nDEBUG: Checking for matches")
            
            result_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['result']}"
            trace_topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['trace']}"
            
            result_messages = self.buffers.get(result_topic, [])
            trace_messages = self.buffers.get(trace_topic, [])
            
            print(f"Result messages: {len(result_messages)}")
            print(f"Trace messages: {len(trace_messages)}")
            
            # Keep track of messages to remove
            to_remove_result = []
            to_remove_trace = []
            
            # Find all matches without removing immediately
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
                        
                        print(f"\nFound matching pair!")
                        print(f"Tool: {r_tool_key}")
                        print(f"Timestamp: {datetime.fromtimestamp(result_msg.timestamp)}")
                        
                        # Process the pair
                        self.process_matching_messages([result_msg, trace_msg])
                        
                        # Mark for removal
                        to_remove_result.append(result_msg)
                        to_remove_trace.append(trace_msg)
            
            # Remove processed messages after finding all matches
            self.buffers[result_topic] = [msg for msg in result_messages if msg not in to_remove_result]
            self.buffers[trace_topic] = [msg for msg in trace_messages if msg not in to_remove_trace]
            
        except Exception as e:
            print(f"Error in find_and_process_matches: {str(e)}")

    def cleanup_old_messages(self):
        """Remove messages older than the cleanup interval"""
        try:
            print("\nDEBUG: Cleaning up old messages")
            current_time = datetime.now().timestamp()
            removed_count = 0
            
            for topic in self.topics:
                original_length = len(self.buffers[topic])
                self.buffers[topic] = [
                    msg for msg in self.buffers[topic]
                    if current_time - msg.timestamp <= self.cleanup_interval
                ]
                removed_count += original_length - len(self.buffers[topic])
            
            print(f"Removed {removed_count} old messages")
            
        except Exception as e:
            print(f"Error in cleanup_old_messages: {str(e)}")

    def process_matching_messages(self, matches: List[TimestampedData]):
        """Process messages that have matching timestamps"""
        try:
            result_msg = next(m for m in matches if self.config['mqtt']['listener']['result'] in m.source)
            trace_msg = next(m for m in matches if self.config['mqtt']['listener']['trace'] in m.source)
            
            parts = result_msg.source.split('/')
            toolbox_id = parts[1]
            tool_id = parts[2]
            
            print("\nProcessing matched pair:")
            print(f"Toolbox ID: {toolbox_id}")
            print(f"Tool ID: {tool_id}")
            print(f"Timestamp: {datetime.fromtimestamp(result_msg.timestamp)}")
            print(f"Time difference: {abs(result_msg.timestamp - trace_msg.timestamp):.3f} seconds")
            
            # Here you can process the paired data as needed
            # result_data = result_msg.data
            # trace_data = trace_msg.data
            
        except Exception as e:
            print(f"Error in process_matching_messages: {str(e)}")

    def run(self):
        """Main method to set up MQTT clients and start listening"""
        result_client = self.create_result_listener()
        trace_client = self.create_trace_listener()

        result_client.connect(self.config['mqtt']['broker']['host'], 
                            self.config['mqtt']['broker']['port'])
        trace_client.connect(self.config['mqtt']['broker']['host'], 
                           self.config['mqtt']['broker']['port'])
        
        # Start both loops in non-blocking mode
        result_client.loop_start()
        trace_client.loop_start()

        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping MQTT clients...")
            result_client.loop_stop()
            trace_client.loop_stop()
            result_client.disconnect()
            trace_client.disconnect()

def main():
    """
    Main entry point for the application.
    
    Command line arguments:
        --config: Path to YAML configuration file (default: mqtt_conf.yaml)
    
    Example usage:
        python -m drilling_analyzer
        python -m drilling_analyzer --config=/path/to/custom_config.yaml
    """
    parser = argparse.ArgumentParser(description='MQTT Drilling Data Analyzer')
    parser.add_argument('--config', 
                       type=str,
                       default='mqtt_conf.yaml',
                       help='Path to YAML configuration file (default: mqtt_conf.yaml)')
    
    args = parser.parse_args()
    
    try:
        analyzer = DrillingDataAnalyser(config_path=args.config)
        analyzer.run()
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML configuration in '{args.config}': {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()