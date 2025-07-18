"""
Test script for verifying exact timestamp matching in MQTT messages.

This script subscribes to MQTT topics and logs the SourceTimestamp values
to help debug the exact matching system.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime

import paho.mqtt.client as mqtt
import yaml

from abyss.uos_depth_est_utils import setup_logging


def find_in_dict(data: dict, target_key: str) -> list:
    """Recursively search for a key in a nested dictionary."""
    results = []
    
    def _search(current_dict):
        for key, value in current_dict.items():
            if key == target_key:
                results.append(value)
            if isinstance(value, dict):
                _search(value)
    
    _search(data)
    return results


class ExactMatchTester:
    def __init__(self, config):
        self.config = config
        self.messages_received = defaultdict(list)  # (tool_key, timestamp) -> list of message types
        self.all_messages = []  # All messages in order received
        self.start_time = time.time()
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("Connected to MQTT broker")
            # Subscribe to all three topics
            topics = [
                f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['result']}",
                f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['trace']}",
                f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['heads']}"
            ]
            for topic in topics:
                client.subscribe(topic)
                logging.info(f"Subscribed to: {topic}")
        else:
            logging.error(f"Failed to connect with result code {rc}")
            
    def on_message(self, client, userdata, msg):
        try:
            # Parse message
            data = json.loads(msg.payload)
            
            # Extract tool key from topic
            parts = msg.topic.split('/')
            if len(parts) >= 3:
                tool_key = f"{parts[1]}/{parts[2]}"
            else:
                tool_key = "unknown"
                
            # Determine message type
            if 'ResultManagement' in msg.topic and 'Trace' not in msg.topic:
                msg_type = 'result'
            elif 'Trace' in msg.topic:
                msg_type = 'trace'
            elif 'AssetManagement' in msg.topic or 'Heads' in msg.topic:
                msg_type = 'heads'
            else:
                msg_type = 'unknown'
                
            # Extract SourceTimestamp
            timestamps = find_in_dict(data, 'SourceTimestamp')
            if timestamps:
                source_timestamp = timestamps[0]
            else:
                source_timestamp = "NO_TIMESTAMP_FOUND"
                
            # Log the message
            receipt_time = time.time() - self.start_time
            logging.info(f"[{receipt_time:.3f}s] Received {msg_type} from {tool_key} with timestamp: {source_timestamp}")
            
            # Track the message
            key = (tool_key, source_timestamp)
            self.messages_received[key].append(msg_type)
            
            # Store full message info
            self.all_messages.append({
                'time': receipt_time,
                'tool_key': tool_key,
                'msg_type': msg_type,
                'timestamp': source_timestamp,
                'topic': msg.topic
            })
            
            # Check if we have a complete set
            if set(self.messages_received[key]) == {'result', 'trace', 'heads'}:
                logging.info(f"✓ COMPLETE MATCH for {tool_key} with timestamp {source_timestamp}")
                
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from {msg.topic}: {e}")
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            
    def print_summary(self):
        """Print a summary of all messages received."""
        print("\n" + "="*80)
        print("EXACT MATCHING TEST SUMMARY")
        print("="*80)
        
        # Group by tool_key and timestamp
        complete_matches = 0
        incomplete_matches = 0
        
        for (tool_key, timestamp), msg_types in sorted(self.messages_received.items()):
            msg_set = set(msg_types)
            if msg_set == {'result', 'trace', 'heads'}:
                status = "✓ COMPLETE"
                complete_matches += 1
            else:
                status = "✗ INCOMPLETE"
                incomplete_matches += 1
                
            print(f"\n{status} | {tool_key} | {timestamp}")
            print(f"  Messages received: {', '.join(sorted(msg_types))}")
            
        print(f"\n" + "-"*80)
        print(f"Total unique (tool_key, timestamp) combinations: {len(self.messages_received)}")
        print(f"Complete matches: {complete_matches}")
        print(f"Incomplete matches: {incomplete_matches}")
        
        # Show timeline
        print(f"\n" + "-"*80)
        print("MESSAGE TIMELINE:")
        for msg in self.all_messages[-20:]:  # Last 20 messages
            print(f"  [{msg['time']:6.3f}s] {msg['msg_type']:6} | {msg['tool_key']} | {msg['timestamp']}")
            
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Test exact timestamp matching in MQTT messages")
    parser.add_argument(
        "-c", "--conf", 
        type=str, 
        default="config/mqtt_conf_docker.yaml",
        help="YAML configuration file"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=30,
        help="Duration to run the test (seconds)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    # Priority: Environment variable > Command line argument
    log_level = os.environ.get('LOG_LEVEL', args.log_level)
    setup_logging(getattr(logging, log_level))
    
    # Load configuration
    try:
        with open(args.conf) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
        
    # Create tester
    tester = ExactMatchTester(config)
    
    # Set up MQTT client
    client = mqtt.Client()
    client.on_connect = tester.on_connect
    client.on_message = tester.on_message
    
    # Set up signal handler
    def signal_handler(sig, frame):
        print("\nShutting down...")
        tester.print_summary()
        client.disconnect()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Connect to broker
    try:
        client.connect(
            config['mqtt']['broker']['host'],
            config['mqtt']['broker']['port']
        )
    except Exception as e:
        logging.error(f"Failed to connect to MQTT broker: {e}")
        sys.exit(1)
        
    # Start listening
    client.loop_start()
    
    logging.info(f"Listening for {args.duration} seconds...")
    logging.info("Press Ctrl+C to stop early and see summary")
    
    # Run for specified duration
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
        
    # Print summary and exit
    tester.print_summary()
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()