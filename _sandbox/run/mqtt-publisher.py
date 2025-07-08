#!/usr/bin/env python3

import paho.mqtt.client as mqtt
import time
import json
import argparse
import os
import yaml
import random
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class MQTTPublisher:
    """MQTT Publisher that sends data from JSON files to specified topics."""
    
    def __init__(self, config: Dict, data_folders: List[str]):
        """Initialize MQTT Publisher with configuration and data folders.
        
        Args:
            config: Dictionary containing MQTT configuration
            data_folders: List of folder names containing data to publish
        """
        self.config = config
        self.data_folders = data_folders
        # Toolbox and tool IDs, totally made up
        #TODO: make these configurable
        self.toolbox_ids = ['ILLL502033771', 'ILLL502033772', 'ILLL502033773', 
                           'ILLL502033774', 'ILLL502033775']
        self.tool_ids = ['setitec001', 'setitec002', 'setitec003', 
                        'setitec004', 'setitec005']
        
        # Setup MQTT client
        self.client = mqtt.Client()
        self._connect_mqtt()
        
        # Setup logging
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

    def _connect_mqtt(self) -> None:
        """Establish connection to MQTT broker."""
        try:
            self.client.connect(
                self.config['mqtt']['broker']['host'],
                self.config['mqtt']['broker']['port']
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def _find_in_dict(self, data: Dict, target_key: str) -> List:
        """Recursively search for a key in a nested dictionary.
        
        Args:
            data: Dictionary to search in
            target_key: Key to search for
            
        Returns:
            List of values found for that key
        """
        results = []
        
        def _search(current_dict: Dict) -> None:
            for key, value in current_dict.items():
                if key == target_key:
                    results.append(value)
                if isinstance(value, dict):
                    _search(value)
        
        _search(data)
        return results

    def _read_json_file(self, file_path: Path) -> Tuple[str, List[str]]:
        """Read JSON file and extract source timestamps.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Tuple of (file content, source timestamps)
        """
        try:
            with open(file_path) as f:
                content = f.read()
                data = json.loads(content)
                timestamps = self._find_in_dict(dict(data), 'SourceTimestamp')
                if not timestamps:
                    raise ValueError("No SourceTimestamp found in data")
                if len(set(timestamps)) != 1:
                    raise ValueError("Multiple different SourceTimestamps found")
                return content, timestamps
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            raise

    def publish(self, topic: str, payload: str, 
                timestamp0: str, timestamp1: str) -> None:
        """Publish data to MQTT topic.
        
        Args:
            topic: MQTT topic to publish to
            payload: Data to publish
            timestamp0: Original timestamp to replace
            timestamp1: New timestamp to use
        """
        try:
            payload = payload.replace(timestamp0, timestamp1)
            data = json.loads(payload)
            self.client.publish(topic, json.dumps(data))
            self.logger.info(f"Published data on {topic} '{timestamp1}'")
        except Exception as e:
            self.logger.error(f"Error publishing to {topic}: {e}")
            raise

    def run(self) -> None:
        """Main loop to continuously publish data."""
        while True:
            try:
                # Select random data folder
                random.shuffle(self.data_folders)
                data_folder = self.data_folders[0]
                data_path = Path(os.getcwd()) / 'data' / data_folder
                
                # Read data files
                result_content, result_timestamps = self._read_json_file(
                    data_path / 'Result.json'
                )
                trace_content, _ = self._read_json_file(
                    data_path / 'Trace.json'
                )
                
                # Update timestamp
                original_timestamp = result_timestamps[0]
                new_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", 
                                            time.localtime())
                
                # Prepare topics
                random.shuffle(self.toolbox_ids)
                random.shuffle(self.tool_ids)
                
                topic_base = (f"{self.config['mqtt']['listener']['root']}/"
                            f"{self.toolbox_ids[0]}/{self.tool_ids[0]}")
                
                topic_result = f"{topic_base}/{self.config['mqtt']['listener']['result']}"
                topic_trace = f"{topic_base}/{self.config['mqtt']['listener']['trace']}"
                
                # Publish data in random order
                publish_list = [(topic_result, result_content), 
                              (topic_trace, trace_content)]
                random.shuffle(publish_list)
                
                for topic, content in publish_list:
                    self.publish(topic, content, original_timestamp, new_timestamp)
                    time.sleep(random.randint(1, 3))
                    
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(5)  # Wait before retrying

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='MQTT Publisher for publishing JSON data to topics.'
    )
    parser.add_argument(
        "-c", "--conf",
        type=str,
        help="YAML configuration file",
        default="mqtt_localhost_conf.yaml"
    )
    parser.add_argument(
        "-l", "--log-level",
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Set the logging level"
    )
    return parser.parse_args()

def main() -> None:
    """Main entry point for the MQTT publisher."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=getattr(logging, args.log_level)
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        with open(args.conf) as f:
            config = yaml.safe_load(f)
            
        # Initialize and run publisher
        # TODO: make the data folders configurable
        data_folders = ['data0', 'data1', 'data2']
        publisher = MQTTPublisher(config, data_folders)
        publisher.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
