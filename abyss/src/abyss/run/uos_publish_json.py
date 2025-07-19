"""
MQTT JSON Data Publisher

A simple MQTT publisher that publishes JSON data continuously to a broker.
The data is read from specified files and published to topics with:
- Random order simulation for real-time data
- Updated source timestamps
- Random toolbox and tool IDs
"""

import argparse
import csv
import json
import logging
import random
import signal
import sys
import time
import uuid
from pathlib import Path

import paho.mqtt.client as mqtt
import yaml

from abyss.uos_depth_est_utils import setup_logging


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

def publish(client, topic, payload, timestamp0, timestamp1) -> None:
    """
    Publish a payload to a specified MQTT topic after replacing timestamps.

    Args:
        client: The MQTT client instance used for publishing
        topic: The MQTT topic to publish to
        payload: The message payload (JSON as string)
        timestamp0: The original timestamp string to be replaced
        timestamp1: The new timestamp string to use in the payload
    """
    if timestamp0 in payload:
        payload = payload.replace(timestamp0, timestamp1)
    else:
        logging.warning(f"'{timestamp0}' not found in payload. Skipping replacement.")
    
    client.publish(topic, payload)
    current_time = time.strftime("%H:%M:%S", time.localtime())
    logging.info(f"[{current_time}]: Published data on {topic} '{timestamp1}'")


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up and return the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Publish JSON data from files to MQTT broker with simulated real-time behavior"
    )
    
    parser.add_argument(
        "path", 
        type=str, 
        help="Path to the data folder containing JSON files"
    )
    parser.add_argument(
        "-c", "--conf", 
        type=str, 
        help="YAML configuration file", 
        default="config/mqtt_conf_docker.yaml"
    )
    parser.add_argument(
        "--sleep-min", 
        type=float, 
        default=0.1, 
        help="Minimum sleep interval between publishes (seconds)"
    )
    parser.add_argument(
        "--sleep-max", 
        type=float, 
        default=0.3, 
        help="Maximum sleep interval between publishes (seconds)"
    )
    parser.add_argument(
        "-r", "--repetitions", 
        type=int, 
        default=10, 
        help="Number of repetitions for publishing data"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        help="Set the logging level"
    )
    parser.add_argument(
        "--track-signals",
        action="store_true",
        help="Enable signal tracking with unique IDs"
    )
    parser.add_argument(
        "--signal-log",
        type=str,
        default="sent_signals.csv",
        help="CSV file for signal tracking log"
    )
    
    return parser


def validate_and_get_data_folders(path_str: str) -> list[Path]:
    """
    Validate the input path and return a list of data folders containing JSON files.
    
    Args:
        path_str: Path string from command line arguments
        
    Returns:
        List of Path objects representing directories containing JSON files
        
    Raises:
        ValueError: If path is invalid or doesn't contain required structure
    """
    path = Path(path_str)
    
    if path.is_file():
        raise ValueError(f"Path '{path_str}' is a file, but a directory is required")
    
    if not path.is_dir():
        raise ValueError(f"Path '{path_str}' does not exist")
    
    # Check for subdirectories first
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    
    if subdirs:
        # Filter subdirectories that contain JSON files
        data_folders = []
        for subdir in subdirs:
            json_files = [f for f in subdir.iterdir() if f.is_file() and f.suffix.lower() == '.json']
            if json_files:
                data_folders.append(subdir)
        
        if not data_folders:
            raise ValueError(f"No subdirectories in '{path_str}' contain JSON files")
        
        return data_folders
    else:
        # No subdirectories, check if the directory itself contains JSON files
        json_files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() == '.json']
        if json_files:
            return [path]
        else:
            raise ValueError(f"Directory '{path_str}' contains no JSON files")


def setup_mqtt_client(config: dict) -> mqtt.Client:
    """Set up and connect MQTT client."""
    client = mqtt.Client()
    try:
        client.connect(config['mqtt']['broker']['host'], config['mqtt']['broker']['port'])
        return client
    except Exception as e:
        logging.error(f"Failed to connect to MQTT broker: {e}")
        sys.exit(1)


def setup_signal_handlers(client: mqtt.Client) -> None:
    """Set up signal handlers for graceful termination."""
    def signal_handler(sig, frame):
        logging.info("\nTermination signal received. Exiting gracefully...")
        client.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def read_and_validate_json_files(data_folder: Path) -> tuple[str, str, str, str]:
    """
    Read and validate JSON files from a data folder.
    
    Returns:
        Tuple of (result_data, trace_data, heads_data, original_timestamp)
    """
    # Read ResultManagement.json
    with open(data_folder / "ResultManagement.json") as f:
        d_result = f.read()
        result_source_timestamps = find_in_dict(json.loads(d_result), 'SourceTimestamp')
        if len(set(result_source_timestamps)) != 1:
            raise ValueError("ResultManagement SourceTimestamp values are not identical.")
        original_source_timestamp = result_source_timestamps[0]
    
    # Read Trace.json
    with open(data_folder / "Trace.json") as f:
        d_trace = f.read()
        trace_source_timestamps = find_in_dict(json.loads(d_trace), 'SourceTimestamp')
        if len(set(trace_source_timestamps)) != 1:
            raise ValueError("Trace SourceTimestamp values are not identical.")
    
    # Read Heads.json
    with open(data_folder / "Heads.json") as f:
        d_heads = f.read()
        heads_source_timestamps = find_in_dict(json.loads(d_heads), 'SourceTimestamp')
        if len(set(heads_source_timestamps)) != 1:
            raise ValueError("Heads SourceTimestamp values are not identical.")
    
    return d_result, d_trace, d_heads, original_source_timestamp


def main():
    """
    Main function to continuously publish JSON data from specified files to MQTT topics.
    Reads configuration from a YAML file, selects random data, updates timestamps, 
    and publishes to MQTT broker.
    """
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    setup_logging(getattr(logging, args.log_level.upper()))

    # Load configuration
    try:
        with open(args.conf) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file '{args.conf}' not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)

    # Validate path and get data folders
    try:
        data_folders = validate_and_get_data_folders(args.path)
        logging.info(f"Found {len(data_folders)} data folder(s) to process")
    except ValueError as e:
        logging.error(e)
        sys.exit(1)

    # Constants for simulation
    TOOLBOX_IDS = [
        'ILLL502033771', 'ILLL502033772', 'ILLL502033773', 
        'ILLL502033774', 'ILLL502033775'
    ]
    TOOL_IDS = [
        'setitec001', 'setitec002', 'setitec003', 
        'setitec004', 'setitec005'
    ]

    # Set up MQTT client and signal handlers
    client = setup_mqtt_client(config)
    setup_signal_handlers(client)

    # Publish data for specified number of repetitions
    for repetition in range(args.repetitions):
        logging.info(f"Starting repetition {repetition + 1}/{args.repetitions}")
        
        # Select random data folder
        data_folder = random.choice(data_folders)
        logging.debug(f"Selected data folder: {data_folder}")
        
        try:
            # Read and validate JSON files
            d_result, d_trace, d_heads, original_timestamp = read_and_validate_json_files(data_folder)

            # Generate new timestamp
            new_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())
            
            # Select random toolbox and tool IDs
            toolbox_id = random.choice(TOOLBOX_IDS)
            tool_id = random.choice(TOOL_IDS)
            
            # Handle signal tracking if enabled
            signal_id = None
            if args.track_signals:
                signal_id = str(uuid.uuid4())
                # Log to CSV immediately
                with open(args.signal_log, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([signal_id, time.time(), toolbox_id, tool_id])
                
                # Inject signal ID into JSON data
                result_data = json.loads(d_result)
                # Add signal ID at top level for easy access
                result_data['_signal_id'] = signal_id
                d_result = json.dumps(result_data)
                
                trace_data = json.loads(d_trace)
                # Add signal ID at top level for easy access
                trace_data['_signal_id'] = signal_id
                d_trace = json.dumps(trace_data)
                
                heads_data = json.loads(d_heads)
                # Add signal ID at top level for easy access
                heads_data['_signal_id'] = signal_id
                d_heads = json.dumps(heads_data)
                
                logging.debug(f"Signal tracking enabled - ID: {signal_id}")
            
            # Build topics
            mqtt_root = config['mqtt']['listener']['root']
            topic_result = f"{mqtt_root}/{toolbox_id}/{tool_id}/{config['mqtt']['listener']['result']}"
            topic_trace = f"{mqtt_root}/{toolbox_id}/{tool_id}/{config['mqtt']['listener']['trace']}"
            topic_heads = f"{mqtt_root}/{toolbox_id}/{tool_id}/{config['mqtt']['listener']['heads']}"

            # Prepare data for publishing in random order
            publish_items = [
                (topic_result, d_result),
                (topic_trace, d_trace),
                (topic_heads, d_heads)
            ]
            random.shuffle(publish_items)

            # Publish the data
            for topic, payload in publish_items:
                publish(client, topic, payload, original_timestamp, new_timestamp)
                time.sleep(random.uniform(args.sleep_min, args.sleep_max))
        
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}. Skipping this data folder.")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}. Skipping this data folder.")
        except ValueError as e:
            logging.error(f"Data validation error: {e}. Skipping this data folder.")
        except Exception as e:
            logging.error(f"Unexpected error: {e}. Skipping this data folder.")

    logging.info("Publishing completed")
    client.disconnect()

if __name__ == "__main__":
    main()