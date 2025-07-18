"""
Simplified MQTT JSON Data Publisher

A simplified MQTT publisher that preserves original timestamps from JSON files.
This version does NOT modify timestamps, ensuring exact matching can work properly.

Key differences from uos_publish_json.py:
- No timestamp modification (preserves original SourceTimestamp values)
- No random shuffling of message order
- Publishes all three message types immediately for atomic delivery
- Validates that all three files have matching timestamps
"""

import argparse
import json
import logging
import os
import random
import signal
import sys
import time
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


def publish_simple(client, topic, payload, debug_timestamps=False) -> None:
    """
    Publish a payload to a specified MQTT topic WITHOUT modifying timestamps.

    Args:
        client: The MQTT client instance used for publishing
        topic: The MQTT topic to publish to
        payload: The message payload (JSON as string)
        debug_timestamps: If True, log the SourceTimestamp values found
    """
    if debug_timestamps:
        try:
            data = json.loads(payload)
            timestamps = find_in_dict(data, 'SourceTimestamp')
            if timestamps:
                logging.debug(f"Publishing to {topic} with SourceTimestamp: {timestamps[0]}")
        except Exception as e:
            logging.debug(f"Could not extract timestamp for debugging: {e}")
    
    client.publish(topic, payload)
    current_time = time.strftime("%H:%M:%S", time.localtime())
    logging.info(f"[{current_time}]: Published data on {topic}")


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up and return the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Simplified JSON publisher that preserves original timestamps for exact matching"
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
        "--delay-between-sets", 
        type=float, 
        default=1.0, 
        help="Delay between publishing different datasets (seconds)"
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
        "--debug-timestamps",
        action="store_true",
        help="Log the SourceTimestamp values being published"
    )
    parser.add_argument(
        "--update-timestamps",
        action="store_true",
        help="Update timestamps to current time (legacy behavior)"
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


def read_and_validate_json_files(data_folder: Path, validate_matching_timestamps=True) -> tuple[str, str, str, str]:
    """
    Read and validate JSON files from a data folder.
    
    Args:
        data_folder: Path to the folder containing JSON files
        validate_matching_timestamps: If True, ensure all files have identical timestamps
    
    Returns:
        Tuple of (result_data, trace_data, heads_data, original_timestamp)
        
    Raises:
        ValueError: If timestamps don't match across files (when validation is enabled)
    """
    timestamps_found = {}
    
    # Read ResultManagement.json
    result_file = data_folder / "ResultManagement.json"
    with open(result_file) as f:
        d_result = f.read()
        # Validate JSON and extract timestamps
        try:
            result_json = json.loads(d_result)
            result_timestamps = find_in_dict(result_json, 'SourceTimestamp')
            if result_timestamps:
                timestamps_found['result'] = result_timestamps[0]
                logging.debug(f"Result timestamp: {result_timestamps[0]}")
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {result_file}: {e}")
            raise
    
    # Read Trace.json
    trace_file = data_folder / "Trace.json"
    with open(trace_file) as f:
        d_trace = f.read()
        try:
            trace_json = json.loads(d_trace)
            trace_timestamps = find_in_dict(trace_json, 'SourceTimestamp')
            if trace_timestamps:
                timestamps_found['trace'] = trace_timestamps[0]
                logging.debug(f"Trace timestamp: {trace_timestamps[0]}")
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {trace_file}: {e}")
            raise
    
    # Read Heads.json
    heads_file = data_folder / "Heads.json"
    with open(heads_file) as f:
        d_heads = f.read()
        try:
            heads_json = json.loads(d_heads)
            heads_timestamps = find_in_dict(heads_json, 'SourceTimestamp')
            if heads_timestamps:
                timestamps_found['heads'] = heads_timestamps[0]
                logging.debug(f"Heads timestamp: {heads_timestamps[0]}")
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {heads_file}: {e}")
            raise
    
    # Validate that all timestamps match
    if validate_matching_timestamps:
        unique_timestamps = set(timestamps_found.values())
        if len(unique_timestamps) > 1:
            logging.error(f"Timestamp mismatch in {data_folder}:")
            for file_type, timestamp in timestamps_found.items():
                logging.error(f"  {file_type}: {timestamp}")
            raise ValueError(f"SourceTimestamp values don't match across files in {data_folder}")
        elif len(unique_timestamps) == 0:
            raise ValueError(f"No SourceTimestamp found in files in {data_folder}")
    
    # Return the data and the common timestamp
    original_timestamp = list(timestamps_found.values())[0] if timestamps_found else "unknown"
    return d_result, d_trace, d_heads, original_timestamp


def main():
    """
    Main function to publish JSON data from files to MQTT topics.
    This simplified version preserves original timestamps for exact matching.
    """
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    # Priority: Environment variable > Command line argument
    log_level = os.environ.get('LOG_LEVEL', args.log_level.upper())
    setup_logging(getattr(logging, log_level))
    
    logging.info("Starting Simplified MQTT Publisher (timestamp-preserving mode)")
    if args.update_timestamps:
        logging.warning("Running with --update-timestamps flag (legacy behavior)")

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

    # Statistics
    successful_publishes = 0
    failed_publishes = 0

    # Publish data for specified number of repetitions
    for repetition in range(args.repetitions):
        logging.info(f"Starting repetition {repetition + 1}/{args.repetitions}")
        
        # Select random data folder
        data_folder = random.choice(data_folders)
        logging.debug(f"Selected data folder: {data_folder}")
        
        try:
            # Read and validate JSON files
            d_result, d_trace, d_heads, original_timestamp = read_and_validate_json_files(
                data_folder, 
                validate_matching_timestamps=True
            )
            
            logging.info(f"Publishing dataset with SourceTimestamp: {original_timestamp}")
            
            # Select random toolbox and tool IDs
            toolbox_id = random.choice(TOOLBOX_IDS)
            tool_id = random.choice(TOOL_IDS)
            
            # Build topics
            mqtt_root = config['mqtt']['listener']['root']
            topic_result = f"{mqtt_root}/{toolbox_id}/{tool_id}/{config['mqtt']['listener']['result']}"
            topic_trace = f"{mqtt_root}/{toolbox_id}/{tool_id}/{config['mqtt']['listener']['trace']}"
            topic_heads = f"{mqtt_root}/{toolbox_id}/{tool_id}/{config['mqtt']['listener']['heads']}"

            # If update_timestamps is enabled (legacy mode), update the timestamps
            if args.update_timestamps:
                new_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())
                logging.debug(f"Updating timestamp from {original_timestamp} to {new_timestamp}")
                d_result = d_result.replace(original_timestamp, new_timestamp)
                d_trace = d_trace.replace(original_timestamp, new_timestamp)
                d_heads = d_heads.replace(original_timestamp, new_timestamp)

            # Publish all three messages immediately (no delays between them)
            # This ensures they arrive close together for exact matching
            publish_simple(client, topic_result, d_result, args.debug_timestamps)
            publish_simple(client, topic_trace, d_trace, args.debug_timestamps)
            publish_simple(client, topic_heads, d_heads, args.debug_timestamps)
            
            successful_publishes += 1
            
            # Delay before next dataset (not between message types)
            if repetition < args.repetitions - 1:
                time.sleep(args.delay_between_sets)
        
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}. Skipping this data folder.")
            failed_publishes += 1
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error in {data_folder}: {e}")
            logging.error("This might be due to file truncation. Check file integrity.")
            failed_publishes += 1
        except ValueError as e:
            logging.error(f"Data validation error: {e}. Skipping this data folder.")
            failed_publishes += 1
        except Exception as e:
            logging.error(f"Unexpected error with {data_folder}: {e}")
            failed_publishes += 1

    # Summary
    logging.info(f"Publishing completed. Success: {successful_publishes}, Failed: {failed_publishes}")
    
    # Give time for last messages to be sent
    time.sleep(0.5)
    
    client.disconnect()


if __name__ == "__main__":
    main()