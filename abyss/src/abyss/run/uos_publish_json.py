import paho.mqtt.client as mqtt
import time
import json
import argparse
import logging
import yaml
import random
from pathlib import Path
import signal
import sys

# This is a simple MQTT publisher that publishes data continuously
# to a broker. The data is read from a set of files and published to a topic.
# The data is published in a random order to simulate real-time data.
# The data is published to a set of topics in a random order.
# The source timestamp is updated to the current time before publishing.
# The data is published to a random toolboxid and toolid.






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
    Publishes a payload to a specified MQTT topic after replacing an old timestamp with a new one.

    Args:
        client: The MQTT client instance used for publishing.
        topic (str): The MQTT topic to publish to.
        payload (str): The message payload (JSON as string).
        timestamp0 (str): The original timestamp string to be replaced.
        timestamp1 (str): The new timestamp string to use in the payload.
    """
    if timestamp0 in payload:
        payload = payload.replace(timestamp0, timestamp1)
    else:
        logging.warning(f"'{timestamp0}' not found in payload. Skipping replacement.")
    client.publish(topic, payload)
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    logging.info(f"[{current_time}]: Publish data on {topic} '{timestamp1}'")


def main():
    """
    Main function to continuously publish JSON data from specified files to MQTT topics.
    Reads configuration from a YAML file, selects random data, updates timestamps, and publishes to MQTT broker.
    """
    # Parse command line arguments
    # Initialise parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("path", type=str, help="Path to the data folder")
    parser.add_argument("-c", "--conf", type=str, help="YAML configuration file", default="config/mqtt_conf_docker.yaml")
    parser.add_argument("--sleep-min", type=float, default=0.1, help="Minimum sleep interval between publishes (seconds)")
    parser.add_argument("--sleep-max", type=float, default=0.3, help="Maximum sleep interval between publishes (seconds)")
    parser.add_argument("-r","--repetitions", type=int, default=10, help="Number of repetitions for publishing data")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")

    # Read arguments
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


    # config = yaml.safe_load(open("mqtt_conf.yaml"))
    with open(args.conf) as f:
        config = yaml.safe_load(f)



    # Sample data folders, these are the new datasets
    # DATA_FOLDERS = ['data0', 'data1']
    DATA_FOLDERS = [args.path]

    # MQTT topics for publishing
    # Validate configuration keys
    # required_keys = ['mqtt', 'publisher', 'topics', 'result', 'trace']
    # current_config = config
    # for key in required_keys:
    #     if key not in current_config:
    #         raise KeyError(f"Missing required configuration key: '{key}'")
    #     current_config = current_config[key] if isinstance(current_config, dict) else {}

    # TOPIC_RESULT = config['mqtt']['publisher']['topics']['result']
    # TOPIC_TRACE = config['mqtt']['publisher']['topics']['trace']

    # Made up toolboxids and toolids for simulating multiple tools
    TOOLBOXIDS = ['ILLL502033771', 'ILLL502033772', 'ILLL502033773', 'ILLL502033774', 'ILLL502033775']
    TOOLIDS = ['setitec001', 'setitec002', 'setitec003', 'setitec004', 'setitec005']

    # Create client
    client = mqtt.Client()
    try:
        client.connect(config['mqtt']['broker']['host'], 
                    config['mqtt']['broker']['port'])
    except Exception as e:
        logging.error(f"Failed to connect to MQTT broker: {e}")
        sys.exit(1)
    # Handle termination signals


    def signal_handler(sig, frame):
        logging.info("\nTermination signal received. Exiting gracefully...")
        client.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Publish data continuously
    # while True:
    for _ in range(args.repetitions):
        # Select a random data folder to publish and read the data
        random.shuffle(DATA_FOLDERS)
        data_folder = DATA_FOLDERS[0]
        data_folder = Path(data_folder) 
        # with open (f'{os.getcwd()}\\{data_folder}\\ResultManagement.json') as f:
        try:
            # Read the JSON files
            with open(data_folder / "ResultManagement.json") as f:
                d_result = f.read()
                result_source_timestamps = find_in_dict(json.loads(d_result), 'SourceTimestamp')
                if len(set(result_source_timestamps)) != 1:
                    raise ValueError("SourceTimestamp values are not identical.")
                original_source_timestamp = result_source_timestamps[0]
            with open(data_folder / "Trace.json") as f:
                d_trace = f.read()
                trace_source_timestamps = find_in_dict(json.loads(d_trace), 'SourceTimestamp')
                if len(set(trace_source_timestamps)) != 1:
                    raise ValueError("Trace SourceTimestamp values are not identical.")
            with open(data_folder / "Heads.json") as f:
                d_heads = f.read()
                heads_source_timestamps = find_in_dict(json.loads(d_heads), 'SourceTimestamp')
                if len(set(heads_source_timestamps)) != 1:
                    raise ValueError("Heads SourceTimestamp values are not identical.")

            # Update the source timestamp        
            new_source_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())
            
            # Prepare a shuffled topic set
            random.shuffle(TOOLBOXIDS)
            random.shuffle(TOOLIDS)
            
            # Prepare the topics
            topic_result = f"{config['mqtt']['listener']['root']}/{TOOLBOXIDS[0]}/{TOOLIDS[0]}/{config['mqtt']['listener']['result']}"
            topic_trace = f"{config['mqtt']['listener']['root']}/{TOOLBOXIDS[0]}/{TOOLIDS[0]}/{config['mqtt']['listener']['trace']}"
            topic_heads = f"{config['mqtt']['listener']['root']}/{TOOLBOXIDS[0]}/{TOOLIDS[0]}/{config['mqtt']['listener']['heads']}"



            # Shuffle the data order
            list_to_publish = [(topic_result, d_result), (topic_trace, d_trace), (topic_heads, d_heads)] 
            random.shuffle(list_to_publish)

            # Publish the data
            for item in list_to_publish:
                publish(client, item[0], item[1], original_source_timestamp, new_source_timestamp)
                time.sleep(random.uniform(args.sleep_min, args.sleep_max))  # Sleep for a random time between min and max
        
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}. Skipping this data folder.")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}. Skipping this data folder.")
        except ValueError as e:
            logging.error(f"Data validation error: {e}. Skipping this data folder.")

if __name__ == "__main__":
    main()