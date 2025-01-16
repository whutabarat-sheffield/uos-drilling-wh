import paho.mqtt.client as mqtt
import json
import yaml
import argparse
import datetime

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

def create_result_listener():
    """Create an MQTT client for listening to Result data"""
    def on_connect(client, userdata, flags, rc):
        print("Result Listener Connected with result code " + str(rc))
        # Subscribe to all result topics under all toolboxes and tools
        topic = f"{config['mqtt']['listener']['root']}/+/+/{config['mqtt']['listener']['result']}"
        client.subscribe(topic)
        print(f"Subscribed to {topic}")

    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload)
            print(f"\nResult Message on {msg.topic}:")
            st = find_in_dict(data, 'SourceTimestamp')
            print(f"Source Timestamp: {st[0]}")
            # Convert to datetime object
            dt = datetime.datetime.strptime(st[0], "%Y-%m-%dT%H:%M:%SZ")
            # Convert to Unix timestamp (seconds since epoch)
            unix_timestamp = dt.timestamp()
            print(f"Unix Timestamp: {unix_timestamp}")
            print(f"ToolboxId: {msg.topic.split('/')[1]}")
            print(f"ToolId: {msg.topic.split('/')[2]}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {msg.topic}")

    client = mqtt.Client("result_listener")
    client.on_connect = on_connect
    client.on_message = on_message
    return client

def create_trace_listener():
    """Create an MQTT client for listening to Trace data"""
    def on_connect(client, userdata, flags, rc):
        print("Trace Listener Connected with result code " + str(rc))
        # Subscribe to all trace topics under all toolboxes and tools
        topic = f"{config['mqtt']['listener']['root']}/+/+/{config['mqtt']['listener']['trace']}"
        client.subscribe(topic)
        print(f"Subscribed to {topic}")

    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload)
            print(f"\nTrace Message on {msg.topic}:")
            st = find_in_dict(data, 'SourceTimestamp')
            print(f"Source Timestamp: {st[0]}")
            # Convert to datetime object
            dt = datetime.datetime.strptime(st[0], "%Y-%m-%dT%H:%M:%SZ")
            # Convert to Unix timestamp (seconds since epoch)
            unix_timestamp = dt.timestamp()
            print(f"Unix Timestamp: {unix_timestamp}")
            print(f"ToolboxId: {msg.topic.split('/')[1]}")
            print(f"ToolId: {msg.topic.split('/')[2]}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {msg.topic}")

    client = mqtt.Client("trace_listener")
    client.on_connect = on_connect
    client.on_message = on_message
    return client

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", type=str, help="YAML configuration file", default="mqtt_conf.yaml")
    args = parser.parse_args()

    # Load configuration
    with open(args.conf, 'r') as f:
        config = yaml.safe_load(f)

    # Create and start Result listener
    result_client = create_result_listener()
    result_client.connect(config['mqtt']['broker']['host'], 
                         config['mqtt']['broker']['port'])
    result_client.loop_start()

    # Create and start Trace listener
    trace_client = create_trace_listener()
    trace_client.connect(config['mqtt']['broker']['host'], 
                        config['mqtt']['broker']['port'])
    trace_client.loop_start()

    try:
        # Keep the script running
        while True:
            pass
    except KeyboardInterrupt:
        print("\nShutting down listeners...")
        result_client.loop_stop()
        trace_client.loop_stop()