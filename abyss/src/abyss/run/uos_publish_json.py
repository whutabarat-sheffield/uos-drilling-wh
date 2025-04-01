import paho.mqtt.client as mqtt
import time
import json
import argparse
import os
import yaml
import random

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


def main():
    # Initialise parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("path", type=str, help="Path to the data folder")
    parser.add_argument("-c", "--conf", type=str, help="YAML configuration file", default="mqtt_conf.yaml")

    # Read arguments
    args = parser.parse_args()


    # config = yaml.safe_load(open("mqtt_conf.yaml"))
    config = yaml.safe_load(open(args.conf))

    def publish(topic, payload, timestamp0, timestamp1) -> None:
        payload = payload.replace(timestamp0, timestamp1)
        d = json.loads(payload)
        client.publish(topic, json.dumps(d))
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(f"[{current_time}]: Publish data on {topic} '{timestamp1}'")

    # Sample data folders, these are the new datasets
    # DATA_FOLDERS = ['data0', 'data1']
    DATA_FOLDERS = [args.path]

    # MQTT topics for publishing
    TOPIC_RESULT = config['mqtt']['publisher']['topics']['result']
    TOPIC_TRACE = config['mqtt']['publisher']['topics']['trace']

    # Made up toolboxids and toolids for simulating multiple tools
    TOOLBOXIDS = ['ILLL502033771', 'ILLL502033772', 'ILLL502033773', 'ILLL502033774', 'ILLL502033775']
    TOOLIDS = ['setitec001', 'setitec002', 'setitec003', 'setitec004', 'setitec005']

    # Create client
    client = mqtt.Client()
    client.connect(config['mqtt']['broker']['host'], 
                config['mqtt']['broker']['port'])

    # Publish data continuously
    while True:
        # Select a random data folder to publish and read the data
        random.shuffle(DATA_FOLDERS)
        data_folder = DATA_FOLDERS[0]
        with open (f'{os.getcwd()}\\{data_folder}\\ResultManagement.json') as f:
            d_result = f.read()
            source_timestamps = find_in_dict(dict(json.loads(d_result)), 'SourceTimestamp')
            assert len(set(source_timestamps)) == 1
        with open (f'{os.getcwd()}\\{data_folder}\\Trace.json') as f:
            d_trace = f.read()
            source_timestamps = find_in_dict(dict(json.loads(d_result)), 'SourceTimestamp')
            assert len(set(source_timestamps)) == 1

        # Update the source timestamp        
        original_source_timestamp = source_timestamps[0]
        new_source_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())
        
        
        # Prepare a shuffled topic set
        random.shuffle(TOOLBOXIDS)
        random.shuffle(TOOLIDS)
        
        # Prepare the topics
        topic_result = f"{config['mqtt']['listener']['root']}/{TOOLBOXIDS[0]}/{TOOLIDS[0]}/{config['mqtt']['listener']['result']}"
        topic_trace = f"{config['mqtt']['listener']['root']}/{TOOLBOXIDS[0]}/{TOOLIDS[0]}/{config['mqtt']['listener']['trace']}"

        # Shuffle the data order
        list_to_publish = [(topic_result, d_result), (topic_trace, d_trace)]
        random.shuffle(list_to_publish)

        # Publish the data
        for item in list_to_publish:
            publish(item[0], item[1], original_source_timestamp, new_source_timestamp)
            time.sleep(random.randint(1, 3))

if __name__ == "__main__":
    main()