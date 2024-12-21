import paho.mqtt.client as mqtt
import yaml
import json
import time
import statistics
import datetime


# TOPIC = "test/topic"


# # Callback when connection is established
# def on_connect(client, userdata, flags, rc):
#     print("Connected with result code "+str(rc))
#     # Subscribe to a topic
#     client.subscribe(TOPIC)

# # Callback when a message is received
# def on_message(client, userdata, msg):
#     # print(f"Received message on {msg.topic}")
#     t = time.localtime()
#     current_time = time.strftime("%H:%M:%S", t)
#     print(f"[{current_time}]: Received MQTT drilling data on {msg.topic}")
#     decoded = msg.payload.decode("utf-8")
#     depth = depth_est_ml_mqtt(json.loads(decoded))
#     print(f"Entry, Exit: {depth}")

# # Create client
# client = mqtt.Client()

# # Set callbacks
# client.on_connect = on_connect
# client.on_message = on_message

# # Connect to broker
# client.connect("localhost", 1883, 60)

# # Keep listening
# client.loop_forever()


# #    host: "OPCPUBSUB"
#     toolboxid: "ILLL502033771"
#     toolid: "setitectest"
#     result: "ResultManagement"
#     trace: "ResultManagement/Trace"

import paho.mqtt.client as mqtt
import yaml
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
import pandas as pd

from abyss.uos_depth_est_core import (
    depth_est_ml_mqtt
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


class DrillingDataAnalyser:
    def __init__(self, config_path='mqtt_conf.yaml'):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # MQTT connection parameters
        self.broker = self.config['mqtt']['broker']
        
    def create_mqtt_client(self, client_id):
        """Create MQTT client with basic configuration"""
        client = mqtt.Client(client_id)
        
        # Optional authentication
        if self.broker.get('username') and self.broker.get('password'):
            client.username_pw_set(
                self.broker['username'], 
                self.broker['password']
            )
        
        return client
    

    def create_result_listener(self):
        """Create an MQTT client for listening to Result data"""
        def on_connect(client, userdata, flags, rc):
            print("Result Listener Connected with result code " + str(rc))
            # Subscribe to all result topics under all toolboxes and tools
            topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['result']}"
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
    
    def create_trace_listener(self):
        """Create an MQTT client for listening to Trace data"""
        def on_connect(client, userdata, flags, rc):
            print("Trace Listener Connected with result code " + str(rc))
            # Subscribe to all trace topics under all toolboxes and tools
            topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener']['trace']}"
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


    def run(self):
        """Main method to set up MQTT clients and start listening"""
        # Create clients
        # result_client = self.create_mqtt_client('result_listener')
        # trace_client = self.create_mqtt_client('trace_listener')
        # analysis_client = self.create_mqtt_client('analysis_publisher')

        result_client = self.create_result_listener()
        trace_client = self.create_trace_listener()

        result_client.connect(self.config['mqtt']['broker']['host'], 
                                self.config['mqtt']['broker']['port'])
        # result_client.loop_start()
        

        trace_client.connect(self.config['mqtt']['broker']['host'], 
                                self.config['mqtt']['broker']['port'])
        # trace_client.loop_start()
        

        try:
            result_client.loop_forever()
            trace_client.loop_forever()
        except KeyboardInterrupt:
            print("\nStopping MQTT clients...")
            result_client.disconnect()
            trace_client.disconnect()
            # analysis_client.disconnect()

    #     def process_message(message):
    #         """Process incoming message"""
    #         payload = json.loads(message.payload.decode('utf-8'))
    #         timestamp = payload['Messages']['Timestamp']
    #         return timestamp, payload['Messages']['Payload']

    #     def on_result_message(client, userdata, message):
    #         """Handle incoming result data"""
    #         try:
    #             timestamp, result = process_message(message)
    #             self.result_data[timestamp] = result
    #             self.check_and_analyze()
    #         except Exception as e:
    #             print(f"Result data error: {e}")

    #     def on_trace_message(client, userdata, message):
    #         """Handle incoming trace data"""
    #         try:
    #             timestamp, trace = process_message(message)
    #             self.trace_data[timestamp] = trace
    #             self.check_and_analyze()
    #         except Exception as e:
    #             print(f"Trace data error: {e}")

    #     def on_connect(client, userdata, flags, rc):
    #         """Connection callback"""
    #         print(f"Connected with result code {rc}")
    #         client.subscribe(
    #             self.config['mqtt']['listener']['topic'], 
    #             qos=1
    #         )

    #     # Set up callbacks
    #     torque_client.on_message = on_torque_message
    #     position_client.on_message = on_position_message
    #     torque_client.on_connect = on_connect
    #     position_client.on_connect = on_connect

    #     # Connect clients
    #     torque_client.connect(
    #         self.broker['host'], 
    #         self.broker['port']
    #     )
    #     position_client.connect(
    #         self.broker['host'], 
    #         self.broker['port']
    #     )
    #     analysis_client.connect(
    #         self.broker['host'], 
    #         self.broker['port']
    #     )

    #     def check_and_analyze(self):
    #         """Check for matching timestamps and perform analysis"""
    #         common_timestamps = set(self.torque_data.keys()) & set(self.position_data.keys())
            
    #         for timestamp in common_timestamps:
    #             # Ensure data arrays are of equal length
    #             if (len(self.torque_data[timestamp]) == len(self.position_data[timestamp])):
    #                 # Perform analysis
    #                 analysis_result = self.analyze_data(
    #                     self.torque_data[timestamp], 
    #                     self.position_data[timestamp]
    #                 )
                    
    #                 # Publish analysis
    #                 analysis_client.publish(
    #                     self.config['channels']['analysis_output']['topic'],
    #                     payload=json.dumps(analysis_result),
    #                     qos=1
    #                 )
                    
    #                 # Optional: print analysis
    #                 print("Analysis Result:", json.dumps(analysis_result, indent=2))
                    
    #                 # Clean up processed data
    #                 del self.torque_data[timestamp]
    #                 del self.position_data[timestamp]
        
    #     # Bind method to instance
    #     self.check_and_analyze = check_and_analyze.__get__(self)

    #     # Start listening
    #     try:
    #         torque_client.loop_forever()
    #     except KeyboardInterrupt:
    #         print("\nStopping MQTT clients...")
    #         torque_client.disconnect()
    #         position_client.disconnect()
    #         analysis_client.disconnect()

def main():
    analyzer = DrillingDataAnalyser()
    analyzer.run()

if __name__ == "__main__":
    main()