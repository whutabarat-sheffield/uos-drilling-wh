#!/usr/bin/env python3
import paho.mqtt.client as mqtt
import json
import time

def on_message(client, userdata, msg):
    """Print full message to see structure"""
    try:
        data = json.loads(msg.payload)
        # Check for signal ID
        signal_id = None
        if '_signal_id' in data:
            signal_id = data['_signal_id']
        elif 'ResultManagement' in data:
            results = data.get('ResultManagement', {}).get('Results', [])
            if results and '_signal_id' in results[0]:
                signal_id = results[0]['_signal_id']
        elif 'AssetManagement' in data and '_signal_id' in data['AssetManagement']:
            signal_id = data['AssetManagement']['_signal_id']
        
        print(f"\n=== Message on {msg.topic} ===")
        print(f"Signal ID: {signal_id}")
        print(f"Keys: {list(data.keys())}")
        if 'ResultManagement' in data and 'Results' in data['ResultManagement']:
            if data['ResultManagement']['Results']:
                print(f"Result keys: {list(data['ResultManagement']['Results'][0].keys())}")
                
    except Exception as e:
        print(f"Error parsing: {e}")

client = mqtt.Client(client_id="test_subscriber", protocol=mqtt.MQTTv311)
client.on_message = on_message
client.connect("mqtt-broker", 1883)
client.subscribe("#")
print("Listening for messages...")
client.loop_forever()