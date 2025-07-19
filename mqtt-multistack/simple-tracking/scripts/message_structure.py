#!/usr/bin/env python3
import paho.mqtt.client as mqtt
import json

def on_message(client, userdata, msg):
    """Print message structure"""
    try:
        data = json.loads(msg.payload)
        print(f"\n=== Message on {msg.topic} ===")
        print(f"Top-level keys: {list(data.keys())}")
        
        # Check Messages array
        if 'Messages' in data and data['Messages']:
            print(f"Messages[0] keys: {list(data['Messages'][0].keys())}")
            
            # Check for nested structures
            for key in ['ResultManagement', 'AssetManagement']:
                if key in data['Messages'][0]:
                    print(f"Messages[0]['{key}'] keys: {list(data['Messages'][0][key].keys())}")
                    if 'Results' in data['Messages'][0][key]:
                        print(f"Messages[0]['{key}']['Results'] length: {len(data['Messages'][0][key]['Results'])}")
                        if data['Messages'][0][key]['Results']:
                            print(f"Messages[0]['{key}']['Results'][0] keys: {list(data['Messages'][0][key]['Results'][0].keys())}")
        
        # Stop after first message
        client.disconnect()
                
    except Exception as e:
        print(f"Error parsing: {e}")

client = mqtt.Client(client_id="structure_checker", protocol=mqtt.MQTTv311)
client.on_message = on_message
client.connect("mqtt-broker", 1883)
client.subscribe("#")
print("Waiting for first message...")
client.loop_forever()