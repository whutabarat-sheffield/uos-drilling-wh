#!/usr/bin/env python3
"""Minimal MQTT subscriber that logs received signal IDs to CSV - with debug output"""
import csv
import json
import os
import time
import paho.mqtt.client as mqtt

message_count = 0

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("#")
    print("Subscribed to all topics (#)")

def on_message(client, userdata, msg):
    """Extract and log signal IDs from received messages"""
    global message_count
    message_count += 1
    
    # Debug: show first few messages
    if message_count <= 5:
        print(f"Message {message_count} on {msg.topic}: {msg.payload[:100]}...")
    
    try:
        data = json.loads(msg.payload)
        signal_id = None
        
        # Try different locations for signal ID
        if '_signal_id' in data:
            signal_id = data['_signal_id']
        elif 'ResultManagement' in data:
            results = data.get('ResultManagement', {}).get('Results', [])
            if results and '_signal_id' in results[0]:
                signal_id = results[0]['_signal_id']
        elif 'AssetManagement' in data and '_signal_id' in data['AssetManagement']:
            signal_id = data['AssetManagement']['_signal_id']
        
        if signal_id:
            with open('/tracking/received_signals.csv', 'a', newline='') as f:
                csv.writer(f).writerow([signal_id, time.time(), msg.topic])
            print(f"Tracked signal: {signal_id}")
    except Exception as e:
        if message_count <= 10:
            print(f"Error processing message: {e}")

# Setup and run
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(os.getenv('MQTT_HOST', 'mqtt-broker'), 1883)
print(f"Connecting to {os.getenv('MQTT_HOST', 'mqtt-broker')}:1883")
client.loop_forever()