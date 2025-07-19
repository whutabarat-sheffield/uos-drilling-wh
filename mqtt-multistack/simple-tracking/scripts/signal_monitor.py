#!/usr/bin/env python3
"""Minimal MQTT subscriber that logs received signal IDs to CSV"""
import csv
import json
import os
import time
import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    """Extract and log signal IDs from received messages"""
    try:
        data = json.loads(msg.payload)
        
        # Look for signal ID at top level (where we inject it)
        signal_id = data.get('_signal_id')
        
        if signal_id:
            with open('/tracking/received_signals.csv', 'a', newline='') as f:
                csv.writer(f).writerow([signal_id, time.time(), msg.topic])
            print(f"Tracked signal: {signal_id}")
    except:
        pass  # Ignore messages without tracking or parse errors

# Setup and run
client = mqtt.Client()
client.on_message = on_message
client.connect(os.getenv('MQTT_HOST', 'mqtt-broker'), 1883)
client.subscribe("#")  # Subscribe to all topics
print("Signal monitor started - tracking to received_signals.csv")
client.loop_forever()