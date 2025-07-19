#!/usr/bin/env python3
import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    print(f"Received on {msg.topic}: {msg.payload[:100]}...")

client = mqtt.Client()
client.on_message = on_message
client.connect("mqtt-broker", 1883)
client.subscribe("#")
print("Listening for messages...")
client.loop_forever()