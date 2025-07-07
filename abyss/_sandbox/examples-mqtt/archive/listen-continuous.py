import paho.mqtt.client as mqtt
import json
import time
from abyss.uos_depth_est_core import (
    depth_est_ml_mqtt
    )

# Callback when connection is established
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribe to a topic
    client.subscribe("test/topic")

# Callback when a message is received
def on_message(client, userdata, msg):
    # print(f"Received message on {msg.topic}")
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(f"[{current_time}]: Received MQTT drilling data on {msg.topic}")
    decoded = msg.payload.decode("utf-8")
    depth = depth_est_ml_mqtt(json.loads(decoded))
    print(f"Entry, Exit: {depth}")

# Create client
client = mqtt.Client()

# Set callbacks
client.on_connect = on_connect
client.on_message = on_message

# Connect to broker
client.connect("localhost", 1883, 60)

# Keep listening
client.loop_forever()